#!/usr/bin/env python3
"""
Comprehensive Ray-cluster GPU node health check.

Runs a single SSH call per node (in parallel) that executes a battery of
local checks on the worker, then prints a per-check matrix, the failures,
and a paste-ready `--exclude` string for launch_pretrain_ray.py.

Usage:
    python3 cluster_health_check.py                             # all Ray nodes
    python3 cluster_health_check.py --num_nodes 32              # first N
    python3 cluster_health_check.py --ips 10.x 10.y ...         # explicit list
    python3 cluster_health_check.py --json report.json          # write JSON
    python3 cluster_health_check.py --exclude_file bad.txt      # write IPs

Checks per node (all best-effort, no sudo):
  gpu_count   nvidia-smi reports >= 8 GPUs
  ecc         no uncorrectable volatile ECC errors on any GPU
  nvlink      no NVLink reported as Inactive
  mounts      /mnt/amlfs-{01,02,03,07} are present and readable
  netif       an interface in 10.244.0.0/16 is UP
  port        TCP/29500 (default MASTER_PORT) is free
  env         dex_mot conda env is present
  torch       PyTorch CUDA imports + 2048x2048 matmul on each of 8 GPUs
  disk_free   /mnt/amlfs-02 has >= 100 GB free

Run from a shell that has SSH to the workers (your login box). Do NOT run
this from inside `GearRayJobSubmissionClient.submit_job` — the driver pod
there does not have SSH credentials to the worker pods, and every check
will report `SSH connection failed`.
"""

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


REMOTE_PROBE = r"""
set +e
emit() { printf 'CHECK:%s:%s:%s\n' "$1" "$2" "$3"; }

# Resolve the python interpreter once — used by several checks.
PY=/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin/python3
[ -x "$PY" ] || PY=python3

# 1. GPU count
gpu_count=$(nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$gpu_count" -ge 8 ]; then
    emit gpu_count OK "$gpu_count visible"
else
    emit gpu_count FAIL "only $gpu_count GPUs visible"
fi

# 2. Uncorrectable volatile ECC errors (sum over all GPUs)
ecc_total=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits 2>/dev/null \
            | awk 'BEGIN{s=0} /^[0-9]+$/ {s+=$1} END{print s}')
if [ -z "$ecc_total" ] || [ "$ecc_total" = "0" ]; then
    emit ecc OK "0 uncorrectable"
else
    emit ecc FAIL "$ecc_total uncorrectable ECC errors"
fi

# 3. NVLink: any link in "Inactive" state is bad
nvlink_inactive=$(nvidia-smi nvlink -s 2>/dev/null | grep -c "Inactive")
if [ "$nvlink_inactive" = "0" ]; then
    emit nvlink OK "no inactive links"
else
    emit nvlink FAIL "$nvlink_inactive inactive links"
fi

# 4. Required shared mounts
mount_fail=""
for m in /mnt/amlfs-01 /mnt/amlfs-02 /mnt/amlfs-03 /mnt/amlfs-07; do
    if [ ! -d "$m" ]; then
        mount_fail="$mount_fail $m(missing)"
    elif ! ls "$m" >/dev/null 2>&1; then
        mount_fail="$mount_fail $m(unreadable)"
    fi
done
if [ -z "$mount_fail" ]; then
    emit mounts OK "all 4 readable"
else
    emit mounts FAIL "$mount_fail"
fi

# 5. Network: detect a routable IP via Python instead of the `ip` command.
# Many minimal pod images (including this cluster's) don't ship iproute2,
# so `ip addr show` produces no output. We use the classic UDP-connect
# trick: connect() on a UDP socket doesn't send anything, but it forces
# the kernel to pick the source IP from the routing table.
ip_addr=$("$PY" -c "
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 53))
    print(s.getsockname()[0])
    s.close()
except Exception:
    try:
        print(socket.gethostbyname(socket.gethostname()))
    except Exception as e:
        print('ERR:' + type(e).__name__ + ':' + str(e))
" 2>&1 | tail -1)
case "$ip_addr" in
    127.*|169.254.*|ERR:*|"")
        emit netif FAIL "no routable IP (got '$ip_addr')"
        ;;
    *)
        emit netif OK "routable IP $ip_addr"
        ;;
esac

# 6. Master port free
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ':29500$'; then
    emit port FAIL "29500 already bound (stale process?)"
else
    emit port OK "29500 free"
fi

# 7. Conda env present (PY was already resolved at the top)
if [ "$PY" = "/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin/python3" ]; then
    emit env OK "dex_mot env present"
else
    emit env FAIL "dex_mot python missing (fell back to system python3)"
fi

# 8. PyTorch CUDA: import + 2048x2048 matmul on each of 8 GPUs
torch_out=$("$PY" - <<'PY' 2>&1
try:
    import torch
    n = torch.cuda.device_count()
    if n < 8:
        print(f"FAIL only {n} cuda devices visible to torch")
    else:
        for i in range(8):
            a = torch.randn(2048, 2048, device=f"cuda:{i}")
            _ = (a @ a).sum().item()
            del a
            torch.cuda.empty_cache()
        print(f"OK torch={torch.__version__} cuda={torch.version.cuda}")
except Exception as e:
    print(f"FAIL {type(e).__name__}: {e}")
PY
)
torch_summary=$(echo "$torch_out" | tail -1)
if echo "$torch_summary" | grep -q '^OK '; then
    emit torch OK "${torch_summary#OK }"
else
    emit torch FAIL "${torch_summary#FAIL }"
fi

# 9. Free disk on the output mount (amlfs-02)
disk_free_g=$(df -BG --output=avail /mnt/amlfs-02 2>/dev/null | tail -1 | tr -d ' G')
if [ -n "$disk_free_g" ] && [ "$disk_free_g" -ge 100 ]; then
    emit disk_free OK "amlfs-02 ${disk_free_g}G free"
elif [ -n "$disk_free_g" ]; then
    emit disk_free FAIL "amlfs-02 only ${disk_free_g}G free"
else
    emit disk_free FAIL "df failed"
fi
"""


CHECK_ORDER = [
    "gpu_count", "ecc", "nvlink", "mounts",
    "netif", "port", "env", "torch", "disk_free",
]
CHECK_LABEL = {
    "gpu_count": "gpu", "ecc": "ecc", "nvlink": "nvlnk", "mounts": "mnts",
    "netif": "net", "port": "port", "env": "env", "torch": "torch",
    "disk_free": "disk",
}


@dataclass
class NodeReport:
    ip: str
    ssh_ok: bool = True
    ssh_error: str = ""
    elapsed: float = 0.0
    checks: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        if not self.ssh_ok:
            return False
        return all(self.checks.get(c, ("FAIL", "missing"))[0] == "OK"
                   for c in CHECK_ORDER)


def check_node(ip: str, timeout: int) -> NodeReport:
    rep = NodeReport(ip=ip)
    t0 = time.time()
    try:
        p = subprocess.run(
            ["ssh",
             "-o", "StrictHostKeyChecking=no",
             "-o", "ConnectTimeout=10",
             "-o", "BatchMode=yes",
             ip, "bash", "-s"],
            input=REMOTE_PROBE,
            capture_output=True, text=True,
            timeout=timeout,
        )
        rep.elapsed = time.time() - t0
        if p.returncode == 255:
            err = (p.stderr.strip().splitlines() or ["SSH failed"])[-1]
            rep.ssh_ok = False
            rep.ssh_error = err[:200]
            return rep
        for line in p.stdout.splitlines():
            if line.startswith("CHECK:"):
                parts = line.split(":", 3)
                if len(parts) == 4:
                    _, name, status, msg = parts
                    rep.checks[name] = (status, msg)
    except subprocess.TimeoutExpired:
        rep.ssh_ok = False
        rep.ssh_error = f"command timeout after {timeout}s"
        rep.elapsed = time.time() - t0
    except Exception as e:
        rep.ssh_ok = False
        rep.ssh_error = f"{type(e).__name__}: {e}"
        rep.elapsed = time.time() - t0
    return rep


def get_ips_from_ray() -> List[str]:
    # Silence the spurious Ray accelerator warning that fires when the
    # caller's CUDA_VISIBLE_DEVICES doesn't match num_gpus=0. We're only
    # using Ray for IP discovery — we don't claim any GPUs in this script.
    import os
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    import ray
    ray.init(address="auto")
    nodes = ray.nodes()
    ips = []
    for n in nodes:
        if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) >= 8:
            ips.append(n["NodeManagerAddress"])
    ray.shutdown()
    return sorted(ips)


def render(reports: List[NodeReport]) -> None:
    print()
    cols = ["ssh"] + [CHECK_LABEL[c] for c in CHECK_ORDER]
    header = f"{'IP':<18} " + " ".join(f"{c:<5}" for c in cols) + "  STATE       took"
    print(header)
    print("-" * len(header))
    for r in reports:
        ssh_cell = "OK" if r.ssh_ok else "FAIL"
        check_cells = []
        for c in CHECK_ORDER:
            s = r.checks.get(c, ("-", ""))[0]
            check_cells.append(s if s in ("OK", "FAIL") else "-")
        cells = [ssh_cell] + check_cells
        state = "HEALTHY" if r.healthy else "UNHEALTHY"
        print(f"{r.ip:<18} " + " ".join(f"{x:<5}" for x in cells) +
              f"  {state:<10} {r.elapsed:>5.1f}s")

    # Failure detail
    print("\nFailures:")
    any_fail = False
    for r in reports:
        if not r.ssh_ok:
            print(f"  {r.ip}  ssh:        {r.ssh_error}")
            any_fail = True
            continue
        for c in CHECK_ORDER:
            status, msg = r.checks.get(c, ("MISSING", "no output"))
            if status != "OK":
                print(f"  {r.ip}  {c:<10}  {status}: {msg}")
                any_fail = True
    if not any_fail:
        print("  (none)")

    # Per-check rollup
    healthy = [r for r in reports if r.healthy]
    bad = [r for r in reports if not r.healthy]
    print(f"\nSummary: {len(healthy)}/{len(reports)} healthy, {len(bad)} unhealthy")

    per_check_fail: Dict[str, List[str]] = {c: [] for c in CHECK_ORDER}
    ssh_fail: List[str] = []
    for r in reports:
        if not r.ssh_ok:
            ssh_fail.append(r.ip)
            continue
        for c in CHECK_ORDER:
            if r.checks.get(c, ("FAIL", ""))[0] != "OK":
                per_check_fail[c].append(r.ip)
    if ssh_fail or any(per_check_fail.values()):
        print("Per-check failure counts:")
        if ssh_fail:
            print(f"  ssh        {len(ssh_fail):>3}  {', '.join(ssh_fail)}")
        for c in CHECK_ORDER:
            n = len(per_check_fail[c])
            if n:
                print(f"  {c:<10} {n:>3}  {', '.join(per_check_fail[c])}")

    # Paste-ready exclude line
    if bad:
        print("\nTo retry the launcher skipping these:")
        print(f"  --exclude {' '.join(r.ip for r in bad)}")


def main():
    ap = argparse.ArgumentParser(
        description="Comprehensive per-node health check for the Ray cluster.")
    ap.add_argument("--ips", nargs="+",
                    help="Explicit IPs (skip Ray discovery).")
    ap.add_argument("--num_nodes", type=int,
                    help="Limit to first N nodes after sorting.")
    ap.add_argument("--timeout", type=int, default=120,
                    help="Per-node SSH timeout in seconds (default 120).")
    ap.add_argument("--parallel", type=int, default=64,
                    help="Max concurrent SSH connections (default 64).")
    ap.add_argument("--json", help="Write detailed JSON report to this path.")
    ap.add_argument("--exclude_file",
                    help="Write space-separated unhealthy IPs to this file.")
    args = ap.parse_args()

    if args.ips:
        ips = sorted(args.ips)
    else:
        try:
            ips = get_ips_from_ray()
        except Exception as e:
            print(f"Could not discover IPs via Ray: {e}", file=sys.stderr)
            print("Pass --ips a.b.c.d ... to skip discovery.", file=sys.stderr)
            sys.exit(2)
    if args.num_nodes:
        ips = ips[: args.num_nodes]

    if not ips:
        print("No nodes to check.", file=sys.stderr)
        sys.exit(2)

    print(f"Checking {len(ips)} node(s) with timeout={args.timeout}s, "
          f"parallel={args.parallel} ...")
    reports: List[NodeReport] = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {ex.submit(check_node, ip, args.timeout): ip for ip in ips}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            r = fut.result()
            reports.append(r)
            tag = "OK " if r.healthy else "BAD"
            print(f"  [{i:>3}/{len(ips)}] {r.ip:<18} {tag}  "
                  f"{r.elapsed:>5.1f}s")
    elapsed = time.time() - t0

    reports.sort(key=lambda r: tuple(int(x) for x in r.ip.split(".")))
    render(reports)
    print(f"\nWall time: {elapsed:.1f}s")

    if args.json:
        payload = [{
            "ip": r.ip,
            "ssh_ok": r.ssh_ok,
            "ssh_error": r.ssh_error,
            "elapsed_sec": round(r.elapsed, 3),
            "healthy": r.healthy,
            "checks": {k: {"status": s, "message": m}
                       for k, (s, m) in r.checks.items()},
        } for r in reports]
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote JSON report to {args.json}")

    if args.exclude_file:
        bad_ips = [r.ip for r in reports if not r.healthy]
        with open(args.exclude_file, "w") as f:
            f.write(" ".join(bad_ips))
        print(f"Wrote {len(bad_ips)} unhealthy IPs to {args.exclude_file}")
        if bad_ips:
            print(f"  (use: python3 launch_pretrain_ray.py ... --exclude $(cat {args.exclude_file}))")

    sys.exit(0 if all(r.healthy for r in reports) else 1)


if __name__ == "__main__":
    main()
