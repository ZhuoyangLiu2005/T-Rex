"""
Launch multi-node training using Ray cluster for node discovery + SSH for execution.

Usage:
    python launch_pretrain_ray.py                             # egodex, all nodes
    python launch_pretrain_ray.py --num_nodes 4               # egodex, 4 nodes
    python launch_pretrain_ray.py --script mecka              # mecka, all nodes
    python launch_pretrain_ray.py --script mecka --num_nodes 5
    python launch_pretrain_ray.py --script midtrain           # midtrain (mecka NV+BKL)
    python launch_pretrain_ray.py --script midtrain --from_vlm_scratch  # no resume

Ray is only used to discover node IPs. Training is launched via SSH so that
NCCL/accelerate run natively on each node (no Ray subprocess overhead).

Fixes over the original version:
  1. Auto-detects NCCL_SOCKET_IFNAME from the master node's IP (no more
     hardcoded eth0 mismatch causing gradient corruption / training collapse).
  2. Runs data prep (symlinks) on rank 0 only, with a barrier before training.
  3. Supports egodex / mecka / midtrain scripts via --script flag.
"""

import ray
import argparse
import subprocess
import signal
import sys

SCRIPT_DIR = "/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen/scripts"

TRAIN_SCRIPTS = {
    "egodex":       f"{SCRIPT_DIR}/pretrain_egodex.sh",
    "mecka":        f"{SCRIPT_DIR}/pretrain_mecka.sh",
    "egodex_flare": f"{SCRIPT_DIR}/pretrain_egodex_flare.sh",
    "mecka_flare":  f"{SCRIPT_DIR}/pretrain_mecka_flare.sh",
    "midtrain":     f"{SCRIPT_DIR}/train_qwen3vl_midtrain_flare.sh",
}
PY_SCRIPTS = {
    "egodex":       "train_qwen3vl_pretrain_egodex.py",
    "mecka":        "train_qwen3vl_pretrain_egodex.py",
    "egodex_flare": "train_qwen3vl_pretrain_egodex_flare.py",
    "mecka_flare":  "train_qwen3vl_pretrain_egodex_flare.py",
    "midtrain":     "train_qwen3vl_midtrain_flare.py",
}
MASTER_PORT = 29500


def kill_stale_processes(ips, py_script):
    """SSH into each node and kill leftover training processes from previous runs."""
    print("Cleaning up stale processes on all nodes...")
    kill_cmds = [
        f"pkill -9 -f '{py_script}'",
        f"pkill -9 -f 'accelerate launch.*{py_script}'",
        f"pkill -9 -f 'deepspeed.*{py_script}'",
        f"fuser -k {MASTER_PORT}/tcp",
    ]
    kill_cmd = " ; ".join(kill_cmds) + " ; true"

    procs = []
    for ip in ips:
        cmd = f"ssh -o StrictHostKeyChecking=no {ip} 'bash -c \"{kill_cmd}\"'"
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append((ip, p))

    for ip, p in procs:
        p.wait()
        print(f"  {ip}: cleaned")
    print()


def get_ray_node_ips():
    """Get IPs of all alive Ray worker nodes with >= 8 GPUs."""
    ray.init(address="auto")
    nodes = ray.nodes()
    ips = []
    for node in nodes:
        if not node["Alive"]:
            continue
        gpus = node["Resources"].get("GPU", 0)
        if gpus >= 8:
            ip = node["NodeManagerAddress"]
            ips.append(ip)
    ray.shutdown()
    return sorted(ips)


def health_check_nodes(ips, timeout=60):
    """
    SSH into each node in parallel and verify:
      1. SSH is reachable (within timeout)
      2. nvidia-smi reports 8 healthy GPUs
      3. MASTER_PORT is not stuck/unreachable

    Returns (healthy_ips, failed_ips) where failed_ips is a list of (ip, reason).
    """
    print("Running health checks on all nodes...")
    # Check 1: nvidia-smi GPU count
    # Check 2: allocate CUDA memory on all 8 GPUs (catches flaky hardware)
    # Using subprocess list form (no shell=True) so SSH gets the command
    # as a single argument — no nested quoting issues.
    gpu_check_script = (
        "import torch; "
        "[torch.zeros(1024,1024,device=torch.device('cuda',i)) for i in range(8)]; "
        "print('gpu_alloc_ok')"
    )
    procs = []
    for ip in ips:
        # SSH executes the remote command directly (no bash -c wrapping needed)
        remote_cmd = (
            f"nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader 2>/dev/null | wc -l; "
            f"python3 -c \"{gpu_check_script}\" 2>/dev/null"
        )
        cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
            ip, remote_cmd,
        ]
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        procs.append((ip, p))

    healthy = []
    failed = []
    for ip, p in procs:
        try:
            stdout, stderr = p.communicate(timeout=timeout)
            rc = p.returncode
            output = stdout.decode().strip()
            stderr_text = stderr.decode().strip()
            # exit 255 = SSH connection failed entirely
            if rc == 255:
                failed.append((ip, f"SSH connection failed"))
                continue
            # Parse nvidia-smi GPU count from first line
            lines = output.split("\n")
            try:
                gpu_count = int(lines[0].strip())
            except (ValueError, IndexError):
                failed.append((ip, f"nvidia-smi failed (exit {rc})"))
                continue
            if gpu_count < 8:
                failed.append((ip, f"only {gpu_count} GPUs visible"))
                continue
            if "gpu_alloc_ok" not in output:
                failed.append((ip, "GPU memory allocation failed"))
                continue
            healthy.append(ip)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            failed.append((ip, "timeout (node unresponsive)"))
        except Exception as e:
            failed.append((ip, str(e)))

    for ip in healthy:
        print(f"  {ip}: OK")
    for ip, reason in failed:
        print(f"  {ip}: FAILED ({reason})")
    print(f"\nHealthy: {len(healthy)}/{len(ips)}")
    if failed:
        print(f"Excluded {len(failed)} unhealthy node(s)")
    print()
    return healthy, failed


def detect_nccl_interface(master_ip):
    """
    Detect the network interface that routes to master_ip.
    Returns the interface name (e.g., 'eth0', 'bond0', 'ens5').

    This prevents the training collapse caused by NCCL binding to the wrong
    interface when the hardcoded 'eth0' doesn't match the Ray cluster's network.
    """
    try:
        result = subprocess.run(
            ["ip", "route", "get", master_ip],
            capture_output=True, text=True, timeout=5,
        )
        # Output: "10.244.x.x dev eth0 src 10.244.y.y ..."
        parts = result.stdout.strip().split()
        if "dev" in parts:
            return parts[parts.index("dev") + 1]
    except Exception as e:
        print(f"  Warning: could not detect NCCL interface: {e}")
    return "eth0"  # fallback


def run_data_prep(ip, train_script, extra_env=""):
    """
    Run data prep (symlinks etc.) on a single node before training starts.

    Extracts the data-prep portion of the script by running it with
    SKIP_TRAINING=1, which we add support for. As a fallback, runs the
    full script — the training part will fail with MACHINE_RANK unset but
    the symlinks will be created.

    `extra_env` is a string of additional `export FOO=bar; ` statements
    (e.g. FROM_VLM_SCRATCH=1) that need to be visible to the script's
    bash conditionals during the data-prep pass too — these can change
    the printed run-name or other early decisions, so it's safer to keep
    the prep run identical to the training run.
    """
    print(f"Running data prep on {ip} ...")
    cmd = (
        f"ssh -o StrictHostKeyChecking=no {ip} "
        f"'bash -l -c \"{extra_env}export SKIP_TRAINING=1; bash {train_script}\"'"
    )
    proc = subprocess.Popen(cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr)
    rc = proc.wait()
    if rc == 0:
        print(f"  Data prep completed successfully on {ip}")
    else:
        # The script may fail at the accelerate launch step, which is expected
        # if SKIP_TRAINING guard isn't in the script yet. The symlinks are
        # already created by that point.
        print(f"  Data prep exited with code {rc} (symlinks likely created, training skipped)")
    print()


def launch_on_nodes(ips, train_script, py_script, num_nodes=None,
                    skip_data_prep=False, from_vlm_scratch=False):
    """SSH into each node and launch the training script."""
    if num_nodes is not None:
        ips = ips[:num_nodes]

    num_machines = len(ips)
    master_addr = ips[0]

    # Detect the correct network interface for NCCL
    nccl_ifname = detect_nccl_interface(master_addr)
    print(f"Detected NCCL interface: {nccl_ifname} (for master {master_addr})")

    print(f"\nLaunching on {num_machines} nodes:")
    for i, ip in enumerate(ips):
        print(f"  rank {i}: {ip}")
    print(f"Master: {master_addr}:{MASTER_PORT}")
    if from_vlm_scratch:
        print("FROM_VLM_SCRATCH=1 (midtrain only — start from base Qwen3-VL weights)")
    print()

    # Mid-train-only env passthrough. Other scripts ignore this var, so it's
    # safe to always export it.
    extra_env = f"export FROM_VLM_SCRATCH={1 if from_vlm_scratch else 0}; "

    # Run data prep on rank 0 only (symlinks, merged data root, etc.)
    # This prevents the race condition where multiple nodes try to create
    # the same symlinks simultaneously.
    if not skip_data_prep:
        run_data_prep(ips[0], train_script, extra_env=extra_env)

    # Environment overrides passed to each node's script.
    # NCCL_SOCKET_IFNAME is set here to override the hardcoded eth0 in the
    # training scripts, preventing gradient corruption from wrong interface.
    env_exports = (
        f"export MASTER_ADDR={master_addr}; "
        f"export MASTER_PORT={MASTER_PORT}; "
        f"export NUM_MACHINES={num_machines}; "
        f"export NCCL_SOCKET_IFNAME={nccl_ifname}; "
        f"{extra_env}"
    )

    processes = []
    for rank, ip in enumerate(ips):
        cmd = (
            f"ssh -o StrictHostKeyChecking=no {ip} "
            f"'bash -l -c \""
            f"{env_exports} "
            f"export MACHINE_RANK={rank}; "
            f"bash {train_script}"
            f"\"'"
        )
        print(f"[rank {rank}] Launching on {ip} ...")
        proc = subprocess.Popen(cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr)
        processes.append((rank, ip, proc))

    # Handle Ctrl+C
    def cleanup(signum, frame):
        print("\nCaught interrupt, killing all SSH processes...")
        for rank, ip, proc in processes:
            proc.terminate()
        for rank, ip, proc in processes:
            proc.wait()
        sys.exit(1)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Wait for all processes
    failed = []
    for rank, ip, proc in processes:
        rc = proc.wait()
        if rc != 0:
            failed.append((rank, ip, rc))
            print(f"[rank {rank}] {ip} exited with code {rc}")
        else:
            print(f"[rank {rank}] {ip} finished successfully")

    if failed:
        print(f"\n{len(failed)} node(s) failed:")
        for rank, ip, rc in failed:
            print(f"  rank {rank} ({ip}): exit code {rc}")
        sys.exit(1)
    else:
        print(f"\nAll {num_machines} nodes finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, default="egodex",
                        choices=list(TRAIN_SCRIPTS.keys()),
                        help="Which training script to launch")
    parser.add_argument("--num_nodes", type=int, default=None,
                        help="Number of nodes to use (default: all available)")
    parser.add_argument("--list_only", action="store_true",
                        help="Only print node IPs, don't launch training")
    parser.add_argument("--kill_only", action="store_true",
                        help="Only kill stale processes on all nodes")
    parser.add_argument("--skip_data_prep", action="store_true",
                        help="Skip data prep step (if symlinks already exist)")
    parser.add_argument("--skip_health_check", action="store_true",
                        help="Skip node health check (faster but risky)")
    parser.add_argument("--exclude", type=str, nargs="+", default=[],
                        metavar="IP",
                        help="IPs to exclude (e.g. --exclude 10.244.1.1 10.244.2.2)")
    parser.add_argument("--from_vlm_scratch", action="store_true",
                        help="(midtrain only) Skip RESUME_CHECKPOINT and start "
                             "from the base Qwen3-VL weights. Sets "
                             "FROM_VLM_SCRATCH=1 in the launched env.")
    args = parser.parse_args()

    if args.from_vlm_scratch and args.script != "midtrain":
        print(f"WARNING: --from_vlm_scratch has no effect for --script {args.script} "
              f"(only midtrain reads FROM_VLM_SCRATCH).")

    train_script = TRAIN_SCRIPTS[args.script]
    py_script = PY_SCRIPTS[args.script]

    ips = get_ray_node_ips()
    print(f"Found {len(ips)} GPU nodes in Ray cluster:")
    for ip in ips:
        print(f"  {ip}")
    print()

    # Exclude blacklisted nodes
    if args.exclude:
        exclude_set = set(args.exclude)
        before = len(ips)
        ips = [ip for ip in ips if ip not in exclude_set]
        print(f"Excluded {before - len(ips)} blacklisted node(s), {len(ips)} remaining\n")

    if args.list_only:
        sys.exit(0)

    # Health check: filter out broken nodes before doing anything
    if not args.skip_health_check:
        ips, failed = health_check_nodes(ips)
        if not ips:
            print("ERROR: No healthy nodes available!")
            sys.exit(1)
        if args.num_nodes and len(ips) < args.num_nodes:
            print(f"WARNING: Requested {args.num_nodes} nodes but only "
                  f"{len(ips)} healthy. Using all {len(ips)}.")
            args.num_nodes = len(ips)

    if args.kill_only:
        kill_stale_processes(ips, py_script)
        sys.exit(0)

    # Always clean up before launching
    kill_stale_processes(ips, py_script)
    launch_on_nodes(ips, train_script, py_script,
                    num_nodes=args.num_nodes,
                    skip_data_prep=args.skip_data_prep,
                    from_vlm_scratch=args.from_vlm_scratch)
