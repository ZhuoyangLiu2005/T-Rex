"""
Launch multi-node pretraining using Ray cluster for node discovery + SSH for execution.

Usage:
    python launch_pretrain_ray.py                  # use all Ray nodes
    python launch_pretrain_ray.py --num_nodes 4    # use first 4 nodes

Ray is only used to discover node IPs. Training is launched via SSH so that
NCCL/accelerate run natively on each node (no Ray subprocess overhead).
"""

import ray
import argparse
import subprocess
import signal
import sys

# ─── Training config ───
TRAIN_SCRIPT = "/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen/scripts/pretrain_egodex.sh"
TRAIN_PY_SCRIPT = "train_qwen3vl_pretrain_egodex.py"
MASTER_PORT = 29500


def kill_stale_processes(ips):
    """SSH into each node and kill leftover training processes from previous runs."""
    print("Cleaning up stale processes on all nodes...")
    kill_cmds = [
        f"pkill -9 -f '{TRAIN_PY_SCRIPT}'",
        f"pkill -9 -f 'accelerate launch.*{TRAIN_PY_SCRIPT}'",
        f"pkill -9 -f 'deepspeed.*{TRAIN_PY_SCRIPT}'",
        # Kill anything holding the NCCL port
        f"fuser -k {MASTER_PORT}/tcp",
    ]
    kill_cmd = " ; ".join(kill_cmds) + " ; true"  # always exit 0

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
    """Get IPs of all alive Ray worker nodes (excluding head if it has no GPU)."""
    ray.init(address="auto")
    nodes = ray.nodes()
    ips = []
    for node in nodes:
        if not node["Alive"]:
            continue
        # Only include nodes with GPUs
        gpus = node["Resources"].get("GPU", 0)
        if gpus >= 8:
            ip = node["NodeManagerAddress"]
            ips.append(ip)
    ray.shutdown()
    return sorted(ips)


def launch_on_nodes(ips, num_nodes=None):
    """SSH into each node and launch the training script."""
    if num_nodes is not None:
        ips = ips[:num_nodes]

    num_machines = len(ips)
    master_addr = ips[0]

    print(f"Launching on {num_machines} nodes:")
    for i, ip in enumerate(ips):
        print(f"  rank {i}: {ip}")
    print(f"Master: {master_addr}:{MASTER_PORT}")
    print()

    # Environment overrides passed to the script
    env_exports = (
        f"export MASTER_ADDR={master_addr}; "
        f"export MASTER_PORT={MASTER_PORT}; "
        f"export NUM_MACHINES={num_machines}; "
    )

    processes = []
    for rank, ip in enumerate(ips):
        cmd = (
            f"ssh -o StrictHostKeyChecking=no {ip} "
            f"'bash -l -c \""
            f"{env_exports} "
            f"export MACHINE_RANK={rank}; "
            f"bash {TRAIN_SCRIPT}"
            f"\"'"
        )
        print(f"[rank {rank}] Launching on {ip} ...")
        proc = subprocess.Popen(cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr)
        processes.append((rank, ip, proc))

    # Wait for all and handle Ctrl+C
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
    parser.add_argument("--num_nodes", type=int, default=None,
                        help="Number of nodes to use (default: all available)")
    parser.add_argument("--list_only", action="store_true",
                        help="Only print node IPs, don't launch training")
    parser.add_argument("--kill_only", action="store_true",
                        help="Only kill stale processes on all nodes, don't launch")
    args = parser.parse_args()

    ips = get_ray_node_ips()
    print(f"Found {len(ips)} GPU nodes in Ray cluster:")
    for ip in ips:
        print(f"  {ip}")
    print()

    if args.list_only:
        sys.exit(0)

    if args.kill_only:
        kill_stale_processes(ips)
        sys.exit(0)

    # Always clean up before launching
    kill_stale_processes(ips)
    launch_on_nodes(ips, num_nodes=args.num_nodes)
