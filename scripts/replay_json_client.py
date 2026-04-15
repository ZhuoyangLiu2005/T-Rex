"""
Replay pre-recorded actions from a training .json file via ZeroMQ.

Drop-in replacement for the VLA inference server (test_qwen3vl_flare_real.py):
the robot client sends observation payloads (ignored here) and receives
pre-recorded action chunks stepped through the JSON file frame by frame.

Usage:
    python replay_json_client.py \
        --json_path /path/to/training_data.json \
        --port 5678 \
        --stride 1
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import zmq


def load_actions(json_path, stride):
    """Load all action chunks from the JSON file in order."""
    print(f"Loading actions from: {json_path}")

    # The .json is a JSON array written by jsonl_2_json()
    with open(json_path, "r") as f:
        data = json.load(f)

    action_chunks = []
    for sample in data[::stride]:
        action_chunks.append(np.array(sample["action"], dtype=np.float32))

    print(f"  Total samples: {len(data)}, stride={stride} → {len(action_chunks)} steps to replay")
    return action_chunks


def main(args):
    action_chunks = load_actions(args.json_path, args.stride)
    total_steps = len(action_chunks)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"Replay server listening on port {args.port}...")

    step = 0
    while step < total_steps:
        try:
            payload = pickle.loads(socket.recv())
        except Exception:
            payload = {}

        actions = action_chunks[step]   # (action_chunk, action_dim)
        response = {
            "status": "success",
            "actions": actions.tolist(),
            "frame_idx": step,
            "total_frames": total_steps,
        }
        socket.send(pickle.dumps(response))

        step += 1
        if step % 50 == 0 or step == total_steps:
            task = payload.get("task_description", "")[:50] if isinstance(payload, dict) else ""
            print(f"  [{step}/{total_steps}]  task=\"{task}\"")

    print("Replay complete.")

    # Drain one final request so the client gets a clean done signal
    try:
        socket.recv(zmq.NOBLOCK)
        socket.send(pickle.dumps({"status": "done", "actions": []}))
    except zmq.ZMQError:
        pass

    socket.close()
    context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay JSON actions via ZMQ REP server")
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to training .json file")
    parser.add_argument("--port",   type=int, default=5678,
                        help="ZMQ port to bind (default: 5678)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Step through JSON samples by this stride "
                             "(1=every sample, 16=non-overlapping chunks)")
    args = parser.parse_args()

    if not os.path.isfile(args.json_path):
        sys.exit(f"ERROR: JSON file not found: {args.json_path}")

    main(args)
