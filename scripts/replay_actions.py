"""
Replay pre-recorded action sequences from pretrain.hdf5 via ZeroMQ.

Drop-in replacement for the VLA inference server: the robot client sends
observation payloads (ignored) and receives pre-recorded action chunks,
stepping through the HDF5 file frame by frame.

Usage:
    python replay_actions.py \
        --hdf5_path /path/to/pretrain.hdf5 \
        --port 5678 \
        --stride 1
"""

import argparse
import pickle

import h5py
import numpy as np
import zmq


def main(args):
    with h5py.File(args.hdf5_path, "r") as f:
        action_chunks = f["action_chunks"][:]   # (T, chunk, 62)
        states = f["states"][:]                  # (T, 62)
        num_frames = f.attrs["num_frames"]
        fps = f.attrs.get("fps", 30.0)
        language = f.attrs.get("language", "")
        action_dim = f.attrs.get("action_dim", action_chunks.shape[-1])
        chunk_size = f.attrs.get("action_chunk_size", action_chunks.shape[1])

    print(f"Loaded: {args.hdf5_path}")
    print(f"  frames={num_frames}, action_dim={action_dim}, chunk={chunk_size}, fps={fps}")
    print(f"  task: {language}")
    print(f"  stride={args.stride}")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"Replay server listening on port {args.port}...")

    frame_idx = 0
    step_counter = 0

    while frame_idx < num_frames:
        message = socket.recv()
        # Parse client request (same protocol as test_qwen3vl_real.py) but ignore contents
        try:
            _ = pickle.loads(message)
        except Exception:
            pass

        actions = action_chunks[frame_idx]  # (chunk, 62)
        response = {
            "status": "success",
            "actions": list(actions),
            "frame_idx": frame_idx,
            "total_frames": int(num_frames),
        }
        socket.send(pickle.dumps(response))

        step_counter += 1
        frame_idx += args.stride

        if step_counter % 50 == 0 or frame_idx >= num_frames:
            print(f"  [{step_counter}] frame {frame_idx}/{num_frames}")

    print(f"Replay complete. Served {step_counter} steps.")
    # Send done signal on next request
    try:
        socket.recv(zmq.NOBLOCK)
        socket.send(pickle.dumps({"status": "done", "actions": []}))
    except zmq.ZMQError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay HDF5 actions via ZMQ")
    parser.add_argument("--hdf5_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=5678)
    parser.add_argument("--stride", type=int, default=1,
                        help="Step through frames by this stride (1=every frame, "
                             "16=non-overlapping chunks)")
    args = parser.parse_args()
    main(args)


