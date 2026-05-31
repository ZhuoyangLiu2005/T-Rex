"""Convert egodex/mecka-style pretrain episodes -> a LeRobot v3.0 dataset.

Raw layout (per episode dir):
    <ep>/pretrain.hdf5   states[N,62], action_chunks[N,16,62] (already baked,
                         delta-base), tracking_error[N-1,56] (optional),
                         attrs["language"]
    <ep>/ego_view.mp4    (or *head*.mp4) — the single egocentric view

Pretrain has no tactile and no absolute target on disk; it uses robot state with
tracking-error noise.  We therefore write only the head video, observation.state
and the baked action chunk, and carry tracking_error (if present) into the
q01/q99 sidecar so the loader can do state-noise injection identically.

Episodes are discovered by globbing for `pretrain.hdf5` under one or more roots
(handles the flat merged layout used by the pretrain shell scripts).

Usage:
  python utils/convert_egodex_to_lerobot.py \
      --data_roots /path/egodex_merged \
      --output_root /path/lerobot/egodex \
      --repo_id trex/egodex
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import h5py
import numpy as np

from utils.lerobot_common import (
    KEY_HEAD, KEY_STATE, KEY_ACTION,
    build_trex_features, NormStatsAccumulator,
)


def _find_head_video(ep_dir):
    cand = os.path.join(ep_dir, "ego_view.mp4")
    if os.path.isfile(cand):
        return cand
    matches = glob.glob(os.path.join(ep_dir, "*head*.mp4"))
    return matches[0] if matches else None


def _read_video_rgb(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames if frames else None


def _list_episode_dirs(data_roots):
    eps = []
    for root in data_roots:
        for h5 in sorted(glob.glob(os.path.join(root, "**", "pretrain.hdf5"), recursive=True)):
            eps.append(os.path.dirname(h5))
    return eps


def _load_episode(ep_dir):
    h5_path = os.path.join(ep_dir, "pretrain.hdf5")
    vid = _find_head_video(ep_dir)
    if not os.path.isfile(h5_path) or vid is None:
        return None
    frames = _read_video_rgb(vid)
    if frames is None:
        return None
    with h5py.File(h5_path, "r") as f:
        states = f["states"][:]
        action_chunks = f["action_chunks"][:]
        language = f.attrs.get("language", "")
        track = f["tracking_error"][:] if "tracking_error" in f else None
    return dict(states=states, action_chunks=action_chunks, language=language,
                track=track, frames=frames,
                min_len=min(len(states), len(action_chunks), len(frames)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_roots", nargs="+", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--repo_id", default="trex/egodex")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--default_instruction", default="",
                    help="Fallback task string when an episode has no language attr.")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # lazy

    eps = _list_episode_dirs(args.data_roots)
    print(f"Found {len(eps)} pretrain episodes.")

    probe = None
    for ep in eps:
        probe = _load_episode(ep)
        if probe is not None and probe["min_len"] > 0:
            break
    if probe is None:
        raise RuntimeError("No usable pretrain episode found.")
    head_hw = probe["frames"][0].shape[:2]
    features = build_trex_features(
        head_shape=(3, *head_hw), include_wrist=False,
        include_tactile=False, include_action_abs=False)
    print(f"head shape={head_hw}")

    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, features=features,
        root=args.output_root, robot_type="trex_bimanual", use_videos=True)

    acc = NormStatsAccumulator()
    n_ok = 0
    for ep in eps:
        d = _load_episode(ep)
        if d is None or d["min_len"] <= 0:
            continue
        N = d["min_len"]
        task = d["language"] or args.default_instruction
        for i in range(N):
            ds.add_frame({
                "task": task,
                KEY_HEAD:   d["frames"][i],
                KEY_STATE:  d["states"][i].astype(np.float32),
                KEY_ACTION: d["action_chunks"][i].astype(np.float32),
            })
            acc.add_frame(d["action_chunks"][i], d["states"][i])
        ds.save_episode()
        acc.num_traj += 1
        if d["track"] is not None and len(d["track"]) > 0:
            acc.add_tracking_errors(d["track"])
        n_ok += 1
        print(f"  [{n_ok}] {os.path.basename(ep)} ({N} frames)")

    ds.finalize()
    sidecar = acc.write(args.output_root)
    print(f">>> wrote {args.output_root}\n>>> norm-stats sidecar: {sidecar}")


if __name__ == "__main__":
    main()
