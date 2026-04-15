"""
Precompute fingertip 2D coordinates for all EgoDex pretraining episodes.

For each episode, reads the raw EgoDex .hdf5 (4D tracking + camera intrinsics)
and saves a lightweight fingertip_coords.npy [T, 10, 2] alongside the episode.

This avoids opening the raw .hdf5 during training — the pretraining dataloader
just loads the precomputed .npy.

Usage:
    python precompute_fingertip_coords.py \
        --data_root /mnt/amlfs-07/shared/datasets/dniu/egodex/cotrain_processed_new \
        --workers 16
"""

import os
import re
import sys
import json
import glob
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
from tqdm import tqdm


_TIP_NAMES = {
    "left":  ["leftThumbTip", "leftIndexFingerTip", "leftMiddleFingerTip",
              "leftRingFingerTip", "leftLittleFingerTip"],
    "right": ["rightThumbTip", "rightIndexFingerTip", "rightMiddleFingerTip",
              "rightRingFingerTip", "rightLittleFingerTip"],
}

_RAW_DATA_ROOTS = [
    "/mnt/amlfs-03/shared/datasets/dniu/egodex/extra",
    "/mnt/amlfs-03/shared/datasets/dniu/egodex/part1",
    "/mnt/amlfs-03/shared/datasets/dniu/egodex/part2",
    "/mnt/amlfs-03/shared/datasets/dniu/egodex/part3",
    "/mnt/amlfs-03/shared/datasets/dniu/egodex/part4",
    "/mnt/amlfs-03/shared/datasets/dniu/egodex/part5",
]


def resolve_raw_h5(episode_dir):
    """Map a cotrain_processed episode dir to its raw EgoDex .hdf5."""
    ep_name = os.path.basename(episode_dir)

    meta_path = os.path.join(episode_dir, "metadata.json")
    meta_traj_id = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta_traj_id = json.load(f).get("trajectory_id")

    task_name = ep_name
    prefix_match = re.match(r'^(extra_|part\d+_|bkl_inlab_\w+_)', task_name)
    if prefix_match:
        task_name = task_name[prefix_match.end():]

    dir_id_match = re.match(r'^(.+?)_(\d+)$', task_name)
    dir_task = dir_id_match.group(1) if dir_id_match else task_name
    dir_id = dir_id_match.group(2) if dir_id_match else None

    candidate_ids = []
    if dir_id is not None:
        candidate_ids.append(dir_id)
    if meta_traj_id is not None:
        candidate_ids.append(str(meta_traj_id))

    for root in _RAW_DATA_ROOTS:
        for tid in candidate_ids:
            candidate = os.path.join(root, dir_task, f"{tid}.hdf5")
            if os.path.isfile(candidate):
                return candidate
    return None


def compute_fingertip_coords(raw_h5_path, num_frames):
    """
    Compute [T, 10, 2] fingertip pixel coordinates from raw EgoDex .hdf5.
    10 = left 5 + right 5 (Thumb, Index, Middle, Ring, Pinky per hand).
    """
    with h5py.File(raw_h5_path, "r") as f:
        K = f["camera/intrinsic"][:].astype(np.float64)
        cam_T = f["transforms/camera"][:].astype(np.float64)
        T_raw = cam_T.shape[0]

        tip_data = {}
        for hand in ["left", "right"]:
            for name in _TIP_NAMES[hand]:
                if name in f["transforms"]:
                    tip_data[name] = f[f"transforms/{name}"][:].astype(np.float64)

    T = min(num_frames, T_raw)
    coords = np.zeros((T, 10, 2), dtype=np.float32)

    ordered_tips = _TIP_NAMES["left"] + _TIP_NAMES["right"]

    for t in range(T):
        cam_inv = np.linalg.inv(cam_T[t])
        for i, name in enumerate(ordered_tips):
            if name not in tip_data:
                continue
            world_pos = tip_data[name][min(t, len(tip_data[name]) - 1), :3, 3]
            cam_pos = (cam_inv @ np.append(world_pos, 1.0))[:3]
            uv_h = K @ cam_pos
            coords[t, i, 0] = uv_h[0] / (uv_h[2] + 1e-8)
            coords[t, i, 1] = uv_h[1] / (uv_h[2] + 1e-8)

    return coords


def process_one_episode(ep_dir):
    """Process a single episode. Returns (success, ep_dir, message)."""
    out_path = os.path.join(ep_dir, "fingertip_coords.npy")
    if os.path.exists(out_path):
        return True, ep_dir, "already exists"

    # Skip non-EgoDex episodes (in-lab data has no raw .hdf5)
    if not os.path.isfile(os.path.join(ep_dir, "ego_view.mp4")):
        return True, ep_dir, "skipped (no ego_view.mp4)"

    raw_h5 = resolve_raw_h5(ep_dir)
    if raw_h5 is None:
        return False, ep_dir, "raw .hdf5 not found"

    try:
        meta_path = os.path.join(ep_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                num_frames = json.load(f).get("frame_count", 0)
        else:
            # Fallback: get frame count from pretrain.hdf5
            pretrain_h5 = os.path.join(ep_dir, "pretrain.hdf5")
            if os.path.exists(pretrain_h5):
                with h5py.File(pretrain_h5, "r") as f:
                    num_frames = f["states"].shape[0]
            else:
                return False, ep_dir, "no metadata or pretrain.hdf5"

        coords = compute_fingertip_coords(raw_h5, num_frames)
        np.save(out_path, coords)
        return True, ep_dir, f"saved ({coords.shape[0]} frames)"
    except Exception as e:
        return False, ep_dir, f"error: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Precompute fingertip 2D coordinates for EgoDex episodes")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root dir containing batch subdirs with episodes")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel workers")
    parser.add_argument("--batches", type=str, default="",
                        help="Comma-separated batch names to process (empty=all)")
    args = parser.parse_args()

    # Discover all episodes
    if args.batches:
        batch_names = [b.strip() for b in args.batches.split(",")]
        episode_dirs = []
        for bn in batch_names:
            batch_dir = os.path.join(args.data_root, bn)
            if os.path.isdir(batch_dir):
                episode_dirs.extend(sorted(glob.glob(os.path.join(batch_dir, "*/"))))
    else:
        episode_dirs = sorted(glob.glob(os.path.join(args.data_root, "*", "*/")))

    # Filter to directories only
    episode_dirs = [d.rstrip("/") for d in episode_dirs if os.path.isdir(d)]
    print(f"Found {len(episode_dirs)} episodes")

    success, fail, skip = 0, 0, 0

    if args.workers <= 1:
        for ep_dir in tqdm(episode_dirs, desc="Processing"):
            ok, _, msg = process_one_episode(ep_dir)
            if ok:
                if "skip" in msg or "already" in msg:
                    skip += 1
                else:
                    success += 1
            else:
                fail += 1
                print(f"  FAIL: {os.path.basename(ep_dir)}: {msg}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one_episode, d): d for d in episode_dirs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                ok, ep_dir, msg = future.result()
                if ok:
                    if "skip" in msg or "already" in msg:
                        skip += 1
                    else:
                        success += 1
                else:
                    fail += 1
                    print(f"  FAIL: {os.path.basename(ep_dir)}: {msg}")

    print(f"\nDone: {success} computed, {skip} skipped, {fail} failed "
          f"(total {len(episode_dirs)} episodes)")


if __name__ == "__main__":
    main()

