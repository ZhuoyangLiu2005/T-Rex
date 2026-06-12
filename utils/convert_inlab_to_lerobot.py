"""Convert raw in-lab bimanual episodes -> a LeRobot v3.0 dataset.

Raw layout (per the in-lab tactile collection):
    <data_root>/success/episode_*/
        episode_*.h5                 left/right_arm_current_pose [N,4,4],
                                     *_arm_target_pose [N,4,4],
                                     *_hand_joint_positions [N,22],
                                     *_hand_target_joint_positions [N,22],
                                     *_hand_tactile_f6 [N,5,6],
                                     *_hand_tactile_deform [N,5,H,W]
        episode_*_head_left_rgb.mp4  (cropped to CROP_BOX_SLOW)
        episode_*_left_wrist.mp4
        episode_*_right_wrist.mp4

This reproduces the action/tactile semantics of
`gen_json_tac_deltabase_eef_bimanual_parallel.py` exactly (delta-base arm +
absolute target hand, FRAME_STRIDE=1, ACTION_CHUNK=16), but writes a LeRobot
v3.0 dataset instead of a JSON.  Each frame carries the baked `action` chunk
[16,62], `observation.state` [62], `action_abs` [62], `observation.tactile_f6`
[10,6], the three camera videos, and 10 per-finger deform videos.

A `meta/trex_norm_stats.json` sidecar (q01/q99 + tracking_error) is written so
the loader normalizes identically to the JSON pipeline.

Usage:
  python utils/convert_inlab_to_lerobot.py \
      --data_roots /path/task_a /path/task_b \
      --output_root /path/lerobot/place_card \
      --repo_id trex/place_card \
      --instruction "Pick up a stack of playing cards ..."
"""
from __future__ import annotations

import argparse
import os

import cv2
import h5py
import numpy as np

from utils.lerobot_common import (
    ACTION_CHUNK, FRAME_STRIDE, N_FINGERS_PER_HAND,
    KEY_HEAD, KEY_WRIST_R, KEY_WRIST_L, KEY_STATE, KEY_ACTION, KEY_ACTION_ABS,
    KEY_TACF6, DEFORM_KEYS,
    pose_matrix_to_9d, build_action_chunk, build_trex_features, NormStatsAccumulator,
)

DEFAULT_INSTRUCTION = (
    "Pick up a stack of playing cards with your right hand, then transfer it to "
    "your left; hold the stack aloft with your left hand, use your right thumb to "
    "slide out the top card, grasp it, and place it into the card holder.")
DEFAULT_CROP_BOX = (0, 300, 140, 540)   # (y_min, y_max, x_min, x_max)


def _read_video_rgb(path, crop_box=None):
    """Decode an mp4 into a list of RGB uint8 frames (cropped if requested)."""
    if not os.path.exists(path):
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if crop_box is not None:
            y0, y1, x0, x1 = crop_box
            frame = frame[y0:y1, x0:x1]
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))   # BGR -> RGB
    cap.release()
    return frames if frames else None


def _gray_to_3ch(img):
    """[H,W] (any dtype) -> [H,W,3] uint8 (replicated)."""
    a = np.asarray(img)
    if a.ndim == 3:
        a = a[..., 0]
    a = np.clip(a, 0, 255).astype(np.uint8)
    return np.repeat(a[:, :, None], 3, axis=2)


def _list_episodes(data_roots):
    entries = []
    for root in data_roots:
        prefix = os.path.basename(root.rstrip("/"))
        succ = os.path.join(root, "success")
        if not os.path.isdir(succ):
            print(f"  warn: {succ} missing, skip")
            continue
        for name in sorted(os.listdir(succ)):
            d = os.path.join(succ, name)
            if os.path.isdir(d) and name.startswith("episode_"):
                entries.append((prefix, d))
    return entries


def _load_episode_arrays(ep_dir, crop_box):
    """Return per-frame arrays + decoded frames, or None on any failure."""
    name = os.path.basename(ep_dir)
    h5_path = os.path.join(ep_dir, f"{name}.h5")
    if not os.path.exists(h5_path):
        return None
    head = _read_video_rgb(os.path.join(ep_dir, f"{name}_head_left_rgb.mp4"), crop_box)
    wl   = _read_video_rgb(os.path.join(ep_dir, f"{name}_left_wrist.mp4"))
    wr   = _read_video_rgb(os.path.join(ep_dir, f"{name}_right_wrist.mp4"))
    if head is None or wl is None or wr is None:
        return None

    with h5py.File(h5_path, "r") as f:
        s_l_pose = f["left_arm_current_pose"][:]
        s_r_pose = f["right_arm_current_pose"][:]
        a_l_pose = f["left_arm_target_pose"][:]
        a_r_pose = f["right_arm_target_pose"][:]
        s_l_hnd = f["left_hand_joint_positions"][:]
        s_r_hnd = f["right_hand_joint_positions"][:]
        a_l_hnd = f["left_hand_target_joint_positions"][:]
        a_r_hnd = f["right_hand_target_joint_positions"][:]
        t_l_f6 = f["left_hand_tactile_f6"][:]      # [N,5,6]
        t_r_f6 = f["right_hand_tactile_f6"][:]
        t_l_def = f["left_hand_tactile_deform"][:]  # [N,5,H,W]
        t_r_def = f["right_hand_tactile_deform"][:]

    s_l_9d = pose_matrix_to_9d(s_l_pose)
    s_r_9d = pose_matrix_to_9d(s_r_pose)
    a_l_9d = pose_matrix_to_9d(a_l_pose)
    a_r_9d = pose_matrix_to_9d(a_r_pose)
    states = np.concatenate([s_l_9d, s_l_hnd, s_r_9d, s_r_hnd], axis=1)          # [N,62]
    abs_targets = np.concatenate([a_l_9d, a_l_hnd, a_r_9d, a_r_hnd], axis=1)     # [N,62]

    min_len = min(len(states), len(head), len(wl), len(wr))
    return dict(
        min_len=min_len, head=head, wl=wl, wr=wr,
        s_l_pose=s_l_pose, s_r_pose=s_r_pose, a_l_pose=a_l_pose, a_r_pose=a_r_pose,
        a_l_hnd=a_l_hnd, a_r_hnd=a_r_hnd,
        states=states, abs_targets=abs_targets,
        t_l_f6=t_l_f6, t_r_f6=t_r_f6, t_l_def=t_l_def, t_r_def=t_r_def,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_roots", nargs="+", required=True,
                    help="Raw roots; each must contain success/episode_*/.")
    ap.add_argument("--output_root", required=True, help="LeRobot dataset dir to create.")
    ap.add_argument("--repo_id", default="trex/inlab", help="LeRobot repo_id (offline; any string).")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crop_box", type=int, nargs=4, default=list(DEFAULT_CROP_BOX),
                    metavar=("Y0", "Y1", "X0", "X1"))
    ap.add_argument("--num_trajectories", type=int, default=0, help="0 = all; else few-shot subset.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # lazy

    crop_box = tuple(args.crop_box)
    entries = _list_episodes(args.data_roots)
    if args.num_trajectories and 0 < args.num_trajectories < len(entries):
        import random
        rng = random.Random(args.seed)
        entries = sorted(rng.sample(entries, args.num_trajectories))
        print(f"Few-shot: {len(entries)} episodes (seed={args.seed})")
    print(f"Found {len(entries)} episodes across {len(args.data_roots)} root(s).")

    # Probe the first usable episode for camera/deform resolutions.
    probe = None
    for _, ep in entries:
        probe = _load_episode_arrays(ep, crop_box)
        if probe is not None and probe["min_len"] > 0:
            break
    if probe is None:
        raise RuntimeError("No usable episode found to probe shapes.")
    head_hw = probe["head"][0].shape[:2]
    wrist_hw = probe["wl"][0].shape[:2]
    def_hw = probe["t_l_def"][0, 0].shape[:2]
    features = build_trex_features(
        head_shape=(3, *head_hw), include_wrist=True, wrist_shape=(3, *wrist_hw),
        include_tactile=True, deform_shape=(3, *def_hw))
    print(f"shapes: head={head_hw} wrist={wrist_hw} deform={def_hw}")

    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, features=features,
        root=args.output_root, robot_type="trex_bimanual", use_videos=True)

    acc = NormStatsAccumulator()
    n_ok = 0
    for prefix, ep in entries:
        d = _load_episode_arrays(ep, crop_box)
        if d is None or d["min_len"] <= 0:
            print(f"  skip {prefix}/{os.path.basename(ep)}")
            continue
        N = d["min_len"]
        for i in range(N):
            chunk = build_action_chunk(
                d["s_l_pose"], d["a_l_pose"], d["a_l_hnd"],
                d["s_r_pose"], d["a_r_pose"], d["a_r_hnd"], i, N)
            tgt_next = min(i + FRAME_STRIDE - 1, N - 1)
            tac_f6 = np.concatenate([d["t_l_f6"][i], d["t_r_f6"][i]], axis=0).astype(np.float32)  # [10,6]

            frame = {
                "task": args.instruction,
                KEY_HEAD:       d["head"][i],
                KEY_WRIST_R:    d["wr"][i],
                KEY_WRIST_L:    d["wl"][i],
                KEY_STATE:      d["states"][i].astype(np.float32),
                KEY_ACTION:     chunk,
                KEY_ACTION_ABS: d["abs_targets"][tgt_next].astype(np.float32),
                KEY_TACF6:      tac_f6,
            }
            for fk in range(N_FINGERS_PER_HAND):
                frame[DEFORM_KEYS[fk]] = _gray_to_3ch(d["t_l_def"][i, fk])
                frame[DEFORM_KEYS[N_FINGERS_PER_HAND + fk]] = _gray_to_3ch(d["t_r_def"][i, fk])
            ds.add_frame(frame)
            acc.add_frame(chunk, d["states"][i], tac_f6)

        ds.save_episode()
        acc.add_episode_tracking(d["states"][:N], d["abs_targets"][:N])
        n_ok += 1
        print(f"  [{n_ok}] wrote {prefix}/{os.path.basename(ep)} ({N} frames)")

    ds.finalize()
    sidecar = acc.write(args.output_root)
    print(f">>> LeRobot dataset written to {args.output_root}")
    print(f">>> norm-stats sidecar: {sidecar}")


if __name__ == "__main__":
    main()
