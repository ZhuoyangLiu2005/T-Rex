"""
Optical-Flow Tactile Proxy — Proof of Concept
==============================================
Extracts surrogate tactile signals from ego-centric hand videos using RAFT
optical flow, guided by (mock) forward-kinematics fingertip projections.

Two proxy metrics per fingertip per frame:
  1. Transient proxy  (acceleration): temporal derivative of local flow magnitude
     a_t = mean(|Flow_t - Flow_{t-1}|)  within a 16x16 patch
  2. Steady-state proxy (flow sync):   difference between fingertip-patch mean
     flow and surrounding donut region. ~0 means finger locked to object.

Usage:
    python extract_flow_tactile_proxy.py \
        --episode_dir /path/to/episode \
        --output_dir  ./flow_tactile_vis \
        --max_frames 120 \
        --device cuda:0

Requirements: torchvision (RAFT), opencv-python, matplotlib, h5py
"""

import os
import argparse

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


FINGER_NAMES_PER_HAND = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
# 10 fingers: Left 5 + Right 5
FINGER_NAMES = [f"L_{n}" for n in FINGER_NAMES_PER_HAND] + \
               [f"R_{n}" for n in FINGER_NAMES_PER_HAND]
FINGER_COLORS_LEFT = [
    (200, 80, 80),     # L Thumb  — dark red
    (80, 160, 80),     # L Index  — dark green
    (80, 80, 200),     # L Middle — dark blue
    (200, 160, 0),     # L Ring   — dark yellow
    (160, 80, 200),    # L Pinky  — dark purple
]
FINGER_COLORS_RIGHT = [
    (255, 50, 50),     # R Thumb  — bright red
    (50, 255, 50),     # R Index  — bright green
    (50, 100, 255),    # R Middle — bright blue
    (255, 255, 0),     # R Ring   — bright yellow
    (255, 50, 255),    # R Pinky  — bright magenta
]
FINGER_COLORS = FINGER_COLORS_LEFT + FINGER_COLORS_RIGHT


_TIP_NAMES = {
    "left":  ["leftThumbTip", "leftIndexFingerTip", "leftMiddleFingerTip",
              "leftRingFingerTip", "leftLittleFingerTip"],
    "right": ["rightThumbTip", "rightIndexFingerTip", "rightMiddleFingerTip",
              "rightRingFingerTip", "rightLittleFingerTip"],
}


class EgoDexFingertipProjector:
    """
    Projects fingertips to 2D using the raw EgoDex 4D-tracking HDF5 file.

    The raw file contains:
      camera/intrinsic  : (3, 3)     pinhole camera intrinsic matrix
      transforms/camera : (T, 4, 4)  camera pose in world frame per frame
      transforms/<tip>  : (T, 4, 4)  per-joint world-frame 4x4 transforms
    """

    def __init__(self, raw_h5_path, hands=("right",)):
        self.hands = hands
        with h5py.File(raw_h5_path, "r") as f:
            self.K = f["camera/intrinsic"][:].astype(np.float64)
            self.cam_T = f["transforms/camera"][:].astype(np.float64)
            self.tip_data = {}
            for hand in hands:
                for name in _TIP_NAMES[hand]:
                    self.tip_data[name] = f[f"transforms/{name}"][:].astype(np.float64)
        self.n_raw_frames = self.cam_T.shape[0]

    def project(self, frame_idx, hand, img_h=None, img_w=None):
        """Return (5, 2) pixel coords for 5 fingertips of the given hand."""
        t = min(frame_idx, self.n_raw_frames - 1)
        cam_inv = np.linalg.inv(self.cam_T[t])
        coords = []
        for name in _TIP_NAMES[hand]:
            world_pos = self.tip_data[name][t, :3, 3]
            cam_pos = (cam_inv @ np.append(world_pos, 1.0))[:3]
            uv_h = self.K @ cam_pos
            u = uv_h[0] / (uv_h[2] + 1e-8)
            v = uv_h[1] / (uv_h[2] + 1e-8)
            if img_w is not None:
                u = np.clip(u, 0, img_w - 1)
            if img_h is not None:
                v = np.clip(v, 0, img_h - 1)
            coords.append([u, v])
        return np.array(coords, dtype=np.float32)


def resolve_raw_h5_path(episode_dir, raw_data_roots=None):
    """
    Map a cotrain_processed_new episode dir to its raw EgoDex .hdf5.

    Episode dir name: extra_{task_name}_{trajectory_id}
    Raw file:         {raw_root}/{task_name}/{trajectory_id}.hdf5
    """
    import json as _json

    import re as _re

    ep_name = os.path.basename(episode_dir)
    meta_path = os.path.join(episode_dir, "metadata.json")
    meta_traj_id = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta_traj_id = _json.load(f).get("trajectory_id")

    # Strip batch prefix: extra_, part1_, part2_, ..., bkl_inlab_*
    task_name = ep_name
    prefix_match = _re.match(r'^(extra_|part\d+_|bkl_inlab_\w+_)', task_name)
    if prefix_match:
        task_name = task_name[prefix_match.end():]

    # Extract trailing numeric ID from dir name: e.g. "assemble_foo_3181" → ("assemble_foo", "3181")
    dir_id_match = _re.match(r'^(.+?)_(\d+)$', task_name)
    dir_task = dir_id_match.group(1) if dir_id_match else task_name
    dir_id = dir_id_match.group(2) if dir_id_match else None

    # Candidate IDs to try: dir name ID, metadata ID
    candidate_ids = []
    if dir_id is not None:
        candidate_ids.append(dir_id)
    if meta_traj_id is not None:
        candidate_ids.append(str(meta_traj_id))

    if raw_data_roots is None:
        raw_data_roots = [
            "/mnt/amlfs-03/shared/datasets/dniu/egodex/extra",
            "/mnt/amlfs-03/shared/datasets/dniu/egodex/part1",
            "/mnt/amlfs-03/shared/datasets/dniu/egodex/part2",
            "/mnt/amlfs-03/shared/datasets/dniu/egodex/part3",
            "/mnt/amlfs-03/shared/datasets/dniu/egodex/part4",
            "/mnt/amlfs-03/shared/datasets/dniu/egodex/part5",
        ]
    for root in raw_data_roots:
        for tid in candidate_ids:
            candidate = os.path.join(root, dir_task, f"{tid}.hdf5")
            if os.path.isfile(candidate):
                return candidate
    return None


# ─────────────────────────────────────────────────────────────────────────────
# RAFT optical flow
# ─────────────────────────────────────────────────────────────────────────────

def load_raft(device):
    """Load RAFT-Large with default weights."""
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_large(weights=weights).to(device).eval()
    return model, transforms


@torch.no_grad()
def compute_flow(model, transforms, frame1, frame2, device):
    """
    Compute optical flow from frame1 → frame2 using RAFT.

    Parameters
    ----------
    frame1, frame2 : np.ndarray [H, W, 3] uint8 RGB

    Returns
    -------
    flow : np.ndarray [H, W, 2] float32 — (dx, dy) per pixel
    """
    h, w = frame1.shape[:2]
    # RAFT needs dimensions divisible by 8
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8

    t1 = torch.from_numpy(frame1).permute(2, 0, 1).unsqueeze(0).float().to(device)
    t2 = torch.from_numpy(frame2).permute(2, 0, 1).unsqueeze(0).float().to(device)

    t1, t2 = transforms(t1, t2)

    if pad_h > 0 or pad_w > 0:
        t1 = F.pad(t1, (0, pad_w, 0, pad_h), mode="replicate")
        t2 = F.pad(t2, (0, pad_w, 0, pad_h), mode="replicate")

    flow_list = model(t1, t2)
    flow = flow_list[-1][0]  # last iteration, remove batch dim → [2, H', W']

    flow = flow[:, :h, :w]  # remove padding
    return flow.cpu().permute(1, 2, 0).numpy()  # [H, W, 2]


# ─────────────────────────────────────────────────────────────────────────────
# Patch extraction & proxy metrics
# ─────────────────────────────────────────────────────────────────────────────

def extract_patch_flow(flow, cx, cy, inner_size=16, outer_size=32):
    """
    Extract flow statistics from an inner patch and surrounding donut.

    Returns
    -------
    inner_mean_flow : np.ndarray [2]  — mean (dx, dy) in inner patch
    inner_mag       : float           — mean flow magnitude in inner patch
    donut_mean_flow : np.ndarray [2]  — mean (dx, dy) in outer donut
    """
    h, w = flow.shape[:2]
    half_in = inner_size // 2
    half_out = outer_size // 2

    # Inner patch
    y1i = max(0, int(cy) - half_in)
    y2i = min(h, int(cy) + half_in)
    x1i = max(0, int(cx) - half_in)
    x2i = min(w, int(cx) + half_in)
    inner = flow[y1i:y2i, x1i:x2i]

    if inner.size == 0:
        return np.zeros(2), 0.0, np.zeros(2)

    inner_mean = inner.mean(axis=(0, 1))
    inner_mag = np.linalg.norm(inner, axis=-1).mean()

    # Outer patch
    y1o = max(0, int(cy) - half_out)
    y2o = min(h, int(cy) + half_out)
    x1o = max(0, int(cx) - half_out)
    x2o = min(w, int(cx) + half_out)
    outer = flow[y1o:y2o, x1o:x2o]

    # Donut = outer minus inner
    mask = np.ones((y2o - y1o, x2o - x1o), dtype=bool)
    # inner relative coords
    iy1, iy2 = y1i - y1o, y2i - y1o
    ix1, ix2 = x1i - x1o, x2i - x1o
    if iy1 >= 0 and iy2 <= mask.shape[0] and ix1 >= 0 and ix2 <= mask.shape[1]:
        mask[iy1:iy2, ix1:ix2] = False

    donut = outer[mask]
    donut_mean = donut.mean(axis=0) if donut.size > 0 else np.zeros(2)

    return inner_mean, inner_mag, donut_mean


def compute_proxy_metrics(flows, fingertip_coords_per_frame):
    """
    Compute transient and steady-state proxy metrics.

    Parameters
    ----------
    flows : list of np.ndarray [H, W, 2], length T-1 (flow between consecutive frames)
    fingertip_coords_per_frame : list of np.ndarray [5, 2], length T

    Returns
    -------
    transient : np.ndarray [T-1, 5] — acceleration proxy per fingertip
    steady    : np.ndarray [T-1, 5] — flow synchronization proxy per fingertip
    flow_mags : np.ndarray [T-1, 5] — raw flow magnitude per fingertip
    """
    T_minus_1 = len(flows)
    n_fingers = fingertip_coords_per_frame[0].shape[0]

    flow_mags = np.zeros((T_minus_1, n_fingers))
    inner_means = np.zeros((T_minus_1, n_fingers, 2))
    steady = np.zeros((T_minus_1, n_fingers))

    for t in range(T_minus_1):
        # Use fingertip coords at frame t+1 (the target frame of flow t→t+1)
        coords = fingertip_coords_per_frame[t + 1]
        for f in range(n_fingers):
            cx, cy = coords[f]
            inner_mean, inner_mag, donut_mean = extract_patch_flow(flows[t], cx, cy)
            flow_mags[t, f] = inner_mag
            inner_means[t, f] = inner_mean
            # Steady-state: how synchronized is the fingertip flow with background
            steady[t, f] = np.linalg.norm(inner_mean - donut_mean)

    # Transient: temporal derivative of flow magnitude (acceleration)
    transient = np.zeros((T_minus_1, n_fingers))
    for t in range(1, T_minus_1):
        transient[t] = np.abs(flow_mags[t] - flow_mags[t - 1])

    return transient, steady, flow_mags


# ─────────────────────────────────────────────────────────────────────────────
# Flow visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def flow_to_color(flow, max_flow=None, percentile=98):
    """
    Convert optical flow to HSV color visualization.
    Uses per-frame percentile normalization for better contrast.
    """
    h, w = flow.shape[:2]
    mag = np.linalg.norm(flow, axis=-1)
    ang = np.arctan2(flow[:, :, 1], flow[:, :, 0])

    if max_flow is None:
        max_flow = max(np.percentile(mag, percentile), 1e-5)

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.clip(mag / max_flow * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def draw_fingertips_on_image(img, coords, colors=None, inner_size=16,
                             flow=None, arrow_scale=3.0):
    """
    Draw fingertip locations, patches, and optionally flow arrows on an image.
    When flow is provided, draws an arrow at each fingertip showing flow direction/magnitude.
    """
    vis = img.copy()
    half = inner_size // 2
    if colors is None:
        colors = FINGER_COLORS
    for idx, (cx, cy) in enumerate(coords):
        color = colors[idx % len(colors)]
        pt = (int(cx), int(cy))
        # Patch rectangle
        tl = (int(cx) - half, int(cy) - half)
        br = (int(cx) + half, int(cy) + half)
        cv2.rectangle(vis, tl, br, color, 2)
        # Center dot
        cv2.circle(vis, pt, 4, color, -1)
        cv2.circle(vis, pt, 5, (255, 255, 255), 1)
        # Flow arrow
        if flow is not None:
            iy, ix = int(np.clip(cy, 0, flow.shape[0] - 1)), int(np.clip(cx, 0, flow.shape[1] - 1))
            dx, dy = flow[iy, ix]
            end_pt = (int(cx + dx * arrow_scale), int(cy + dy * arrow_scale))
            cv2.arrowedLine(vis, pt, end_pt, (255, 255, 255), 2, tipLength=0.3)
            cv2.arrowedLine(vis, pt, end_pt, color, 1, tipLength=0.3)
    return vis


def load_video_frames(video_path, max_frames=None, stride=1):
    """Load frames from an mp4 video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def _find_video(ep_dir):
    """Auto-detect video file: EgoDex (ego_view.mp4) or in-lab (episode_*_head_*.mp4)."""
    import glob as _glob
    for pattern in ["ego_view.mp4", "episode_*_head_left_rgb.mp4", "episode_*_head*.mp4"]:
        matches = _glob.glob(os.path.join(ep_dir, pattern))
        if matches:
            return matches[0]
    return None


def _find_inlab_h5(ep_dir):
    """Find in-lab episode .h5 with tactile/joint data."""
    import glob as _glob
    for pattern in ["episode_*.h5"]:
        matches = _glob.glob(os.path.join(ep_dir, pattern))
        # Skip pretrain.hdf5 etc
        matches = [m for m in matches if m.endswith(".h5")]
        if matches:
            return matches[0]
    return None


def _detect_data_source(ep_dir):
    """
    Detect whether this is EgoDex or in-lab data.
    Returns "egodex" or "inlab".
    """
    if os.path.isfile(os.path.join(ep_dir, "ego_view.mp4")):
        return "egodex"
    if _find_inlab_h5(ep_dir) is not None:
        return "inlab"
    return "unknown"


def _load_gt_tactile(h5_path, hand="left"):
    """Load GT tactile data from in-lab .h5 if available."""
    with h5py.File(h5_path, "r") as f:
        key_f6 = f"{hand}_hand_tactile_f6"
        key_deform = f"{hand}_hand_tactile_deform"
        if key_f6 not in f:
            return None
        f6 = f[key_f6][:]             # (T, 5, 6)
        deform = f[key_deform][:] if key_deform in f else None
    force_mag = np.linalg.norm(f6[:, :, :3], axis=-1)
    baseline = force_mag[:10].mean(axis=0)
    delta_force = force_mag - baseline[None, :]
    deform_diff = np.zeros((f6.shape[0], 5))
    if deform is not None:
        deform_diff[1:] = np.abs(
            deform[1:].astype(float) - deform[:-1].astype(float)
        ).mean(axis=(2, 3))
    gt_contact = (np.abs(delta_force) > 2) | (deform_diff > 2)
    return {
        "force_mag": force_mag,
        "delta_force": delta_force,
        "deform_diff": deform_diff,
        "gt_contact": gt_contact,
    }


def _grid_fingertip_coords(flows, img_h, img_w, grid_r=4, grid_c=4):
    """
    For in-lab data without per-finger projection, use a grid of image regions.
    Find the most active region (highest mean flow) and sample 5 points within it.
    Returns list of (5, 2) arrays — 5 pseudo-fingertip positions.
    """
    T_minus_1 = len(flows)
    cell_h, cell_w = img_h // grid_r, img_w // grid_c

    # Find the most active region across all frames
    region_flow = np.zeros((grid_r, grid_c))
    for flow in flows:
        mag = np.linalg.norm(flow, axis=-1)
        for r in range(grid_r):
            for c in range(grid_c):
                region_flow[r, c] += mag[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w].mean()
    region_flow /= T_minus_1
    hand_r, hand_c = np.unravel_index(region_flow.argmax(), region_flow.shape)
    print(f"  Most active region: row={hand_r}, col={hand_c} "
          f"(y=[{hand_r*cell_h}:{(hand_r+1)*cell_h}], x=[{hand_c*cell_w}:{(hand_c+1)*cell_w}])")

    # Place 5 sample points in a cross pattern within that region
    cy_center = hand_r * cell_h + cell_h // 2
    cx_center = hand_c * cell_w + cell_w // 2
    offsets = [
        (0, 0),                                 # center (Thumb)
        (-cell_h // 4, -cell_w // 4),           # upper-left (Index)
        (-cell_h // 4, +cell_w // 4),           # upper-right (Middle)
        (+cell_h // 4, -cell_w // 4),           # lower-left (Ring)
        (+cell_h // 4, +cell_w // 4),           # lower-right (Pinky)
    ]
    base_coords = np.array(
        [[cx_center + dx, cy_center + dy] for dy, dx in offsets], dtype=np.float32)

    # Return the same coords for all frames (static grid)
    return [base_coords.copy() for _ in range(T_minus_1 + 1)]


def run_pipeline(args):
    ep_dir = args.episode_dir
    video_path = _find_video(ep_dir)

    if video_path is None or not os.path.isfile(video_path):
        raise FileNotFoundError(f"No video found in {ep_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Step 1: Load video frames ───────────────────────────────────────
    print(f"Loading video: {video_path}")
    frames = load_video_frames(video_path, max_frames=args.max_frames,
                               stride=args.frame_stride)
    T = len(frames)
    img_h, img_w = frames[0].shape[:2]
    print(f"Loaded {T} frames ({img_h}x{img_w}), stride={args.frame_stride}")

    # ── Step 2: RAFT optical flow ───────────────────────────────────────
    print("Loading RAFT model...")
    raft_model, raft_transforms = load_raft(device)

    print("Computing optical flow...")
    flows = []
    for t in tqdm(range(T - 1), desc="RAFT flow"):
        flow = compute_flow(raft_model, raft_transforms, frames[t], frames[t + 1], device)
        flows.append(flow)

    # Free GPU memory
    del raft_model
    torch.cuda.empty_cache()

    # ── Step 3: Fingertip coordinates ─────────────────────────────────
    data_source = _detect_data_source(ep_dir)
    gt_tactile = None
    inlab_h5_path = None

    if data_source == "egodex":
        raw_h5 = args.raw_h5_path
        if not raw_h5:
            raw_h5 = resolve_raw_h5_path(ep_dir)
        if raw_h5 is None or not os.path.isfile(raw_h5):
            raise FileNotFoundError(
                f"Cannot find raw EgoDex HDF5 for episode {ep_dir}.\n"
                f"Provide --raw_h5_path or check raw data roots.")

        print(f"[EgoDex] Loading 4D tracking from: {raw_h5}")
        if args.hand == "both":
            track_hands = ("left", "right")
        else:
            track_hands = (args.hand,)
        projector = EgoDexFingertipProjector(raw_h5, hands=track_hands)

        fingertip_coords = []
        for t in range(T):
            real_frame_idx = t * args.frame_stride
            parts = []
            for hand in track_hands:
                parts.append(projector.project(real_frame_idx, hand, img_h, img_w))
            fingertip_coords.append(np.concatenate(parts, axis=0))

    else:
        # In-lab data: use grid-based region detection
        inlab_h5_path = _find_inlab_h5(ep_dir)
        print(f"[In-lab] Using grid-based flow region detection")
        if inlab_h5_path:
            print(f"  Found h5: {os.path.basename(inlab_h5_path)}")
        # For in-lab, we track a single hand's region (5 sample points)
        track_hands = (args.hand,) if args.hand != "both" else ("left",)
        fingertip_coords = _grid_fingertip_coords(flows, img_h, img_w)

        # Load GT tactile for comparison
        if inlab_h5_path:
            for hand in ["left", "right"]:
                gt = _load_gt_tactile(inlab_h5_path, hand)
                if gt is not None and gt["gt_contact"].any():
                    gt_tactile = gt
                    gt_tactile["hand"] = hand
                    print(f"  Loaded GT tactile for {hand} hand")
                    break

    # ── Step 4: Compute proxy metrics ───────────────────────────────────
    print("Computing tactile proxy metrics...")
    transient, steady, flow_mags = compute_proxy_metrics(flows, fingertip_coords)

    # Save raw metrics
    np.savez(os.path.join(args.output_dir, "tactile_proxy_metrics.npz"),
             transient=transient, steady=steady, flow_mags=flow_mags,
             fingertip_coords=np.array(fingertip_coords))
    print(f"Saved metrics to {args.output_dir}/tactile_proxy_metrics.npz")

    # ── Step 5: Visualization ───────────────────────────────────────────
    print("Generating visualizations...")

    # --- 5a: Summary figure (time series) ---
    n_fingers = 5 * len(track_hands)
    if args.hand == "both":
        finger_names = FINGER_NAMES
        finger_colors = FINGER_COLORS
    elif args.hand == "left":
        finger_names = [f"L_{n}" for n in FINGER_NAMES_PER_HAND]
        finger_colors = FINGER_COLORS_LEFT
    else:
        finger_names = [f"R_{n}" for n in FINGER_NAMES_PER_HAND]
        finger_colors = FINGER_COLORS_RIGHT
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)

    # Flow magnitude
    ax0 = fig.add_subplot(gs[0])
    time_axis = np.arange(T - 1)
    for f in range(n_fingers):
        ls = "--" if f < 5 else "-"  # dashed=left, solid=right
        ax0.plot(time_axis, flow_mags[:, f], label=finger_names[f],
                 color=np.array(finger_colors[f]) / 255, alpha=0.8, linestyle=ls)
    ax0.set_ylabel("Flow Magnitude (px)")
    ax0.set_title("Fingertip Optical Flow Magnitude (dashed=Left, solid=Right)")
    ax0.legend(fontsize=7, ncol=5, loc="upper right")
    ax0.grid(True, linestyle=":", alpha=0.5)

    # Transient proxy (acceleration)
    ax1 = fig.add_subplot(gs[1])
    for f in range(n_fingers):
        ls = "--" if f < 5 else "-"
        ax1.plot(time_axis, transient[:, f], label=finger_names[f],
                 color=np.array(finger_colors[f]) / 255, alpha=0.8, linestyle=ls)
    ax1.set_ylabel("Transient Proxy\n(|dFlow/dt|)")
    ax1.set_title("Transient Contact Proxy (Acceleration)")
    ax1.legend(fontsize=7, ncol=5, loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Steady-state proxy (flow sync)
    ax2 = fig.add_subplot(gs[2])
    for f in range(n_fingers):
        ls = "--" if f < 5 else "-"
        ax2.plot(time_axis, steady[:, f], label=finger_names[f],
                 color=np.array(finger_colors[f]) / 255, alpha=0.8, linestyle=ls)
    ax2.set_ylabel("Steady-State Proxy\n(Patch-Donut diff)")
    ax2.set_xlabel("Frame")
    ax2.set_title("Steady-State Grasping Proxy (Flow Synchronization)")
    ax2.legend(fontsize=7, ncol=5, loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.savefig(os.path.join(args.output_dir, "tactile_proxy_timeseries.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: tactile_proxy_timeseries.png")

    # --- 5b: Sampled frame panels ---
    n_panels = min(args.num_panel_frames, T - 1)
    panel_indices = np.linspace(0, T - 2, n_panels, dtype=int)

    fig, axes = plt.subplots(3, n_panels, figsize=(4 * n_panels, 10))
    if n_panels == 1:
        axes = axes[:, np.newaxis]

    for col, t in enumerate(panel_indices):
        # Row 0: RGB + fingertips + flow arrows
        vis_rgb = draw_fingertips_on_image(frames[t + 1], fingertip_coords[t + 1],
                                           finger_colors, flow=flows[t])
        axes[0, col].imshow(vis_rgb)
        axes[0, col].set_title(f"Frame {t+1}", fontsize=9)
        axes[0, col].axis("off")

        # Row 1: Flow color (per-frame normalization) + fingertip patches + arrows
        flow_vis = flow_to_color(flows[t])  # per-frame percentile normalization
        flow_vis = draw_fingertips_on_image(flow_vis, fingertip_coords[t + 1],
                                            finger_colors, flow=flows[t])
        axes[1, col].imshow(flow_vis)
        axes[1, col].set_title(f"Flow {t}→{t+1}", fontsize=9)
        axes[1, col].axis("off")

        # Row 2: Per-finger metrics at this frame
        bar_x = np.arange(n_fingers)
        bar_width = 0.35
        axes[2, col].bar(bar_x - bar_width / 2, transient[t],
                         bar_width, label="Transient", color="coral", alpha=0.8)
        axes[2, col].bar(bar_x + bar_width / 2, steady[t],
                         bar_width, label="Sync", color="steelblue", alpha=0.8)
        axes[2, col].set_xticks(bar_x)
        axes[2, col].set_xticklabels([n.replace("_", "\n") for n in finger_names],
                                      fontsize=5, rotation=45, ha="right")
        if col == 0:
            axes[2, col].legend(fontsize=7)
        axes[2, col].set_title(f"Metrics @ {t+1}", fontsize=9)

    plt.suptitle("Optical Flow Tactile Proxy — Sampled Frames", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "tactile_proxy_panels.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: tactile_proxy_panels.png")

    # --- 5c: Output video (optional) ---
    if args.save_video:
        out_video = os.path.join(args.output_dir, "tactile_proxy_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_h = img_h * 2  # stack RGB and flow vertically
        writer = cv2.VideoWriter(out_video, fourcc, 15, (img_w, out_h))

        for t in tqdm(range(T - 1), desc="Writing video"):
            top = draw_fingertips_on_image(frames[t + 1], fingertip_coords[t + 1],
                                           finger_colors, flow=flows[t])
            flow_vis = flow_to_color(flows[t])  # per-frame normalization
            flow_vis = draw_fingertips_on_image(flow_vis, fingertip_coords[t + 1],
                                                finger_colors, flow=flows[t])
            stacked = np.vstack([top, flow_vis])
            writer.write(cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))

        writer.release()
        print(f"  Saved: {out_video}")

    # ── Step 6: Contact detection & contact video ────────────────────────
    print("Detecting contact events...")

    # Temporally consistent tactile detection:
    #   1. Smooth raw signals with EMA to remove single-frame noise
    #   2. Hysteresis thresholding: high threshold to enter, low to exit
    #   3. Morphological cleanup: remove short events, fill short gaps

    ema_alpha = 0.3  # smoothing factor (lower = smoother)
    min_event_frames = 3   # discard tactile events shorter than this
    min_gap_frames = 5     # fill free gaps shorter than this

    state = np.zeros((T - 1, n_fingers), dtype=int)
    contact_events = {}

    for f in range(n_fingers):
        trans_f = transient[:, f].copy()
        steady_f = steady[:, f].copy()
        flow_f = flow_mags[:, f].copy()

        # --- Step 1: EMA smoothing ---
        for t_i in range(1, len(trans_f)):
            trans_f[t_i] = ema_alpha * trans_f[t_i] + (1 - ema_alpha) * trans_f[t_i - 1]
            steady_f[t_i] = ema_alpha * steady_f[t_i] + (1 - ema_alpha) * steady_f[t_i - 1]
            flow_f[t_i] = ema_alpha * flow_f[t_i] + (1 - ema_alpha) * flow_f[t_i - 1]

        # --- Step 2: Adaptive hysteresis thresholds ---
        med_t = np.median(trans_f)
        mad_t = np.median(np.abs(trans_f - med_t))
        thresh_t_high = med_t + 2.0 * max(mad_t, 1.0)   # enter tactile
        thresh_t_low = med_t + 1.0 * max(mad_t, 1.0)    # exit tactile

        med_s = np.median(steady_f)
        mad_s = np.median(np.abs(steady_f - med_s))
        thresh_s = med_s + 1.5 * max(mad_s, 1.0)

        # Low-flow = finger is stationary (likely in sustained contact)
        # Use 25th percentile as the "stationary" threshold — if flow is
        # in the bottom quarter, the finger isn't moving freely
        flow_p25 = np.percentile(flow_f, 25)
        flow_p50 = np.percentile(flow_f, 50)

        # Hysteresis state machine with 3 detection criteria:
        #   1. Transient spike → contact onset
        #   2. Low steady-state diff → finger locked to object
        #   3. Low flow persistence → finger stationary (sustained grasp)
        in_tactile = False
        raw_state = np.zeros(T - 1, dtype=int)
        for t_i in range(T - 1):
            is_transient = trans_f[t_i] > (thresh_t_low if in_tactile else thresh_t_high)
            is_synced = steady_f[t_i] < thresh_s
            is_stationary = flow_f[t_i] < flow_p50

            if is_transient or (is_synced and is_stationary):
                in_tactile = True
                raw_state[t_i] = 1
            elif in_tactile:
                # Once in tactile, stay in tactile as long as the finger
                # remains relatively still (below median flow) — this is
                # the key for sustained grasps
                if is_stationary:
                    raw_state[t_i] = 1
                else:
                    in_tactile = False

        # --- Step 3: Morphological cleanup ---
        # Remove short tactile events (< min_event_frames)
        clean_state = raw_state.copy()
        i = 0
        while i < len(clean_state):
            if clean_state[i] == 1:
                start = i
                while i < len(clean_state) and clean_state[i] == 1:
                    i += 1
                if i - start < min_event_frames:
                    clean_state[start:i] = 0
            else:
                i += 1

        # Fill short gaps between tactile events (< min_gap_frames)
        i = 0
        while i < len(clean_state):
            if clean_state[i] == 0:
                start = i
                while i < len(clean_state) and clean_state[i] == 0:
                    i += 1
                if (i - start < min_gap_frames
                        and start > 0 and i < len(clean_state)
                        and clean_state[start - 1] == 1 and clean_state[i] == 1):
                    clean_state[start:i] = 1
            else:
                i += 1

        state[:, f] = clean_state

        # Extract events from cleaned state
        events = []
        i = 0
        while i < len(clean_state):
            if clean_state[i] == 1:
                start = i
                while i < len(clean_state) and clean_state[i] == 1:
                    i += 1
                events.append((start, i - 1))
            else:
                i += 1
        contact_events[f] = events

    # Save contact state
    np.savez(os.path.join(args.output_dir, "contact_state.npz"),
             state=state, finger_names=finger_names)

    # --- 6a: Contact timeline ---
    fig, axes = plt.subplots(n_fingers + 1, 1, figsize=(18, 14),
                              gridspec_kw={"height_ratios": [1] * n_fingers + [2]})
    state_cmap = {0: (0.85, 0.85, 0.85), 1: (1.0, 0.2, 0.1)}
    for f in range(n_fingers):
        ax = axes[f]
        for t_i in range(T - 1):
            ax.axvspan(t_i, t_i + 1, color=state_cmap[state[t_i, f]], alpha=0.8)
        ax.set_xlim(0, T - 1)
        ax.set_yticks([])
        ax.set_ylabel(finger_names[f], fontsize=8, rotation=0, ha="right", va="center")
        if f == 0:
            ax.set_title("Tactile State Timeline (gray=Free, red=Tactile)", fontsize=11)
        if f < n_fingers - 1:
            ax.set_xticks([])
    ax_bot = axes[-1]
    total_transient = transient.sum(axis=1)
    ax_bot.fill_between(np.arange(T - 1), total_transient, alpha=0.3, color="coral")
    ax_bot.plot(np.arange(T - 1), total_transient, color="red", linewidth=0.8,
                label="Total transient (all fingers)")
    ax_bot.set_xlabel("Frame")
    ax_bot.set_ylabel("Contact\nIntensity")
    ax_bot.set_xlim(0, T - 1)
    ax_bot.legend(fontsize=8)
    ax_bot.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "contact_timeline.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: contact_timeline.png")

    # --- 6b: Contact video ---
    if args.save_video:
        print("Writing contact video...")
        contact_video = os.path.join(args.output_dir, "contact_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(contact_video, fourcc, 30, (img_w, img_h))

        for t_i in tqdm(range(T - 1), desc="Contact video"):
            frame_vis = frames[t_i + 1].copy()
            for f in range(n_fingers):
                cx, cy = fingertip_coords[t_i + 1][f]
                if state[t_i, f] == 1:  # TACTILE — big red with label
                    cv2.circle(frame_vis, (int(cx), int(cy)), 14, (255, 30, 30), -1)
                    cv2.circle(frame_vis, (int(cx), int(cy)), 15, (255, 255, 255), 2)
                    cv2.putText(frame_vis, "TACTILE", (int(cx) - 30, int(cy) - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 50, 50), 1)
                else:  # FREE — small green
                    cv2.circle(frame_vis, (int(cx), int(cy)), 5, (50, 200, 50), -1)

            n_tactile = (state[t_i] == 1).sum()
            status = f"Frame {t_i:04d} | Tactile: {n_tactile} fingers"
            cv2.putText(frame_vis, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(frame_vis, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
            writer.write(cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR))

        writer.release()
        print(f"  Saved: {contact_video}")

    # ── Step 7: GT tactile comparison (in-lab only) ──────────────────
    if gt_tactile is not None:
        print("Generating GT vs predicted comparison...")
        gt_T = min(T - 1, gt_tactile["delta_force"].shape[0])
        time_ax = np.arange(gt_T)

        def _normalize_sig(s):
            s = s - s.min()
            return s / (s.max() + 1e-8)

        total_transient_sig = transient[:gt_T].sum(axis=1)
        gt_total = np.abs(gt_tactile["delta_force"][:gt_T]).max(axis=1)
        pred_norm = _normalize_sig(total_transient_sig)
        gt_norm = _normalize_sig(gt_total)
        corr = np.corrcoef(gt_norm, pred_norm)[0, 1]

        fig = plt.figure(figsize=(18, 14))
        gs_gt = GridSpec(4, 1, height_ratios=[1.2, 1, 1, 1], hspace=0.35)

        # GT force
        ax_f = fig.add_subplot(gs_gt[0])
        gt_hand = gt_tactile["hand"]
        for i in range(5):
            ax_f.plot(time_ax, gt_tactile["delta_force"][:gt_T, i],
                      label=f"{gt_hand[0].upper()}_{FINGER_NAMES_PER_HAND[i]}",
                      color=np.array(FINGER_COLORS_RIGHT[i]) / 255, linewidth=1.5)
        ax_f.axhline(2, color="gray", linestyle="--", alpha=0.5)
        ax_f.axhline(-2, color="gray", linestyle="--", alpha=0.5)
        ax_f.set_ylabel("Force Delta")
        ax_f.set_title(f"GT: Tactile Force ({gt_hand} hand) — delta from baseline", fontsize=12)
        ax_f.legend(fontsize=8, ncol=5, loc="upper right")
        ax_f.grid(True, linestyle=":", alpha=0.5)

        # GT deform
        ax_d = fig.add_subplot(gs_gt[1])
        for i in range(5):
            ax_d.plot(time_ax, gt_tactile["deform_diff"][:gt_T, i],
                      label=FINGER_NAMES_PER_HAND[i],
                      color=np.array(FINGER_COLORS_RIGHT[i]) / 255, linewidth=1.5)
        ax_d.set_ylabel("Deform Change")
        ax_d.set_title("GT: Tactile Deformation Change", fontsize=12)
        ax_d.legend(fontsize=8, ncol=5, loc="upper right")
        ax_d.grid(True, linestyle=":", alpha=0.5)

        # Predicted transient
        ax_p = fig.add_subplot(gs_gt[2])
        ax_p.fill_between(time_ax, total_transient_sig, alpha=0.3, color="coral")
        ax_p.plot(time_ax, total_transient_sig, color="red", linewidth=1.0)
        ax_p.set_ylabel("Flow Transient")
        ax_p.set_title("Predicted: Optical Flow Transient (proxy for contact)", fontsize=12)
        ax_p.grid(True, linestyle=":", alpha=0.5)

        # Normalized overlay
        ax_o = fig.add_subplot(gs_gt[3])
        ax_o.plot(time_ax, gt_norm, color="blue", linewidth=1.5, label="GT contact intensity", alpha=0.8)
        ax_o.plot(time_ax, pred_norm, color="red", linewidth=1.5, label="Flow transient", alpha=0.8)
        ax_o.set_title(f"Normalized Comparison — Pearson r = {corr:.3f}", fontsize=12)
        ax_o.set_xlabel("Frame")
        ax_o.set_ylabel("Normalized")
        ax_o.legend(fontsize=8)
        ax_o.grid(True, linestyle=":", alpha=0.5)

        plt.savefig(os.path.join(args.output_dir, "gt_vs_predicted_contact.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: gt_vs_predicted_contact.png (correlation={corr:.3f})")

        # Compute detection metrics
        any_gt = gt_tactile["gt_contact"][:gt_T].any(axis=1)
        pred_contact_binary = total_transient_sig > (
            np.median(total_transient_sig) + 3 * max(np.median(
                np.abs(total_transient_sig - np.median(total_transient_sig))), 0.1))
        tp = (pred_contact_binary & any_gt).sum()
        fp = (pred_contact_binary & ~any_gt).sum()
        fn = (~pred_contact_binary & any_gt).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        print(f"  Detection: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

    # ── Summary stats ───────────────────────────────────────────────────
    total_tactile_frames = (state == 1).any(axis=1).sum()
    print("\n=== Summary ===")
    print(f"  Frames processed:    {T}")
    print(f"  Flow fields:         {len(flows)}")
    print(f"  Data source:         {data_source}")
    print(f"  Frames with tactile: {total_tactile_frames} ({100*total_tactile_frames/(T-1):.1f}%)")
    for f in range(n_fingers):
        print(f"  {finger_names[f]:10s}  flow_mag: "
              f"mean={flow_mags[:, f].mean():.2f}, max={flow_mags[:, f].max():.2f}  |  "
              f"transient: mean={transient[:, f].mean():.2f}, max={transient[:, f].max():.2f}  |  "
              f"steady: mean={steady[:, f].mean():.2f}, max={steady[:, f].max():.2f}  |  "
              f"contacts: {len(contact_events[f])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optical-Flow Tactile Proxy — PoC Visualization")
    parser.add_argument("--episode_dir", type=str, required=True,
                        help="Path to an EgoDex episode directory (contains ego_view.mp4)")
    parser.add_argument("--raw_h5_path", type=str, default="",
                        help="Path to raw EgoDex .hdf5 with 4D tracking + camera intrinsics. "
                             "Auto-resolved from episode_dir if omitted.")
    parser.add_argument("--output_dir", type=str, default="./flow_tactile_vis",
                        help="Where to save outputs")
    parser.add_argument("--max_frames", type=int, default=300,
                        help="Max frames to process (0=all)")
    parser.add_argument("--frame_stride", type=int, default=1,
                        help="Process every N-th frame from the video")
    parser.add_argument("--hand", type=str, default="both",
                        choices=["left", "right", "both"],
                        help="Which hand(s) to track")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_panel_frames", type=int, default=6,
                        help="Number of sampled frames in the panel visualization")
    parser.add_argument("--save_video", action="store_true",
                        help="Also save an annotated video (RGB + flow stacked)")

    args = parser.parse_args()
    if args.max_frames == 0:
        args.max_frames = None
    run_pipeline(args)
