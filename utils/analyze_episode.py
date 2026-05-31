"""
Analyze a single randomly sampled episode from the dataset.
Produces:
  1. State curves (arm 9D + hand 22D)
  2. Action curves (delta arm 9D + target hand 22D)
  3. Absolute target action curves
  4. Tactile F6 curves (5 fingers × 6 force/torque)
  5. Tactile deform video (5 fingers side-by-side over time)
  6. Tactile raw video (5 fingers side-by-side over time)
  7. Tracking error curves
"""

import os
import random
import cv2
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Module-level defaults (overridable via CLI in main()).
DATA_ROOT = ""
OUTPUT_DIR = ""
FRAME_STRIDE = 2
SEED = 42

def pose_matrix_to_9d(pose_matrices):
    trans = pose_matrices[:, :3, 3]
    rot_col1 = pose_matrices[:, :3, 0]
    rot_col2 = pose_matrices[:, :3, 1]
    return np.concatenate([trans, rot_col1, rot_col2], axis=-1)


def get_rot_mat(vec6d):
    col1 = vec6d[:3]
    col2 = vec6d[3:6]
    col3 = np.cross(col1, col2)
    return np.column_stack([col1, col2, col3])


def compute_chunk_delta_pose(curr_pose, target_pose):
    R_curr = curr_pose[:3, :3]
    t_curr = curr_pose[:3, 3]
    R_targ = target_pose[:3, :3]
    t_targ = target_pose[:3, 3]
    delta_xyz = R_curr.T @ (t_targ - t_curr)
    R_delta = R_curr.T @ R_targ
    return np.concatenate([delta_xyz, R_delta[:3, 0], R_delta[:3, 1]])


def compute_tracking_error_axis_angle(state_31d, target_31d):
    s_arm = state_31d[:9]
    t_arm = target_31d[:9]
    R_state = get_rot_mat(s_arm[3:9])
    R_targ = get_rot_mat(t_arm[3:9])
    delta_xyz = R_state.T @ (t_arm[:3] - s_arm[:3])
    R_delta = R_state.T @ R_targ
    delta_rot_aa, _ = cv2.Rodrigues(R_delta)
    delta_rot_aa = delta_rot_aa.flatten()
    arm_error_6d = np.concatenate([delta_xyz, delta_rot_aa])
    hand_error_22d = target_31d[9:] - state_31d[9:]
    return np.concatenate([arm_error_6d, hand_error_22d])


# ─── Plotting helpers ─────────────────────────────────────────────────────────
def plot_arm_9d(data, title, filename, ylabel="Value"):
    """Plot 9D arm data: 3 translation + 6 rotation."""
    T = len(data)
    t = np.arange(T)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=14)

    # Translation (x, y, z)
    labels_trans = ['x', 'y', 'z']
    for i in range(3):
        axes[0].plot(t, data[:, i], label=labels_trans[i])
    axes[0].set_ylabel(f"Translation {ylabel}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Rotation col1
    labels_r1 = ['r1_x', 'r1_y', 'r1_z']
    for i in range(3):
        axes[1].plot(t, data[:, 3 + i], label=labels_r1[i])
    axes[1].set_ylabel(f"Rot col1 {ylabel}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Rotation col2
    labels_r2 = ['r2_x', 'r2_y', 'r2_z']
    for i in range(3):
        axes[2].plot(t, data[:, 6 + i], label=labels_r2[i])
    axes[2].set_ylabel(f"Rot col2 {ylabel}")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel("Timestep")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_hand_22d(data, title, filename, ylabel="Joint Angle"):
    """Plot 22D hand joints: split into 4 subplots of ~6 joints each."""
    T = len(data)
    t = np.arange(T)
    n_joints = data.shape[1]
    splits = [range(0, 6), range(6, 12), range(12, 17), range(17, 22)]
    split_names = ["Joints 0-5", "Joints 6-11", "Joints 12-16", "Joints 17-21"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(title, fontsize=14)

    for ax, idxs, name in zip(axes, splits, split_names):
        for j in idxs:
            ax.plot(t, data[:, j], label=f"j{j}")
        ax.set_ylabel(ylabel)
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_tactile_f6(tac_f6, title, filename):
    """Plot tactile F6 for 5 fingers × 6 channels."""
    T = tac_f6.shape[0]
    t = np.arange(T)
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    channel_names = ["fx", "fy", "fz", "tx", "ty", "tz"]

    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(title, fontsize=14)

    for fi in range(5):
        for ci in range(6):
            axes[fi].plot(t, tac_f6[:, fi, ci], label=channel_names[ci])
        axes[fi].set_ylabel(finger_names[fi])
        axes[fi].legend(fontsize=7, ncol=6)
        axes[fi].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_tracking_error(errors, title, filename):
    """Plot 28D tracking error: 3 pos + 3 rot(axis-angle) + 22 hand."""
    T = len(errors)
    t = np.arange(T)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(title, fontsize=14)

    # Position error
    for i, lbl in enumerate(['dx', 'dy', 'dz']):
        axes[0].plot(t, errors[:, i], label=lbl)
    axes[0].set_ylabel("Position Error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Rotation error (axis-angle)
    for i, lbl in enumerate(['aa_x', 'aa_y', 'aa_z']):
        axes[1].plot(t, errors[:, 3 + i], label=lbl)
    axes[1].set_ylabel("Rotation Error (axis-angle)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Hand error (split into two)
    for j in range(6, 17):
        axes[2].plot(t, errors[:, j], label=f"j{j-6}")
    axes[2].set_ylabel("Hand Error (j0-j10)")
    axes[2].legend(fontsize=7, ncol=4)
    axes[2].grid(True, alpha=0.3)

    for j in range(17, 28):
        axes[3].plot(t, errors[:, j], label=f"j{j-6}")
    axes[3].set_ylabel("Hand Error (j11-j21)")
    axes[3].legend(fontsize=7, ncol=4)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlabel("Timestep")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def make_tactile_video(tactile_data, filename, fps=30, label="tactile"):
    """
    tactile_data: (T, 5, H, W) uint8
    Creates a video with 5 finger images side-by-side per frame.
    """
    T, n_fingers, H, W = tactile_data.shape
    canvas_w = W * n_fingers
    canvas_h = H + 30  # extra space for labels

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (canvas_w, canvas_h))

    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    for ti in range(T):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        for fi in range(n_fingers):
            img = tactile_data[ti, fi]
            # Handle both grayscale and color
            if img.ndim == 2:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img
            x_off = fi * W
            canvas[30:30 + H, x_off:x_off + W] = img_bgr
            cv2.putText(canvas, finger_names[fi], (x_off + 5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Frame number
        cv2.putText(canvas, f"t={ti}/{T}", (canvas_w - 120, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        out.write(canvas)

    out.release()
    print(f"  Saved: {filename}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    import argparse
    global DATA_ROOT, OUTPUT_DIR, FRAME_STRIDE, SEED
    p = argparse.ArgumentParser(description="Analyse a single randomly sampled episode from the dataset.")
    p.add_argument("--data_root", required=True, help="Dataset root containing a 'success/' subdir of episode_* dirs.")
    p.add_argument("--output_dir", required=True, help="Where plots and videos will be written.")
    p.add_argument("--frame_stride", type=int, default=FRAME_STRIDE)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    DATA_ROOT = args.data_root
    OUTPUT_DIR = args.output_dir
    FRAME_STRIDE = args.frame_stride
    SEED = args.seed
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Randomly sample one episode
    episodes_root = os.path.join(DATA_ROOT, "success")
    episodes = sorted([
        d for d in os.listdir(episodes_root)
        if os.path.isdir(os.path.join(episodes_root, d)) and d.startswith("episode_")
    ])
    random.seed(SEED)
    episode_name = random.choice(episodes)
    episode_dir = os.path.join(episodes_root, episode_name)
    h5_path = os.path.join(episode_dir, f"{episode_name}.h5")
    print(f"Sampled episode: {episode_name}")
    print(f"H5 path: {h5_path}")

    # ── Load data ──
    with h5py.File(h5_path, 'r') as f:
        s_r_arm_pose = f['right_arm_current_pose'][:]       # (N, 4, 4)
        s_r_hnd = f['right_hand_joint_positions'][:]         # (N, 22)
        a_r_arm_pose = f['right_arm_target_pose'][:]         # (N, 4, 4)
        a_r_hnd = f['right_hand_target_joint_positions'][:]  # (N, 22)
        t_r_f6 = f['right_hand_tactile_f6'][:]               # (N, 5, 6)
        t_r_deform = f['right_hand_tactile_deform'][:]       # (N, 5, 240, 240)
        t_r_raw = f['right_hand_tactile_raw'][:]             # (N, 5, 240, 320)

    N = len(s_r_arm_pose)
    print(f"Episode length: {N} timesteps\n")

    # ── Compute derived quantities ──
    s_r_arm_9d = pose_matrix_to_9d(s_r_arm_pose)     # (N, 9)
    a_r_arm_9d = pose_matrix_to_9d(a_r_arm_pose)     # (N, 9)
    states = np.concatenate([s_r_arm_9d, s_r_hnd], axis=1)            # (N, 31)
    absolute_targets = np.concatenate([a_r_arm_9d, a_r_hnd], axis=1)  # (N, 31)

    # Delta actions (relative to current pose, strided)
    delta_actions = np.zeros((N, 31))
    for i in range(N):
        tgt_idx = min(i + FRAME_STRIDE - 1, N - 1)
        delta_9d = compute_chunk_delta_pose(s_r_arm_pose[i], a_r_arm_pose[tgt_idx])
        delta_actions[i] = np.concatenate([delta_9d, a_r_hnd[tgt_idx]])

    # Tracking error
    tracking_errors = np.zeros((N - 1, 28))
    for t in range(1, N):
        tracking_errors[t - 1] = compute_tracking_error_axis_angle(states[t], absolute_targets[t - 1])

    # ══════════════════════════════════════════════════════════════════════════
    # 1. STATE CURVES
    # ══════════════════════════════════════════════════════════════════════════
    print("[1/7] Plotting state curves...")
    plot_arm_9d(
        states[:, :9],
        f"State - Arm 9D (current pose) [{episode_name}]",
        os.path.join(OUTPUT_DIR, "state_arm_9d.png"),
    )
    plot_hand_22d(
        states[:, 9:],
        f"State - Hand 22D (joint positions) [{episode_name}]",
        os.path.join(OUTPUT_DIR, "state_hand_22d.png"),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 2. ABSOLUTE TARGET ACTION CURVES
    # ══════════════════════════════════════════════════════════════════════════
    print("[2/7] Plotting absolute target action curves...")
    plot_arm_9d(
        absolute_targets[:, :9],
        f"Absolute Target - Arm 9D (target pose) [{episode_name}]",
        os.path.join(OUTPUT_DIR, "abs_target_arm_9d.png"),
    )
    plot_hand_22d(
        absolute_targets[:, 9:],
        f"Absolute Target - Hand 22D (target joints) [{episode_name}]",
        os.path.join(OUTPUT_DIR, "abs_target_hand_22d.png"),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 3. DELTA ACTION CURVES (relative to current frame)
    # ══════════════════════════════════════════════════════════════════════════
    print("[3/7] Plotting delta action curves...")
    plot_arm_9d(
        delta_actions[:, :9],
        f"Delta Action - Arm 9D (delta curr→target) [{episode_name}]",
        os.path.join(OUTPUT_DIR, "delta_action_arm_9d.png"),
        ylabel="Delta",
    )
    plot_hand_22d(
        delta_actions[:, 9:],
        f"Delta Action - Hand 22D (target joints) [{episode_name}]",
        os.path.join(OUTPUT_DIR, "delta_action_hand_22d.png"),
        ylabel="Target Joint Angle",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 4. STATE vs TARGET OVERLAY (to see tracking behavior)
    # ══════════════════════════════════════════════════════════════════════════
    print("[4/7] Plotting state vs target overlay...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"State vs Target - Arm Translation [{episode_name}]", fontsize=14)
    t_arr = np.arange(N)
    for i, lbl in enumerate(['x', 'y', 'z']):
        axes[i].plot(t_arr, states[:, i], label=f'state_{lbl}', linestyle='-')
        axes[i].plot(t_arr, absolute_targets[:, i], label=f'target_{lbl}', linestyle='--')
        axes[i].set_ylabel(lbl)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "state_vs_target_translation.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")

    # ══════════════════════════════════════════════════════════════════════════
    # 5. TACTILE F6 CURVES
    # ══════════════════════════════════════════════════════════════════════════
    print("[5/7] Plotting tactile F6 curves...")
    plot_tactile_f6(
        t_r_f6,
        f"Tactile F6 - Right Hand (5 fingers × 6 channels) [{episode_name}]",
        os.path.join(OUTPUT_DIR, "tactile_f6.png"),
    )

    # Also plot per-channel across all fingers for comparison
    fig, axes = plt.subplots(6, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(f"Tactile F6 - Per Channel Across Fingers [{episode_name}]", fontsize=14)
    channel_names = ["fx", "fy", "fz", "tx", "ty", "tz"]
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    for ci in range(6):
        for fi in range(5):
            axes[ci].plot(t_arr, t_r_f6[:, fi, ci], label=finger_names[fi])
        axes[ci].set_ylabel(channel_names[ci])
        axes[ci].legend(fontsize=7, ncol=5)
        axes[ci].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "tactile_f6_per_channel.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")

    # ══════════════════════════════════════════════════════════════════════════
    # 6. TRACKING ERROR CURVES
    # ══════════════════════════════════════════════════════════════════════════
    print("[6/7] Plotting tracking error curves...")
    plot_tracking_error(
        tracking_errors,
        f"Tracking Error (state[t] vs target[t-1]) [{episode_name}]",
        os.path.join(OUTPUT_DIR, "tracking_error.png"),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 7. TACTILE DEFORM & RAW VIDEOS
    # ══════════════════════════════════════════════════════════════════════════
    print("[7/7] Generating tactile videos...")
    make_tactile_video(
        t_r_deform,
        os.path.join(OUTPUT_DIR, "tactile_deform.mp4"),
        fps=30,
        label="deform",
    )
    make_tactile_video(
        t_r_raw,
        os.path.join(OUTPUT_DIR, "tactile_raw.mp4"),
        fps=30,
        label="raw",
    )

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Analysis complete for {episode_name}")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")
    print("Files generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_mb = os.path.getsize(fpath) / 1024 / 1024
        print(f"  {f:45s} {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
