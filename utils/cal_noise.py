"""
Compute wrist tracking noise statistics from demo data.

Traverses all demos in the two data roots, and for each timestep computes:
  target_pose[t] (commanded) vs observation_pose[t+1] (actual next step).
Aggregates tracking errors to produce separate noise stats for:
  - Left:  trans_xyz (3 dim) + rot_wxyz (4 dim) = 7 dim
  - Right: trans_xyz (3 dim) + rot_wxyz (4 dim) = 7 dim
"""

import json
import os
import glob
import numpy as np
import h5py
from scipy.spatial.transform import Rotation
from tqdm import tqdm


# Data roots (expect ~200 demos total)
DATA_ROOTS = [
    "/home/apai25/data/caip/20260120_pick_orange_cube_rachel_100_transfer",
    "/home/apai25/data/caip/20260122_handover_rl_orange_cube_rachel_105",
    "/home/apai25/data/caip/20260205_pick_orange_cube_rachel_100/success",
]


def pose_4x4_to_xyz_quat_wxyz(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert 4x4 rigid transform to translation and quaternion (wxyz).

    pose: (T, 4, 4) or (4, 4) — standard 4x4 (R | t; 0 0 0 1)
    - xyz from pose[:3, 3]
    - R = pose[:3, :3] -> scipy Rotation -> .as_quat() (xyzw) -> convert to wxyz
    Returns:
        xyz: (T, 3) or (3,)
        quat_wxyz: (T, 4) or (4,) in wxyz order
    """
    single = pose.ndim == 2
    if single:
        pose = pose[None, ...]
    xyz = pose[:, :3, 3]
    R = pose[:, :3, :3]
    quat_xyzw = Rotation.from_matrix(R).as_quat()  # scipy (x,y,z,w)
    quat_wxyz = np.concatenate([
        quat_xyzw[:, 3:4],
        quat_xyzw[:, :3],
    ], axis=1)
    if single:
        return xyz[0], quat_wxyz[0]
    return xyz, quat_wxyz


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 4) xyzw for scipy."""
    if q.ndim == 1:
        return np.array([q[1], q[2], q[3], q[0]])
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def quat_rotation_error_wxyz(q_obs_wxyz: np.ndarray, q_target_wxyz: np.ndarray) -> np.ndarray:
    """
    Rotation error as quaternion: q_err = q_obs * q_target^{-1}.
    Returns quat in wxyz, shape (..., 4).
    """
    single = q_obs_wxyz.ndim == 1
    if single:
        q_obs_wxyz = q_obs_wxyz[None, :]
        q_target_wxyz = q_target_wxyz[None, :]
    q_obs = Rotation.from_quat(quat_wxyz_to_xyzw(q_obs_wxyz))
    q_tar = Rotation.from_quat(quat_wxyz_to_xyzw(q_target_wxyz))
    q_err = q_obs * q_tar.inv()
    quat_xyzw = q_err.as_quat()
    quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1)
    if single:
        return quat_wxyz[0]
    return quat_wxyz


HAND_JOINT_DIM = 22


def collect_errors_from_h5(h5_path: str) -> tuple:
    """
    For each t, use target[t] and current[t+1].
    Returns:
        trans_err_left (N, 3), trans_err_right (N, 3),
        rot_err_left (N, 4) wxyz, rot_err_right (N, 4) wxyz,
        joint_err_left (N, 22), joint_err_right (N, 22).
    """
    with h5py.File(h5_path, "r") as f:
        left_target = np.array(f["left_arm_target_pose"])
        left_current = np.array(f["left_arm_current_pose"])
        right_target = np.array(f["right_arm_target_pose"])
        right_current = np.array(f["right_arm_current_pose"])
        left_hand_target = np.array(f["left_hand_target_joint_positions"])
        left_hand_current = np.array(f["left_hand_joint_positions"])
        right_hand_target = np.array(f["right_hand_target_joint_positions"])
        right_hand_current = np.array(f["right_hand_joint_positions"])
    T = left_target.shape[0]
    if T < 2:
        return (
            np.empty((0, 3)),
            np.empty((0, 3)),
            np.empty((0, 4)),
            np.empty((0, 4)),
            np.empty((0, HAND_JOINT_DIM)),
            np.empty((0, HAND_JOINT_DIM)),
        )
    # Demo length T -> exactly T-1 pairs: (target[t], current[t+1]) for t = 0 .. T-2
    n_pairs = T - 1
    left_xyz_tar, left_quat_tar = pose_4x4_to_xyz_quat_wxyz(left_target[:n_pairs])
    left_xyz_obs, left_quat_obs = pose_4x4_to_xyz_quat_wxyz(left_current[1 : n_pairs + 1])
    right_xyz_tar, right_quat_tar = pose_4x4_to_xyz_quat_wxyz(right_target[:n_pairs])
    right_xyz_obs, right_quat_obs = pose_4x4_to_xyz_quat_wxyz(right_current[1 : n_pairs + 1])

    trans_err_left = left_xyz_obs - left_xyz_tar
    trans_err_right = right_xyz_obs - right_xyz_tar
    rot_err_left = quat_rotation_error_wxyz(left_quat_obs, left_quat_tar)
    rot_err_right = quat_rotation_error_wxyz(right_quat_obs, right_quat_tar)

    # Hand joints: target[t] vs current[t+1], ensure 22 dims (pad or truncate)
    def _to_22(x: np.ndarray, n: int) -> np.ndarray:
        x = np.asarray(x).reshape(n, -1)
        d = x.shape[1]
        if d >= HAND_JOINT_DIM:
            return x[:, :HAND_JOINT_DIM]
        out = np.zeros((n, HAND_JOINT_DIM), dtype=x.dtype)
        out[:, :d] = x
        return out

    left_joint_tar = _to_22(left_hand_target[:n_pairs], n_pairs)
    left_joint_obs = _to_22(left_hand_current[1 : n_pairs + 1], n_pairs)
    right_joint_tar = _to_22(right_hand_target[:n_pairs], n_pairs)
    right_joint_obs = _to_22(right_hand_current[1 : n_pairs + 1], n_pairs)
    joint_err_left = left_joint_obs - left_joint_tar
    joint_err_right = right_joint_obs - right_joint_tar

    return (
        trans_err_left,
        trans_err_right,
        rot_err_left,
        rot_err_right,
        joint_err_left,
        joint_err_right,
    )


def find_all_h5_paths(roots: list[str]) -> list[str]:
    out = []
    for root in roots:
        if not os.path.isdir(root):
            print(f"Warning: not a directory, skipping: {root}")
            continue
        # episode_000, episode_001, ... each with one .h5
        pattern = os.path.join(root, "episode_*", "*.h5")
        files = sorted(glob.glob(pattern))
        out.extend(files)
    return out


def compute_noise_stats(
    roots: list[str] | None = None,
    expected_total: int = 200,
) -> dict:
    """
    Traverse all demos, compute tracking errors, return noise statistics.

    Returns:
        noise_stat: dict with
            - "left": dict with "trans_xyz_std" (3,), "rot_wxyz_std" (4,), "hand_joint_std" (22,), and mean_abs variants
            - "right": same
            - "left_7": np.ndarray (7,) = trans_xyz_std concat rot_wxyz_std for wrist randomize
            - "right_7": np.ndarray (7,)
            - "left_22": np.ndarray (22,) = hand joint std
            - "right_22": np.ndarray (22,)
            - "n_demos": int, "n_samples_left": int, "n_samples_right": int
    """
    roots = roots or DATA_ROOTS
    h5_paths = find_all_h5_paths(roots)
    if len(h5_paths) == 0:
        raise FileNotFoundError(f"No h5 files found under {roots}")
    if expected_total and len(h5_paths) != expected_total:
        print(f"Warning: expected {expected_total} demos, found {len(h5_paths)}")

    all_trans_left = []
    all_trans_right = []
    all_rot_left = []
    all_rot_right = []
    all_joint_left = []
    all_joint_right = []

    for path in tqdm(h5_paths, desc="Scanning demos"):
        try:
            te_l, te_r, re_l, re_r, je_l, je_r = collect_errors_from_h5(path)
            if te_l.shape[0] > 0:
                all_trans_left.append(te_l)
                all_trans_right.append(te_r)
                all_rot_left.append(re_l)
                all_rot_right.append(re_r)
                all_joint_left.append(je_l)
                all_joint_right.append(je_r)
        except Exception as e:
            print(f"Skip {path}: {e}")

    if not all_trans_left:
        raise ValueError("No valid error samples from any demo.")

    trans_left = np.concatenate(all_trans_left, axis=0)
    trans_right = np.concatenate(all_trans_right, axis=0)
    rot_left = np.concatenate(all_rot_left, axis=0)
    rot_right = np.concatenate(all_rot_right, axis=0)
    joint_left = np.concatenate(all_joint_left, axis=0)
    joint_right = np.concatenate(all_joint_right, axis=0)

    def _stats(arr: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
        std = np.std(arr, axis=0)
        mean_abs = np.mean(np.abs(arr), axis=0)
        return std, mean_abs

    trans_std_l, trans_ma_l = _stats(trans_left, "trans_left")
    trans_std_r, trans_ma_r = _stats(trans_right, "trans_right")
    rot_std_l, rot_ma_l = _stats(rot_left, "rot_left")
    rot_std_r, rot_ma_r = _stats(rot_right, "rot_right")
    joint_std_l, joint_ma_l = _stats(joint_left, "joint_left")
    joint_std_r, joint_ma_r = _stats(joint_right, "joint_right")

    noise_stat = {
        "left": {
            "trans_xyz_std": trans_std_l,
            "trans_xyz_mean_abs": trans_ma_l,
            "rot_wxyz_std": rot_std_l,
            "rot_wxyz_mean_abs": rot_ma_l,
            "hand_joint_std": joint_std_l,
            "hand_joint_mean_abs": joint_ma_l,
        },
        "right": {
            "trans_xyz_std": trans_std_r,
            "trans_xyz_mean_abs": trans_ma_r,
            "rot_wxyz_std": rot_std_r,
            "rot_wxyz_mean_abs": rot_ma_r,
            "hand_joint_std": joint_std_r,
            "hand_joint_mean_abs": joint_ma_r,
        },
        "left_7": np.concatenate([trans_std_l, rot_std_l]),
        "right_7": np.concatenate([trans_std_r, rot_std_r]),
        "left_22": joint_std_l,
        "right_22": joint_std_r,
        "n_demos": len(h5_paths),
        "n_samples_left": trans_left.shape[0],
        "n_samples_right": trans_right.shape[0],
    }
    return noise_stat


def noise_stat_to_json_serializable(noise_stat: dict) -> dict:
    """Convert noise_stat (numpy arrays) to a dict of lists/numbers for JSON."""
    out = {
        "left_7": noise_stat["left_7"].tolist(),
        "right_7": noise_stat["right_7"].tolist(),
        "left_22": noise_stat["left_22"].tolist(),
        "right_22": noise_stat["right_22"].tolist(),
        "n_demos": int(noise_stat["n_demos"]),
        "n_samples_left": int(noise_stat["n_samples_left"]),
        "n_samples_right": int(noise_stat["n_samples_right"]),
    }
    for side in ("left", "right"):
        s = noise_stat[side]
        out[side] = {
            "trans_xyz_std": np.asarray(s["trans_xyz_std"]).tolist(),
            "trans_xyz_mean_abs": np.asarray(s["trans_xyz_mean_abs"]).tolist(),
            "rot_wxyz_std": np.asarray(s["rot_wxyz_std"]).tolist(),
            "rot_wxyz_mean_abs": np.asarray(s["rot_wxyz_mean_abs"]).tolist(),
            "hand_joint_std": np.asarray(s["hand_joint_std"]).tolist(),
            "hand_joint_mean_abs": np.asarray(s["hand_joint_mean_abs"]).tolist(),
        }
    return out


def save_noise_stat_json(noise_stat: dict, path: str) -> None:
    """Save noise statistics to a JSON file."""
    data = noise_stat_to_json_serializable(noise_stat)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved noise stat to {path}")


def _quat_wxyz_to_euler_xyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 3) euler XYZ radians."""
    xyzw = quat_wxyz_to_xyzw(quat_wxyz)
    return Rotation.from_quat(xyzw).as_euler("XYZ", degrees=False)


def _euler_xyz_to_quat_wxyz(euler: np.ndarray) -> np.ndarray:
    """(..., 3) euler XYZ -> (..., 4) wxyz."""
    quat_xyzw = Rotation.from_euler("XYZ", euler, degrees=False).as_quat()
    return np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., :3]], axis=-1)


def randomize_with_noise_stat(
    transform: np.ndarray,
    noise_stat: dict,
    side: str,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Apply per-hand, per-component noise using precomputed noise stat.

    transform: (N, 7) in order [trans_xyz(3), rot_wxyz(4)] for the given side.
    noise_stat: from compute_noise_stats(), use keys "left_7" or "right_7".
    side: "left" or "right".
    """
    rng = rng or np.random.default_rng()
    key = f"{side}_7"
    std = noise_stat[key]
    trans_std = std[:3]
    rot_std = std[3:7]

    t = transform[:, :3].copy()
    rot = transform[:, 3:7].copy()

    t += rng.uniform(-trans_std, trans_std, size=t.shape)
    rot_euler = _quat_wxyz_to_euler_xyz(rot)
    rot_euler += rng.uniform(-rot_std, rot_std, size=rot_euler.shape)
    rot = _euler_xyz_to_quat_wxyz(rot_euler)

    return np.concatenate([t, rot], axis=1)


if __name__ == "__main__":
    noise = compute_noise_stats(expected_total=200)
    print("Noise statistics (trans_xyz then rot_wxyz):")
    print("  left_7  (std):", np.round(noise["left_7"], 6))
    print("  right_7 (std):", np.round(noise["right_7"], 6))
    print("Hand joint (22 dim) std:")
    print("  left_22 (std):", np.round(noise["left_22"], 6))
    print("  right_22 (std):", np.round(noise["right_22"], 6))
    print("  n_demos:", noise["n_demos"])
    print("  n_samples (left/right):", noise["n_samples_left"], noise["n_samples_right"])
    print("\nPer-side breakdown:")
    for side in ("left", "right"):
        s = noise[side]
        print(f"  {side}: trans_xyz_std={s['trans_xyz_std']}, rot_wxyz_std={s['rot_wxyz_std']}, hand_joint_std (22) in {side}_22")

    out_path = os.path.join(os.path.dirname(__file__), "noise_stat.json")
    save_noise_stat_json(noise, out_path)