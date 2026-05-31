"""Shared schema + math for converting T-Rex raw data into LeRobot v3.0 and
loading it back.

This module is the single source of truth for:
  * the canonical T-Rex LeRobot **feature schema** (`build_trex_features`),
  * the **pose / delta-base action math** (copied verbatim from the in-lab
    gen_json transform so converters reproduce it byte-for-byte), and
  * the **q01/q99 normalization stats** (`NormStatsAccumulator`) written to a
    `meta/trex_norm_stats.json` sidecar — LeRobot v3.0's native stats only carry
    min/max/mean/std, but T-Rex normalizes with q01/q99 percentile min-max.

Bimanual layout (62-D): Left arm(9D eef) + Left hand(22D) + Right arm(9D) + Right hand(22D).
Tactile F6 is [10 fingers, 6 channels] (left 5 then right 5).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np

# ── canonical dims ───────────────────────────────────────────────────────────
ACTION_DIM        = 62
ACTION_CHUNK      = 16
FRAME_STRIDE      = 1
N_HANDS           = 2
N_FINGERS_PER_HAND = 5
F6_PER_FINGER     = 6
N_FINGERS         = N_HANDS * N_FINGERS_PER_HAND          # 10
F6_DIM            = N_FINGERS * F6_PER_FINGER             # 60
TRACKING_ERROR_DIM = 56                                   # 28 per arm
STATS_KEY         = "rlbench"                             # top-level key (matches existing _statistics.json)

# Feature key names (kept in one place so the loader and converters agree).
KEY_HEAD       = "observation.images.head"
KEY_WRIST_R    = "observation.images.wrist_right"
KEY_WRIST_L    = "observation.images.wrist_left"
KEY_STATE      = "observation.state"
KEY_ACTION     = "action"            # baked delta-base chunk [ACTION_CHUNK, 62]
KEY_ACTION_ABS = "action_abs"        # absolute target [62] (for tracking-error / state noise)
KEY_TACF6      = "observation.tactile_f6"   # [10, 6]
# 10 per-finger deform video keys, left 0..4 then right 0..4 (matches gen_json order).
DEFORM_KEYS = ([f"observation.tactile_deform.l{i}" for i in range(N_FINGERS_PER_HAND)]
               + [f"observation.tactile_deform.r{i}" for i in range(N_FINGERS_PER_HAND)])


# ── pose / delta-base action math (verbatim from gen_json_tac_deltabase_eef_bimanual) ──
def pose_matrix_to_9d(pose_matrices: np.ndarray) -> np.ndarray:
    """[N,4,4] -> [N,9] = translation(3) + rot col1(3) + rot col2(3)."""
    trans = pose_matrices[:, :3, 3]
    rot_col1 = pose_matrices[:, :3, 0]
    rot_col2 = pose_matrices[:, :3, 1]
    return np.concatenate([trans, rot_col1, rot_col2], axis=-1)


def get_rot_mat(vec6d: np.ndarray) -> np.ndarray:
    col1 = vec6d[:3]
    col2 = vec6d[3:6]
    col3 = np.cross(col1, col2)
    return np.column_stack([col1, col2, col3])


def compute_chunk_delta_pose(curr_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    """Delta-base arm action: target pose expressed in the chunk-start frame.
    Both inputs are 4x4. Returns 9D = delta_xyz(3) + delta rot col1(3) + col2(3)."""
    R_curr = curr_pose[:3, :3]
    t_curr = curr_pose[:3, 3]
    R_targ = target_pose[:3, :3]
    t_targ = target_pose[:3, 3]
    delta_xyz = R_curr.T @ (t_targ - t_curr)
    R_delta = R_curr.T @ R_targ
    return np.concatenate([delta_xyz, R_delta[:3, 0], R_delta[:3, 1]])


def compute_tracking_error_axis_angle(state_31d: np.ndarray, target_31d: np.ndarray) -> np.ndarray:
    s_arm = state_31d[:9]
    t_arm = target_31d[:9]
    t_state = s_arm[:3]
    R_state = get_rot_mat(s_arm[3:9])
    t_targ = t_arm[:3]
    R_targ = get_rot_mat(t_arm[3:9])
    delta_xyz = R_state.T @ (t_targ - t_state)
    R_delta = R_state.T @ R_targ
    delta_rot_axis_angle, _ = cv2.Rodrigues(R_delta)
    delta_rot_axis_angle = delta_rot_axis_angle.flatten()
    arm_error_6d = np.concatenate([delta_xyz, delta_rot_axis_angle])
    hand_error_22d = target_31d[9:] - state_31d[9:]
    return np.concatenate([arm_error_6d, hand_error_22d])


def compute_bimanual_tracking_error(state_62d: np.ndarray, target_62d: np.ndarray) -> np.ndarray:
    """56D = 28D left + 28D right."""
    left_err = compute_tracking_error_axis_angle(state_62d[:31], target_62d[:31])
    right_err = compute_tracking_error_axis_angle(state_62d[31:], target_62d[31:])
    return np.concatenate([left_err, right_err])


def build_action_chunk(
    s_l_arm_pose: np.ndarray, a_l_arm_pose: np.ndarray, a_l_hnd: np.ndarray,
    s_r_arm_pose: np.ndarray, a_r_arm_pose: np.ndarray, a_r_hnd: np.ndarray,
    i: int, min_len: int,
) -> np.ndarray:
    """Build the [ACTION_CHUNK, 62] delta-base chunk for frame `i`.
    Mirrors the gen_json loop exactly (FRAME_STRIDE applied)."""
    base_l = s_l_arm_pose[i]
    base_r = s_r_arm_pose[i]
    chunk = np.empty((ACTION_CHUNK, ACTION_DIM), dtype=np.float32)
    for k in range(ACTION_CHUNK):
        base_future_idx = min(i + k * FRAME_STRIDE, min_len - 1)
        fut = min(base_future_idx + FRAME_STRIDE - 1, min_len - 1)
        dl = compute_chunk_delta_pose(base_l, a_l_arm_pose[fut])
        dr = compute_chunk_delta_pose(base_r, a_r_arm_pose[fut])
        chunk[k] = np.concatenate([dl, a_l_hnd[fut], dr, a_r_hnd[fut]])
    return chunk


# ── feature schema ─────────────────────────────────────────────────────────--
def _video(shape):
    return {"dtype": "video", "shape": tuple(shape), "names": ["channels", "height", "width"]}


def _num(shape):
    return {"dtype": "float32", "shape": tuple(shape), "names": None}


def build_trex_features(
    head_shape: tuple,                      # (3, H, W) after any crop
    include_wrist: bool = True,
    wrist_shape: Optional[tuple] = None,    # (3, H, W)
    include_tactile: bool = True,
    deform_shape: Optional[tuple] = None,   # (3, H, W) — grayscale replicated to 3ch
    include_action_abs: bool = True,        # False for pretrain (no abs target on disk)
) -> Dict[str, dict]:
    """Canonical T-Rex LeRobot v3.0 feature dict.

    Stage presence:
      * pretrain (egodex/mecka): include_wrist=False, include_tactile=False,
        include_action_abs=False (head + state + baked action only).
      * midtrain / posttrain:    all True (3 cams + tactile_f6 + 10 deform videos).
    """
    feats: Dict[str, dict] = {
        KEY_HEAD:       _video(head_shape),
        KEY_STATE:      _num((ACTION_DIM,)),
        KEY_ACTION:     _num((ACTION_CHUNK, ACTION_DIM)),
    }
    if include_action_abs:
        feats[KEY_ACTION_ABS] = _num((ACTION_DIM,))
    if include_wrist:
        ws = wrist_shape or head_shape
        feats[KEY_WRIST_R] = _video(ws)
        feats[KEY_WRIST_L] = _video(ws)
    if include_tactile:
        feats[KEY_TACF6] = _num((N_FINGERS, F6_PER_FINGER))
        ds = deform_shape
        if ds is None:
            raise ValueError("deform_shape required when include_tactile=True")
        for k in DEFORM_KEYS:
            feats[k] = _video(ds)
    return feats


# ── q01/q99 normalization sidecar (mirrors gen_json `cal_stats`) ──────────────
def calculate_stats(data: np.ndarray, mask: Optional[List[bool]] = None) -> dict:
    if mask is None:
        mask = [True] * data.shape[-1]
    return {
        "mean": np.mean(data, axis=0).tolist(),
        "std":  np.std(data, axis=0).tolist(),
        "max":  np.max(data, axis=0).tolist(),
        "min":  np.min(data, axis=0).tolist(),
        "q01":  np.quantile(data, 0.01, axis=0).tolist(),
        "q99":  np.quantile(data, 0.99, axis=0).tolist(),
        "mask": mask,
    }


class NormStatsAccumulator:
    """Accumulates per-frame action(chunk)/state/tactile_f6 + per-episode
    tracking errors across a whole dataset, then writes meta/trex_norm_stats.json
    in the same schema SftDataset expects (`stats_data[STATS_KEY][...]`)."""

    def __init__(self):
        self._actions: List[np.ndarray] = []     # each [ACTION_CHUNK, 62]
        self._states:  List[np.ndarray] = []     # each [62]
        self._tac:     List[np.ndarray] = []     # each [60]
        self._track:   List[np.ndarray] = []     # each [56]
        self.num_traj = 0

    def add_frame(self, action_chunk, state, tactile_f6=None):
        self._actions.append(np.asarray(action_chunk, dtype=np.float32))
        self._states.append(np.asarray(state, dtype=np.float32))
        if tactile_f6 is not None:
            self._tac.append(np.asarray(tactile_f6, dtype=np.float32).reshape(-1))

    def add_episode_tracking(self, states_62: np.ndarray, abs_targets_62: np.ndarray):
        """states_62/abs_targets_62: [N,62]; err[t] from (state[t], abs_target[t-1])."""
        self.num_traj += 1
        for t in range(1, len(states_62)):
            self._track.append(
                compute_bimanual_tracking_error(states_62[t], abs_targets_62[t - 1]))

    def add_tracking_errors(self, errors_56: np.ndarray):
        """Add precomputed [K,56] tracking errors (e.g. egodex pretrain.hdf5)."""
        for e in np.asarray(errors_56, dtype=np.float32):
            self._track.append(e)

    def assemble(self) -> dict:
        # Action stats are kept PER-(step,dim): np.*(axis=0) over [M,16,62] -> [16,62],
        # exactly as gen_json `cal_stats` and the midtrain stats do.  SftDataset's
        # _normalize broadcasts these [16,62] q01/q99 against the [B,16,62] chunk.
        actions = np.asarray(self._actions, dtype=np.float32)        # [M, 16, 62]
        states = np.asarray(self._states, dtype=np.float32)          # [M, 62]
        block = {
            "action": calculate_stats(actions, [True] * ACTION_DIM),
            "state":  calculate_stats(states, [True] * ACTION_DIM),
            "num_transitions": int(states.shape[0]),
            "num_trajectories": int(self.num_traj),
        }
        if self._tac:
            tac = np.asarray(self._tac, dtype=np.float32)             # [M, 60]
            block["tactile_f6"] = calculate_stats(tac, [True] * F6_DIM)
        if self._track:
            tr = np.asarray(self._track, dtype=np.float32)            # [K, 56]
            block["tracking_error"] = {
                "mean": np.mean(tr, axis=0).tolist(),
                "std":  np.std(tr, axis=0).tolist(),
                "mean_abs": np.mean(np.abs(tr), axis=0).tolist(),
                "mask": [True] * TRACKING_ERROR_DIM,
            }
        return {STATS_KEY: block}

    def write(self, dataset_root: str) -> str:
        out = self.assemble()
        path = os.path.join(dataset_root, "meta", "trex_norm_stats.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        return path


NORM_STATS_FILENAME = os.path.join("meta", "trex_norm_stats.json")


def load_norm_stats(dataset_root: str) -> dict:
    """Read the q01/q99 sidecar back (used by the loader)."""
    path = os.path.join(dataset_root, NORM_STATS_FILENAME)
    with open(path, "r") as f:
        return json.load(f)
