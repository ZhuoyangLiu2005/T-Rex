"""
Qwen3-VL MoT mid-training with tactile + flare — same model + training loop as
train_qwen3vl_flare.py, but the dataloader reads the EgoDex-style pretrain
manifest format produced by
  data/midtrain/scripts/gen_pretrain_mecka_parallel.py

Per episode the dataset expects:
  <ep>/pretrain.hdf5   states[N,62], action_chunks[N,16,62],
                       tracking_error[N-1,56], tactile_f6[N,10,6],
                       attrs(language, num_frames, fps, ...)
  <ep>/raw.h5          symlink to original episode_XXXX.h5 (read at training
                       time for tactile_deform[N,5,240,240] — too big to copy)
  <ep>/ego_view.mp4    head/slow camera (cropped or symlinked by preprocess)
  <ep>/left_wrist.mp4  fast camera 1
  <ep>/right_wrist.mp4 fast camera 2

The collate output dict is API-compatible with SftDataset.collate_fn so the
rest of the training loop, the residual delta_v from the tactile expert, and
the FLARE prediction loss are unchanged.

CLI:
  --data_root  ROOT   directory containing batch_*/pretrain_manifest.json
                      (replaces --data_path JSON from train_qwen3vl_flare.py)
"""

import os, sys

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import glob
import json
import torch
import logging
import argparse
import shutil
import math
import re
import wandb
import PIL.Image
import numpy as np
import h5py

from typing import List, Dict, Optional
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler as _DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator, DataLoaderConfiguration
from transformers import AutoProcessor, set_seed

from qwen_vla import Qwen3VLVLAModel, extend_position_ids_for_flare, split_slow_fast_embeds
import cv2

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def _rot6d_to_mat(rot6d):
    """6D rotation (first two columns) → 3×3 rotation matrix."""
    col1 = rot6d[:3]
    col2 = rot6d[3:6]
    col3 = np.cross(col1, col2)
    return np.column_stack([col1, col2, col3])


def _arm9d_to_axis_angle(arm_9d):
    """[trans(3), rot6d(6)] → [trans(3), axis_angle(3)]."""
    R = _rot6d_to_mat(arm_9d[3:9])
    aa, _ = cv2.Rodrigues(R)
    return np.concatenate([arm_9d[:3], aa.flatten()])


def _axis_angle_to_arm9d(arm_aa):
    """[trans(3), axis_angle(3)] → [trans(3), rot6d(6)]."""
    R, _ = cv2.Rodrigues(arm_aa[3:6])
    return np.concatenate([arm_aa[:3], R[:, 0], R[:, 1]])


def add_tracking_error_noise(state, te_mean, te_std, action_dim):
    """Add tracking-error noise to robot state.

    Supports single-arm (action_dim=31, te=28D) and bimanual (action_dim=62, te=56D).
    Per arm (28D tracking error = 3 xyz + 3 axis-angle + 22 hand):
      1. Convert arm 9D [trans, rot6d] → [trans, axis_angle]
      2. Sample noise ~ N(te_mean, te_std)
      3. Add noise
      4. Convert back to [trans, rot6d]
    """
    noisy = state.copy()
    n_arms = action_dim // 31
    for arm_idx in range(n_arms):
        offset = arm_idx * 31
        arm_9d = state[offset:offset + 9]
        hand_22d = state[offset + 9:offset + 31]

        arm_aa = _arm9d_to_axis_angle(arm_9d)  # (6,)

        te_off = arm_idx * 28
        noise_arm = np.random.normal(te_mean[te_off:te_off + 6],
                                     te_std[te_off:te_off + 6]).astype(np.float32)
        noise_hand = np.random.normal(te_mean[te_off + 6:te_off + 28],
                                      te_std[te_off + 6:te_off + 28]).astype(np.float32)

        noisy_arm_9d = _axis_angle_to_arm9d(arm_aa + noise_arm)
        noisy[offset:offset + 9] = noisy_arm_9d
        noisy[offset + 9:offset + 31] = hand_22d + noise_hand
    return noisy


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                     min_lr_ratio=0.0, num_cycles=0.5):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
        return (1 - min_lr_ratio) * cosine + min_lr_ratio
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


# ────────────────────────────────────────────────────────────────────────────
# Mid-training pretrain-format dataset (replaces SftDataset).
#
# Reads <data_root>/*/pretrain_manifest.json — per episode, opens
#   pretrain.hdf5  (states, action_chunks, tactile_f6, tracking_error, lang)
#   raw.h5         (tactile_deform — symlink to original episode HDF5)
#   ego_view.mp4 + left_wrist.mp4 + right_wrist.mp4 (decoded once per episode)
# and serves frames from a per-worker cache. EpisodeGroupedSampler keeps the
# cache warm by emitting each episode's frames contiguously to a given rank.
# ────────────────────────────────────────────────────────────────────────────


class EpisodeGroupedSampler(_DistributedSampler):
    """Distributed sampler that emits each episode's frames contiguously so
    that the per-worker episode cache stays warm. Episode *order* is shuffled
    each epoch; frame order *within* an episode is sequential. Subclasses
    DistributedSampler so accelerate/DeepSpeed don't replace it.
    """

    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, seed=0, drop_last=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=shuffle, seed=seed, drop_last=drop_last)
        self._cum_frames = dataset._cum_frames.copy()
        self._num_episodes = dataset._num_episodes
        self._frame_counts = np.diff(self._cum_frames, prepend=0)
        self._orig_starts = np.zeros(self._num_episodes, dtype=np.int64)
        if self._num_episodes > 1:
            self._orig_starts[1:] = self._cum_frames[:-1]

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            ep_perm = torch.randperm(self._num_episodes, generator=g).numpy()
        else:
            ep_perm = np.arange(self._num_episodes)

        shuffled_fc = self._frame_counts[ep_perm]
        shuffled_cum = np.cumsum(shuffled_fc)
        total_frames = int(shuffled_cum[-1])

        chunk = self.num_samples
        rank_start = self.rank * chunk
        rank_end = rank_start + chunk
        actual_end = min(rank_end, total_frames)

        ep_first = int(np.searchsorted(shuffled_cum, rank_start, side='right'))
        ep_last = int(np.searchsorted(shuffled_cum, actual_end, side='right'))
        ep_last = min(ep_last, self._num_episodes - 1)

        parts = []
        for i in range(ep_first, ep_last + 1):
            ep_idx = int(ep_perm[i])
            shuf_start = int(shuffled_cum[i - 1]) if i > 0 else 0
            t_start = max(0, rank_start - shuf_start)
            t_end = min(int(self._frame_counts[ep_idx]), actual_end - shuf_start)
            if t_start < t_end:
                base = int(self._orig_starts[ep_idx])
                parts.append(np.arange(base + t_start, base + t_end, dtype=np.int64))

        indices = np.concatenate(parts) if parts else np.array([], dtype=np.int64)

        if len(indices) < chunk:
            needed = chunk - len(indices)
            wrap_parts = []
            remaining = needed
            for i in range(self._num_episodes):
                if remaining <= 0:
                    break
                ep_idx = int(ep_perm[i])
                base = int(self._orig_starts[ep_idx])
                n = min(int(self._frame_counts[ep_idx]), remaining)
                wrap_parts.append(np.arange(base, base + n, dtype=np.int64))
                remaining -= n
            if wrap_parts:
                indices = np.concatenate([indices] + wrap_parts)

        return iter(indices[:chunk].tolist())


class MidtrainTacFlareDataset(Dataset):
    """Pretrain-format mid-training dataset with tactile + bimanual wrist views.

    Drop-in replacement for SftDataset: same constructor signature,
    `create_val_split`, `collate_fn`, and `stats_data` attribute.
    """

    def __init__(self, config, processor, accelerator):
        self.config = config
        self.processor = processor
        self.accelerator = accelerator

        manifest_paths = sorted(
            glob.glob(os.path.join(config.data_root, "*", "pretrain_manifest.json"))
        )
        if not manifest_paths:
            raise FileNotFoundError(
                f"No pretrain_manifest.json found under {config.data_root}/*/")

        _episode_dirs, _frame_counts = [], []
        _ep_crop_boxes = []   # parallel to _episode_dirs; (y0,y1,x0,x1) | None
        all_action_q01, all_action_q99 = [], []
        all_state_q01,  all_state_q99  = [], []
        all_tacf6_q01,  all_tacf6_q99  = [], []
        all_te_mean,    all_te_std     = [], []

        for mp in manifest_paths:
            with open(mp, "r") as f:
                manifest = json.load(f)
            # Per-batch head_crop_box. Manifests written before the
            # dataloader-crop change (e.g. the already-processed NV data)
            # don't have this field — treat as no-crop.
            cb = manifest.get("head_crop_box", None)
            crop_tuple = tuple(int(x) for x in cb) if cb else None
            for ep in manifest["episodes"]:
                _episode_dirs.append(ep["episode_dir"])
                _frame_counts.append(ep["num_frames"])
                _ep_crop_boxes.append(crop_tuple)
            stats = manifest.get("statistics", {})
            if "action" in stats:
                all_action_q01.append(np.array(stats["action"]["q01"], dtype=np.float32))
                all_action_q99.append(np.array(stats["action"]["q99"], dtype=np.float32))
            if "state" in stats:
                all_state_q01.append(np.array(stats["state"]["q01"], dtype=np.float32))
                all_state_q99.append(np.array(stats["state"]["q99"], dtype=np.float32))
            if "tactile_f6" in stats and stats["tactile_f6"]:
                all_tacf6_q01.append(np.array(stats["tactile_f6"]["q01"], dtype=np.float32))
                all_tacf6_q99.append(np.array(stats["tactile_f6"]["q99"], dtype=np.float32))
            te_block = stats.get("tracking_error", {})
            if te_block:
                all_te_mean.append(np.array(te_block["mean"], dtype=np.float32))
                all_te_std.append(np.array(te_block["std"], dtype=np.float32))

        accelerator.print(f"[Midtrain] Loaded {len(manifest_paths)} batch manifests")

        # Pool stats across batches: min(q01) / max(q99), avg(te)
        self.action_min = np.min(np.stack(all_action_q01), axis=0)
        self.action_max = np.max(np.stack(all_action_q99), axis=0)
        self.state_min  = np.min(np.stack(all_state_q01),  axis=0)
        self.state_max  = np.max(np.stack(all_state_q99),  axis=0)
        if all_tacf6_q01:
            self.tacf6_min = np.min(np.stack(all_tacf6_q01), axis=0)
            self.tacf6_max = np.max(np.stack(all_tacf6_q99), axis=0)
        else:
            self.tacf6_min = np.full(60, -1.0, dtype=np.float32)
            self.tacf6_max = np.full(60, +1.0, dtype=np.float32)

        te_dim = (config.action_dim // 31) * 28
        if all_te_mean:
            self.te_mean = np.mean(np.stack(all_te_mean), axis=0)
            self.te_std  = np.mean(np.stack(all_te_std),  axis=0)
        else:
            self.te_mean = np.zeros(te_dim, dtype=np.float32)
            self.te_std  = np.zeros(te_dim, dtype=np.float32)

        self.action_mask = np.ones(config.action_dim, dtype=bool)
        self.state_mask  = np.ones(config.action_dim, dtype=bool)
        self.tacf6_mask  = np.ones(60, dtype=bool)

        # `stats_data` is consumed by save_checkpoint to produce stats_data.json.
        # Keyed under "midtrain" so eval scripts can `next(iter(...))` like before.
        self.stats_data = {
            "midtrain": {
                "action": {
                    "mask": self.action_mask.tolist(),
                    "q01": self.action_min.tolist(),
                    "q99": self.action_max.tolist(),
                },
                "state": {
                    "mask": self.state_mask.tolist(),
                    "q01": self.state_min.tolist(),
                    "q99": self.state_max.tolist(),
                },
                "tactile_f6": {
                    "mask": self.tacf6_mask.tolist(),
                    "q01": self.tacf6_min.tolist(),
                    "q99": self.tacf6_max.tolist(),
                },
                "tracking_error": {
                    "mean": self.te_mean.tolist(),
                    "std":  self.te_std.tolist(),
                },
            }
        }

        self.image_size = tuple(config.image_size) if config.image_size else None
        self.use_flare = bool(getattr(config, "use_flare", 0))
        self.n_flare_steps = getattr(config, "n_flare_steps", 0) if self.use_flare else 0
        self.flare_frame_stride = getattr(config, "flare_frame_stride", 4)

        self._episode_dirs = _episode_dirs
        self._ep_crop_boxes = _ep_crop_boxes
        _frame_counts = np.array(_frame_counts, dtype=np.int64)
        self._cum_frames = np.cumsum(_frame_counts)
        self._total_transitions = int(self._cum_frames[-1])
        self._num_episodes = len(_episode_dirs)

        accelerator.print(
            f"[Midtrain] {self._num_episodes} episodes, "
            f"{self._total_transitions} transitions, "
            f"flare={self.n_flare_steps}×{self.flare_frame_stride}")

    def __len__(self):
        return self._total_transitions

    def create_val_split(self, val_ratio=0.05, seed=42):
        import copy
        rng = np.random.RandomState(seed)
        n_val = max(1, int(self._num_episodes * val_ratio))
        perm = rng.permutation(self._num_episodes)
        val_eps = sorted(perm[:n_val].tolist())
        train_eps = sorted(perm[n_val:].tolist())

        frame_counts = np.diff(self._cum_frames, prepend=0)

        val_ds = copy.copy(self)
        val_ds._episode_dirs = [self._episode_dirs[i] for i in val_eps]
        val_ds._ep_crop_boxes = [self._ep_crop_boxes[i] for i in val_eps]
        val_fc = frame_counts[val_eps]
        val_ds._cum_frames = np.cumsum(val_fc)
        val_ds._total_transitions = int(val_ds._cum_frames[-1])
        val_ds._num_episodes = len(val_eps)

        self._episode_dirs = [self._episode_dirs[i] for i in train_eps]
        self._ep_crop_boxes = [self._ep_crop_boxes[i] for i in train_eps]
        train_fc = frame_counts[train_eps]
        self._cum_frames = np.cumsum(train_fc)
        self._total_transitions = int(self._cum_frames[-1])
        self._num_episodes = len(train_eps)

        self.accelerator.print(
            f"[Midtrain] Train/Val split: {self._num_episodes} train eps "
            f"({self._total_transitions} frames), "
            f"{val_ds._num_episodes} val eps "
            f"({val_ds._total_transitions} frames)")
        return val_ds

    @staticmethod
    def _normalize(values, mask, vmin, vmax):
        return np.where(
            mask,
            np.clip(2 * (values - vmin) / (vmax - vmin + 1e-8) - 1, -1, 1),
            values,
        )

    @staticmethod
    def _decode_video(path: str, crop_box=None) -> list:
        """Decode all frames into a list of HWC uint8 RGB arrays. When
        `crop_box=(y0,y1,x0,x1)` is given, slice each frame *during* decode
        so the per-worker cache only holds cropped frames (this is how BKL
        head videos get cropped at training time without the preprocess
        re-encoding the mp4 to disk).
        """
        cap = cv2.VideoCapture(path)
        frames = []
        if crop_box is not None:
            y0, y1, x0, x1 = crop_box
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            if crop_box is not None:
                fr = fr[y0:y1, x0:x1]
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def _load_episode_cache(self, ep_idx: int):
        """Bulk-read everything we need for one episode. Called once per
        episode per worker; previous episode's cache is freed when ep_idx
        changes (Python GC reclaims the large numpy arrays/frame lists).
        """
        ep_dir = self._episode_dirs[ep_idx]
        self._cache_ep_idx = ep_idx
        self._cache_h5 = None
        self._cache_tac = None
        self._cache_videos = {"head": None, "wrist_l": None, "wrist_r": None}

        ph5 = os.path.join(ep_dir, "pretrain.hdf5")
        if os.path.isfile(ph5):
            try:
                with h5py.File(ph5, "r") as f:
                    self._cache_h5 = {
                        "states": f["states"][:],
                        "action_chunks": f["action_chunks"][:],
                        "tactile_f6": f["tactile_f6"][:] if "tactile_f6" in f else None,
                        "language": str(f.attrs.get("language", "")),
                    }
            except Exception:
                pass

        # tactile_deform from raw.h5 — only load when the model actually uses it.
        if self.config.use_tactile_deform:
            rh5 = os.path.join(ep_dir, "raw.h5")
            if os.path.isfile(rh5):
                try:
                    with h5py.File(rh5, "r") as f:
                        if ("left_hand_tactile_deform" in f
                                and "right_hand_tactile_deform" in f):
                            l_def = f["left_hand_tactile_deform"][:]   # [N,5,H,W] uint8
                            r_def = f["right_hand_tactile_deform"][:]
                            self._cache_tac = np.concatenate([l_def, r_def], axis=1)
                except Exception:
                    pass

        head_crop = self._ep_crop_boxes[ep_idx]   # None or (y0,y1,x0,x1)
        for vname, fname, cb in [("head", "ego_view.mp4", head_crop),
                                 ("wrist_l", "left_wrist.mp4", None),
                                 ("wrist_r", "right_wrist.mp4", None)]:
            vp = os.path.join(ep_dir, fname)
            if os.path.isfile(vp):
                try:
                    self._cache_videos[vname] = self._decode_video(vp, crop_box=cb)
                except Exception:
                    pass

    def __getitem__(self, idx: int) -> Dict:
        ep_idx = int(np.searchsorted(self._cum_frames, idx, side="right"))
        frame_t = int(idx - (int(self._cum_frames[ep_idx - 1]) if ep_idx > 0 else 0))

        fb_frame = np.zeros((288, 384, 3), dtype=np.uint8)
        fb_def = np.zeros((10, 240, 240), dtype=np.uint8)
        fb_f6 = np.zeros((10, 6), dtype=np.float32)
        fallback = {
            "head": fb_frame, "wrist_l": fb_frame.copy(), "wrist_r": fb_frame.copy(),
            "state": np.zeros(self.config.action_dim, dtype=np.float32),
            "action_chunk": np.zeros(
                (self.config.action_chunk, self.config.action_dim), dtype=np.float32),
            "tactile_f6": fb_f6, "tactile_deform": fb_def,
            "language": "",
            "flare_frames": [fb_frame.copy() for _ in range(self.n_flare_steps)],
        }

        if not hasattr(self, "_cache_ep_idx") or self._cache_ep_idx != ep_idx:
            self._load_episode_cache(ep_idx)

        if self._cache_h5 is None:
            return fallback
        try:
            state = self._cache_h5["states"][frame_t]
            action_chunk = self._cache_h5["action_chunks"][frame_t]
            language = self._cache_h5["language"]
        except (IndexError, KeyError):
            return fallback

        tactile_f6 = (self._cache_h5["tactile_f6"][frame_t]
                      if self._cache_h5.get("tactile_f6") is not None
                      else fb_f6)
        tactile_deform = (self._cache_tac[frame_t]
                          if (self._cache_tac is not None
                              and frame_t < len(self._cache_tac))
                          else fb_def)

        # Paradigm C: delayed tactile observations at frame_t + k where
        # k ~ Uniform({0, 4, 8, 12}).  Used in training as the "fresh tactile"
        # input to the residual flow refinement step (simulates the inference
        # pattern: slow chunk anchored at t=0, tactile fired at t+k).
        delay_offsets = getattr(self.config, "tactile_delay_offsets",
                                (0, 4, 8, 12))
        rng = np.random.default_rng()
        delay_k = int(rng.choice(delay_offsets))
        # Clamp to episode end
        ep_len_tac_f6 = (len(self._cache_h5["tactile_f6"])
                        if self._cache_h5.get("tactile_f6") is not None else 0)
        ep_len_tac_def = (len(self._cache_tac)
                          if self._cache_tac is not None else 0)
        delayed_t_f6  = min(frame_t + delay_k, max(ep_len_tac_f6 - 1, 0))
        delayed_t_def = min(frame_t + delay_k, max(ep_len_tac_def - 1, 0))
        tactile_f6_delayed = (self._cache_h5["tactile_f6"][delayed_t_f6]
                              if ep_len_tac_f6 > 0 else fb_f6)
        tactile_deform_delayed = (self._cache_tac[delayed_t_def]
                                  if ep_len_tac_def > 0 else fb_def)

        head_frames    = self._cache_videos.get("head")    or []
        wrist_l_frames = self._cache_videos.get("wrist_l") or []
        wrist_r_frames = self._cache_videos.get("wrist_r") or []

        def _pick(seq, t):
            if seq and t < len(seq):
                return seq[t]
            return seq[-1] if seq else fb_frame

        head_frame    = _pick(head_frames,    frame_t)
        wrist_l_frame = _pick(wrist_l_frames, frame_t)
        wrist_r_frame = _pick(wrist_r_frames, frame_t)

        # FLARE futures from the head camera (matches the JSON pipeline:
        # flare images come from `input_image_slow`).
        flare_frames = []
        n_head = len(head_frames)
        for k in range(self.n_flare_steps):
            ft = frame_t + (k + 1) * self.flare_frame_stride
            if n_head > 0:
                flare_frames.append(head_frames[min(ft, n_head - 1)])
            else:
                flare_frames.append(head_frame.copy()
                                    if head_frame is not fb_frame
                                    else fb_frame.copy())

        return {
            "head": head_frame,
            "wrist_l": wrist_l_frame,
            "wrist_r": wrist_r_frame,
            "state": state,
            "action_chunk": action_chunk,
            "tactile_f6": tactile_f6,
            "tactile_deform": tactile_deform,
            "tactile_f6_delayed": tactile_f6_delayed,
            "tactile_deform_delayed": tactile_deform_delayed,
            "delay_k": delay_k,
            "language": language,
            "flare_frames": flare_frames,
        }

    def collate_fn(self, batch: List[Dict]) -> Dict:
        cfg = self.config
        B = len(batch)

        # ── action chunk (already 16×D from preprocessing) ──
        actions = np.stack([x["action_chunk"] for x in batch], axis=0).astype(np.float32)
        norm_actions = self._normalize(
            actions, self.action_mask, self.action_min, self.action_max)
        norm_actions = torch.tensor(norm_actions, dtype=torch.bfloat16)

        # ── flow-matching noise ──
        d = torch.distributions.Beta(
            torch.tensor(1.5, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
        )
        time = (d.sample((B,)) * 0.999 + 0.001).to(torch.bfloat16)
        t_ = time[:, None, None]
        noise = torch.randn_like(norm_actions)
        x_t = t_ * noise + (1 - t_) * norm_actions
        u_t = noise - norm_actions

        # ── tactile_f6 ──
        norm_tacf6 = None
        if cfg.use_tactile_vec:
            tacf6 = np.stack([x["tactile_f6"] for x in batch], axis=0).astype(np.float32)
            tacf6_flat = tacf6.reshape(B, -1)  # [B, 60]
            tacf6_flat = self._normalize(
                tacf6_flat, self.tacf6_mask, self.tacf6_min, self.tacf6_max)
            norm_tacf6 = torch.tensor(
                tacf6_flat.reshape(B, -1, 6), dtype=torch.bfloat16)

        # ── tactile_deform [B,10,H,W] uint8 → [B,10,1,H,W] float ──
        deforms_tensor = None
        if cfg.use_tactile_deform:
            deforms = np.stack(
                [x["tactile_deform"] for x in batch], axis=0)   # [B,10,H,W] uint8
            deforms_tensor = torch.tensor(
                deforms.astype(np.float32) / 255.0, dtype=torch.float32
            ).unsqueeze(2)   # [B,10,1,H,W]

        # ── delayed tactile (Paradigm C) ──
        norm_tacf6_delayed = None
        if cfg.use_tactile_vec:
            tacf6_d = np.stack([x["tactile_f6_delayed"] for x in batch],
                               axis=0).astype(np.float32)
            tacf6_d_flat = tacf6_d.reshape(B, -1)
            tacf6_d_flat = self._normalize(
                tacf6_d_flat, self.tacf6_mask, self.tacf6_min, self.tacf6_max)
            norm_tacf6_delayed = torch.tensor(
                tacf6_d_flat.reshape(B, -1, 6), dtype=torch.bfloat16)
        deforms_delayed_tensor = None
        if cfg.use_tactile_deform:
            deforms_d = np.stack(
                [x["tactile_deform_delayed"] for x in batch], axis=0)
            deforms_delayed_tensor = torch.tensor(
                deforms_d.astype(np.float32) / 255.0, dtype=torch.float32
            ).unsqueeze(2)
        delay_k_tensor = torch.tensor(
            [x["delay_k"] for x in batch], dtype=torch.long)

        # ── residual-flow noise sampling for L_refine (Paradigm C) ──
        # Same Beta(1.5, 1.0) prior as the action flow, scaled to a small
        # noise magnitude because the residual r = A_demo − Â is small.
        time_r = (d.sample((B,)) * 0.999 + 0.001).to(torch.bfloat16)
        eps_r = torch.randn_like(norm_actions)

        # ── state (with tracking-error noise when use_robot_state=1) ──
        state_raw_list = []
        if cfg.use_robot_state:
            for x in batch:
                s = np.array(x["state"], dtype=np.float32)
                s = add_tracking_error_noise(
                    s, self.te_mean, self.te_std, cfg.action_dim)
                ns = self._normalize(s, self.state_mask, self.state_min, self.state_max)
                state_raw_list.append(torch.tensor(ns, dtype=torch.bfloat16))
        state_raw = torch.stack(state_raw_list) if state_raw_list else None

        # ── chat template: [head (slow) | text | wrist_r, wrist_l (fast)] ──
        # Order matches gen_json_tac_deltabase_eef_bimanual_parallel.py:
        #   img_fast_list = [fast_r_img_paths[i], fast_l_img_paths[i]]
        all_input_ids, all_pixel_values, all_grid_thw = [], [], []
        n_slow_images = 1

        for x in batch:
            pil_head = PIL.Image.fromarray(x["head"])
            pil_wr = PIL.Image.fromarray(x["wrist_r"])
            pil_wl = PIL.Image.fromarray(x["wrist_l"])
            if self.image_size is not None:
                pil_head = pil_head.resize(self.image_size, PIL.Image.LANCZOS)
                pil_wr = pil_wr.resize(self.image_size, PIL.Image.LANCZOS)
                pil_wl = pil_wl.resize(self.image_size, PIL.Image.LANCZOS)

            content = [
                {"type": "image"},                                  # slow head
                {"type": "text", "text": x.get("language", "")},
                {"type": "image"},                                  # fast right wrist
                {"type": "image"},                                  # fast left wrist
            ]
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inp = self.processor(
                text=text, images=[pil_head, pil_wr, pil_wl],
                return_tensors="pt", padding=False,
            )
            all_input_ids.append(inp.input_ids[0])
            if "pixel_values" in inp and inp.pixel_values is not None:
                all_pixel_values.append(inp.pixel_values)
                all_grid_thw.append(inp.image_grid_thw)

        # ── FLARE future frames (head only) ──
        flare_pixel_values = None
        flare_grid_thw = None
        if self.n_flare_steps > 0:
            flare_pil_imgs = []
            for x in batch:
                for ff in x["flare_frames"]:
                    pil = PIL.Image.fromarray(ff)
                    if self.image_size is not None:
                        pil = pil.resize(self.image_size, PIL.Image.LANCZOS)
                    flare_pil_imgs.append(pil)
            flare_inp = self.processor.image_processor(
                flare_pil_imgs, return_tensors="pt")
            flare_pixel_values = flare_inp.pixel_values.to(torch.bfloat16)
            flare_grid_thw = flare_inp.image_grid_thw

        # ── padding + stacking (mirrors SftDataset.collate_fn) ──
        pad_id = self.processor.tokenizer.pad_token_id or 0
        max_len = max(ids.shape[0] for ids in all_input_ids)
        padded_ids, attention_ms = [], []
        for ids in all_input_ids:
            pad_len = max_len - ids.shape[0]
            padded_ids.append(F.pad(ids, (pad_len, 0), value=pad_id))
            attn = torch.ones(max_len, dtype=torch.long)
            if pad_len > 0:
                attn[:pad_len] = 0
            attention_ms.append(attn)

        input_ids = torch.stack(padded_ids)
        attention_mask = torch.stack(attention_ms)
        pixel_values = (torch.cat(all_pixel_values, dim=0)
                        if all_pixel_values else None)
        image_grid_thw = (torch.cat(all_grid_thw, dim=0)
                          if all_grid_thw else None)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "n_slow_images": n_slow_images,
            "noisy_actions": x_t,
            "target": u_t,
            "timesteps": time,
            "norm_actions": norm_actions,        # ground-truth A_demo (Paradigm C)
            "tactile_f6s": norm_tacf6,
            "tactile_deforms": deforms_tensor,
            "tactile_f6s_delayed": norm_tacf6_delayed,
            "tactile_deforms_delayed": deforms_delayed_tensor,
            "delay_k": delay_k_tensor,
            "time_r": time_r,                    # τ_r ∈ (0,1] for residual flow
            "eps_r": eps_r,                      # ε_r noise for residual flow
            "state_raw": state_raw,
            "flare_pixel_values": flare_pixel_values,
            "flare_grid_thw": flare_grid_thw,
        }


def save_checkpoint(model, processor, accelerator, args, epoch, global_step, stats_data):
    save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")

    if accelerator.is_main_process:
        ckpts = [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint-")]
        if args.max_ckpts > 0 and len(ckpts) >= args.max_ckpts:
            oldest = min(ckpts, key=lambda f: os.path.getctime(os.path.join(args.output_dir, f)))
            shutil.rmtree(os.path.join(args.output_dir, oldest))

        os.makedirs(save_dir, exist_ok=True)

        sd = accelerator.get_state_dict(model)
        torch.save(sd, os.path.join(save_dir, "model.pt"))

        processor.save_pretrained(os.path.join(save_dir, "processor"))

        src_config = os.path.join(args.model_path, "config.json")
        if os.path.exists(src_config):
            shutil.copy(src_config, os.path.join(save_dir, "config.json"))

        with open(os.path.join(save_dir, "training_args.json"), "w") as f:
            json.dump({
                "model_path": args.model_path,
                "action_dim": args.action_dim,
                "action_chunk": args.action_chunk,
                "use_robot_state": args.use_robot_state,
                "use_tactile_deform": args.use_tactile_deform,
                "use_tactile_vec": getattr(args, "use_tactile_vec", 0),
                "tactile_intermediate_size": getattr(args, "tactile_intermediate_size", 0),
                "training_stage": args.training_stage,
                "use_flare": args.use_flare,
                "n_flare_tokens_per_frame": args.n_flare_tokens_per_frame,
                "n_flare_steps": args.n_flare_steps,
                "flare_layer_index": args.flare_layer_index,
            }, f, indent=2)

        with open(os.path.join(save_dir, "stats_data.json"), "w") as f:
            json.dump(stats_data, f, indent=2)

    accelerator.wait_for_everyone()
    logger.info(f"Checkpoint {epoch}-{global_step} saved.")


class TrainingMetrics:
    def __init__(self, device):
        self.n_step       = 0
        self.action_loss  = torch.tensor(0.0, device=device)
        self.tactile_loss = torch.tensor(0.0, device=device)
        self.flare_loss  = torch.tensor(0.0, device=device)
        self.total_loss   = torch.tensor(0.0, device=device)
        self.world_size   = dist.get_world_size()

    def update(self, total, action, tactile=0.0, flare=0.0):
        self.n_step += 1
        self.total_loss   += total.item() if torch.is_tensor(total) else total
        self.action_loss  += action.item() if torch.is_tensor(action) else action
        self.tactile_loss += tactile.item() if torch.is_tensor(tactile) else tactile
        self.flare_loss  += flare.item() if torch.is_tensor(flare) else flare

    def get_metric(self, reset=True):
        for t in [self.total_loss, self.action_loss, self.tactile_loss, self.flare_loss]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        denom = self.world_size * max(self.n_step, 1)
        m = {
            "total_loss":   self.total_loss.item() / denom,
            "action_loss":  self.action_loss.item() / denom,
            "tactile_loss": self.tactile_loss.item() / denom,
            "flare_loss":  self.flare_loss.item() / denom,
        }
        if reset:
            self.n_step = 0
            for t in [self.total_loss, self.action_loss, self.tactile_loss, self.flare_loss]:
                t.fill_(0)
        return m


@torch.no_grad()
def run_validation(model, val_dataloader, accelerator, args,
                   is_stage1, use_flare, K, T_per_frame, flare_layer_idx):
    """Run validation and return averaged metrics (action + tactile + flare)."""
    model.eval()
    device = torch.cuda.current_device()
    val_act = torch.tensor(0.0, device=device)
    val_tac = torch.tensor(0.0, device=device)
    val_flare = torch.tensor(0.0, device=device)
    n_val = 0
    max_batches = getattr(args, "max_val_batches", 50)

    for i, batch in enumerate(val_dataloader):
        if i >= max_batches:
            break
        raw_model = accelerator.unwrap_model(model)

        inputs_embeds = raw_model.prepare_inputs_embeds(
            input_ids=batch["input_ids"],
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"))

        n_slow_imgs = batch["n_slow_images"]
        grid_thw = batch.get("image_grid_thw")
        B = inputs_embeds.shape[0]
        if grid_thw is not None and grid_thw.shape[0] > n_slow_imgs:
            merge = getattr(raw_model.visual, "spatial_merge_size", 2)
            n_imgs_per_sample = grid_thw.shape[0] // B
            n_slow_img_tokens = sum(
                int(g[0] * (g[1] // merge) * (g[2] // merge))
                for g in grid_thw[:n_imgs_per_sample][:n_slow_imgs])
            slow_embeds, fast_embeds = split_slow_fast_embeds(
                inputs_embeds, batch["input_ids"],
                raw_model.image_token_id, n_slow_img_tokens)
        else:
            slow_embeds = inputs_embeds
            fast_embeds = inputs_embeds[:, :0]

        L_slow = slow_embeds.shape[1]
        pos_ids, _ = raw_model.get_rope_index(
            input_ids=batch["input_ids"],
            image_grid_thw=batch.get("image_grid_thw"),
            attention_mask=batch["attention_mask"])
        pos_ids = pos_ids[:, :, :L_slow]

        if use_flare:
            flare_q = raw_model.flare_queries.expand(B, -1, -1).to(
                device=slow_embeds.device, dtype=slow_embeds.dtype)
            slow_embeds_ext = torch.cat([slow_embeds, flare_q], dim=1)
            pos_ids = extend_position_ids_for_flare(pos_ids, K)
            L_latent = slow_embeds_ext.shape[1]
        else:
            slow_embeds_ext = slow_embeds
            L_latent = L_slow

        if args.use_robot_state and batch["state_raw"] is not None:
            state_vec = batch["state_raw"].to(slow_embeds.device, dtype=slow_embeds.dtype)
            state_embeds = raw_model.state_embedder(state_vec).unsqueeze(1)
        else:
            state_embeds = torch.empty((B, 0, slow_embeds.shape[2]),
                                       device=slow_embeds.device, dtype=slow_embeds.dtype)
        n_state = state_embeds.shape[1]

        noisy_actions = raw_model.x_embedder(batch["noisy_actions"].to(slow_embeds.dtype))
        timesteps = raw_model.t_embedder(batch["timesteps"].to(slow_embeds.dtype)).unsqueeze(1)
        chunk = args.action_chunk
        target = batch["target"].to(slow_embeds.dtype)
        n_fast = fast_embeds.shape[1]
        has_any_tac = bool(args.use_tactile_vec or args.use_tactile_deform)

        if args.use_tactile_refine_flow and has_any_tac:
            # Paradigm C validation: action expert single-step + tactile residual flow loss
            full_embeds = torch.cat([
                slow_embeds_ext, fast_embeds, state_embeds, timesteps, noisy_actions,
            ], dim=1)
            L_total = full_embeds.shape[1]
            outputs = model.model(
                inputs_embeds=full_embeds, position_ids=pos_ids,
                attention_mask=batch["attention_mask"], use_cache=False,
                output_hidden_states=use_flare,
                latent_indexes=torch.arange(0, L_latent, device=full_embeds.device),
                action_indexes=torch.arange(L_latent, L_total, device=full_embeds.device),
                tactile_indexes=torch.arange(0, 0, device=full_embeds.device))
            hidden = outputs.last_hidden_state
            act_start = L_latent + n_fast + n_state + 1
            v_act = raw_model.final_layer(hidden[:, act_start:act_start + chunk, :])
            loss_act = nn.MSELoss()(v_act, target)

            fe = fast_embeds if n_fast > 0 else None
            se = state_embeds if n_state > 0 else None
            ahat_noise = torch.randn_like(batch["noisy_actions"])
            a_hat, cached_kv, n_action_in_cache = (
                raw_model.forward_flow_action_only(
                    inputs_embeds=slow_embeds_ext,
                    position_ids=pos_ids,
                    noise=ahat_noise,
                    attention_mask=batch["attention_mask"],
                    state_embeds=se,
                    fast_embeds=fe,
                    num_steps=args.action_flow_train_steps,
                    refresh_clean_kv=True,
                ))
            norm_actions_gt = batch["norm_actions"].to(slow_embeds.dtype)
            a_hat_for_residual = a_hat
            if args.tactile_residual_jitter > 0:
                a_hat_for_residual = a_hat_for_residual + (
                    args.tactile_residual_jitter
                    * torch.randn_like(a_hat_for_residual))
            r_target = norm_actions_gt - a_hat_for_residual
            eps_r    = batch["eps_r"].to(slow_embeds.dtype)
            tau_r    = batch["time_r"].to(slow_embeds.dtype)
            tau_r_b  = tau_r[:, None, None]
            r_tau    = (1 - tau_r_b) * r_target + tau_r_b * eps_r
            v_target_r = eps_r - r_target
            v_pred_r = raw_model.tactile_residual_train_step(
                cached_kv=cached_kv,
                latent_position_ids=pos_ids,
                n_action_in_cache=n_action_in_cache,
                base_chunk=a_hat_for_residual,
                tactile_f6=batch.get("tactile_f6s_delayed"),
                tactile_deform=batch.get("tactile_deforms_delayed"),
                r_tau=r_tau,
                tau=tau_r,
            )
            loss_tac = nn.MSELoss()(v_pred_r, v_target_r)
        elif is_stage1 or not has_any_tac:
            full_embeds = torch.cat([
                slow_embeds_ext, fast_embeds, state_embeds, timesteps, noisy_actions,
            ], dim=1)
            L_total = full_embeds.shape[1]
            outputs = model.model(
                inputs_embeds=full_embeds, position_ids=pos_ids,
                attention_mask=batch["attention_mask"], use_cache=False,
                output_hidden_states=use_flare,
                latent_indexes=torch.arange(0, L_latent, device=full_embeds.device),
                action_indexes=torch.arange(L_latent, L_total, device=full_embeds.device),
                tactile_indexes=torch.arange(0, 0, device=full_embeds.device))
            hidden = outputs.last_hidden_state
            act_start = L_latent + n_fast + n_state + 1
            v_act = raw_model.final_layer(hidden[:, act_start:act_start + chunk, :])
            loss_act = nn.MSELoss()(v_act, target)
            loss_tac = 0.0
        else:
            tac_parts = []
            if args.use_tactile_vec and batch["tactile_f6s"] is not None:
                tac_parts.append(raw_model.tacf6_embedder(batch["tactile_f6s"].to(slow_embeds.dtype)))
            if args.use_tactile_deform and batch["tactile_deforms"] is not None:
                deforms = batch["tactile_deforms"].to(slow_embeds.device, dtype=slow_embeds.dtype)
                Bs, nf, C, H, W = deforms.shape
                feats = raw_model.deform_encoder(deforms.view(-1, C, H, W))
                feats = feats.view(Bs, nf, -1)
                tac_parts.append(raw_model.deform_proj(feats.to(slow_embeds.dtype)))
            tactile_embeds = torch.cat(tac_parts, dim=1) if tac_parts else \
                torch.empty((B, 0, slow_embeds.shape[2]), device=slow_embeds.device, dtype=slow_embeds.dtype)
            has_tac = tactile_embeds.shape[1] > 0
            n_action = n_fast + n_state + 1 + chunk
            if has_tac:
                noisy_actions_tac = raw_model.x_embedder(batch["noisy_actions"].to(slow_embeds.dtype))
                timesteps_tac = raw_model.t_embedder(batch["timesteps"].to(slow_embeds.dtype)).unsqueeze(1)
                full_embeds = torch.cat([
                    slow_embeds_ext, fast_embeds, state_embeds, timesteps, noisy_actions,
                    tactile_embeds, timesteps_tac, noisy_actions_tac,
                ], dim=1)
            else:
                full_embeds = torch.cat([
                    slow_embeds_ext, fast_embeds, state_embeds, timesteps, noisy_actions,
                ], dim=1)
            L_total = full_embeds.shape[1]
            outputs = model.model(
                inputs_embeds=full_embeds, position_ids=pos_ids,
                attention_mask=batch["attention_mask"], use_cache=False,
                output_hidden_states=use_flare,
                latent_indexes=torch.arange(0, L_latent, device=full_embeds.device),
                action_indexes=torch.arange(L_latent, L_latent + n_action, device=full_embeds.device),
                tactile_indexes=torch.arange(L_latent + n_action, L_total, device=full_embeds.device) if has_tac
                    else torch.arange(0, 0, device=full_embeds.device))
            hidden = outputs.last_hidden_state
            act_start = L_latent + n_fast + n_state + 1
            v_act = raw_model.final_layer(hidden[:, act_start:act_start + chunk, :])
            loss_act = nn.MSELoss()(v_act, target)
            if has_tac:
                delta_v = raw_model.final_layer_tactile(hidden[:, -chunk:, :])
                loss_tac = nn.MSELoss()(delta_v, target - v_act.detach())
            else:
                loss_tac = 0.0

        val_act += loss_act.item()
        val_tac += (loss_tac.item() if torch.is_tensor(loss_tac) else loss_tac)
        n_val += 1

    for t in [val_act, val_tac, val_flare]:
        if dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    ws = dist.get_world_size() if dist.is_initialized() else 1
    denom = ws * max(n_val, 1)

    model.train()
    return {
        "val/action_loss": val_act.item() / denom,
        "val/tactile_loss": val_tac.item() / denom,
    }


def train(args):
    # dispatch_batches=False — required because we hand-shard via
    # EpisodeGroupedSampler. With the default (None), accelerate may wrap our
    # batch_sampler in BatchSamplerShard and shard a SECOND time → each rank
    # sees 1/world_size² of the data (the "5925 → 93 steps" bug at ws=64).
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(
            dispatch_batches=False, even_batches=True),
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        wandb.init(project=args.experiment_name, 
                   name=args.run_name,
                   config=args, 
                   dir=args.log_dir
                )

    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config["train_batch_size"] = (args.train_bsz_per_gpu * dist.get_world_size() * accelerator.gradient_accumulation_steps)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    tac_isize = args.tactile_intermediate_size if args.tactile_intermediate_size > 0 else None
    model = Qwen3VLVLAModel.from_pretrained_qwen3vl(
        pretrained_path=args.model_path,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
        use_tactile_deform=bool(args.use_tactile_deform),
        use_robot_state=bool(args.use_robot_state),
        torch_dtype=torch.bfloat16,
        tactile_intermediate_size=tac_isize,
        n_flare_tokens_per_frame=args.n_flare_tokens_per_frame if args.use_flare else 0,
        n_flare_steps=args.n_flare_steps if args.use_flare else 0,
        flare_layer_index=args.flare_layer_index,
    )
    if args.use_tactile_deform:
        model.load_deform_encoder_weights(args.deform_encoder_ckpt)
    model.initialize_vla_weights(skip_tactile_zero_init=bool(args.use_tactile_refine_flow))
    if args.use_flare:
        accelerator.print(
            f"Flare alignment: {args.n_flare_steps} steps × {args.n_flare_tokens_per_frame} tok/frame "
            f"= {model.n_flare_tokens} total tokens, layer_index={args.flare_layer_index}"
        )

    is_stage1 = (args.training_stage == 1)
    has_any_tactile = bool(args.use_tactile_vec or args.use_tactile_deform)
    # Paradigm C always trains tactile when tactile data is available;
    # Paradigm A still freezes tactile during stage-1 pretrain.
    if args.use_tactile_refine_flow:
        freeze_tactile = not has_any_tactile
    else:
        freeze_tactile = is_stage1 or (not has_any_tactile)

    if not args.resume_checkpoint:
        named_params = dict(model.named_parameters())
        for name, param in model.named_parameters():
            if "_action" in name:
                base = name.replace("_action", "")
                if base in named_params:
                    param.data.copy_(named_params[base].data)
        accelerator.print("Action expert initialized from latent expert.")

    if args.resume_checkpoint:
        ckpt_path = args.resume_checkpoint
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "model.pt")
        resume_sd = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in resume_sd:
            resume_sd = resume_sd["state_dict"]
        # Filter out keys with shape mismatch (e.g. tactile MLP from pretrain)
        model_sd = model.state_dict()
        filtered_sd = {}
        skipped = []
        for k, v in resume_sd.items():
            if k in model_sd and model_sd[k].shape != v.shape:
                skipped.append(k)
            else:
                filtered_sd[k] = v
        if skipped:
            accelerator.print(f"Skipped {len(skipped)} keys with shape mismatch (e.g. {skipped[0]})")
        missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
        accelerator.print(f"Resumed: missing={len(missing)}, unexpected={len(unexpected)}")

    resumed_tactile = bool(args.resume_checkpoint) and args.resume_source == "midtrain"
    if not freeze_tactile and not resumed_tactile:
        for name, param in model.named_parameters():
            if "_tactile" in name:
                if param.ndim >= 2:
                    nn.init.xavier_uniform_(param)
                elif param.ndim == 1:
                    nn.init.zeros_(param)
        # Paradigm A: zero-init final_layer_tactile so delta_v starts at 0.
        # Paradigm C: keep xavier-init from initialize_vla_weights so the
        # residual-flow head can learn from the start.
        if not args.use_tactile_refine_flow:
            nn.init.zeros_(model.final_layer_tactile.mlp.fc2.weight)
            if model.final_layer_tactile.mlp.fc2.bias is not None:
                nn.init.zeros_(model.final_layer_tactile.mlp.fc2.bias)
        accelerator.print("Tactile expert re-initialized (resume_source=pretrain or no resume).")
    elif resumed_tactile:
        accelerator.print("Tactile expert weights kept from resumed midtrain checkpoint.")

    for name, param in model.named_parameters():
        if name.startswith("visual") or name.startswith("deform_encoder"):
            param.requires_grad = False
        elif "_tactile" in name or "final_layer_tactile" in name:
            param.requires_grad = (not freeze_tactile)
        else:
            param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Model: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    no_decay = ["bias", "norm.weight", "q_norm.weight", "k_norm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate)

    dataset = MidtrainTacFlareDataset(args, processor, accelerator)

    # ── Episode-cached dataloader knobs ──
    # persistent_workers=True keeps each worker's per-episode cache alive
    # across epochs (saves the ~30–60 s "first batch of epoch" stall after
    # epoch 0). prefetch_factor=4 buffers 4 batches per worker so the GPU
    # has work in flight while another worker stalls on a new-episode load
    # (bulk-read of raw.h5 tactile_deform + 3 mp4 sequential decodes).
    n_workers_train = getattr(args, "num_workers", 4)
    n_workers_val = max(1, n_workers_train // 2)

    # Val dataloader: NO custom sampler — let accelerate add its own
    # DistributedSampler when we prepare it. Keeping the custom sampler
    # here would also trigger the double-shard interaction.
    val_dataloader = None
    if getattr(args, "val_ratio", 0) > 0:
        val_dataset = dataset.create_val_split(
            val_ratio=args.val_ratio, seed=args.seed)
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.train_bsz_per_gpu, shuffle=False,
            drop_last=True, collate_fn=val_dataset.collate_fn,
            num_workers=n_workers_val, pin_memory=True,
            persistent_workers=(n_workers_val > 0),
            prefetch_factor=4 if n_workers_val > 0 else None,
        )

    train_sampler = EpisodeGroupedSampler(
        dataset, shuffle=True, seed=args.seed, drop_last=True)
    dataloader = DataLoader(
        dataset, batch_size=args.train_bsz_per_gpu, sampler=train_sampler,
        drop_last=True,
        collate_fn=dataset.collate_fn,
        num_workers=n_workers_train, pin_memory=True,
        persistent_workers=(n_workers_train > 0),
        prefetch_factor=4 if n_workers_train > 0 else None,
    )

    # Compute steps from ground truth so the LR scheduler doesn't get a
    # bogus total. EpisodeGroupedSampler already shards — so per-rank
    # samples = total // world_size, batches = that // batch_size_per_gpu.
    world_size = accelerator.num_processes
    samples_per_rank = len(dataset) // world_size
    steps_per_epoch = samples_per_rank // args.train_bsz_per_gpu
    num_training_steps = (
        steps_per_epoch * args.n_epochs
        // accelerator.gradient_accumulation_steps
    )
    accelerator.print(
        f"[Midtrain] world_size={world_size}, samples_per_rank={samples_per_rank}, "
        f"steps_per_epoch={steps_per_epoch}, total_steps={num_training_steps}")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_rates * num_training_steps),
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    # IMPORTANT: do NOT prepare the training dataloader. EpisodeGroupedSampler
    # already shards by rank; accelerator.prepare() would wrap the
    # batch_sampler in BatchSamplerShard and shard a SECOND time, giving each
    # rank 1/world_size² of the data. We move train batches to device
    # manually inside the loop.
    if val_dataloader is not None:
        model, optimizer, val_dataloader = accelerator.prepare(
            model, optimizer, val_dataloader)
    else:
        model, optimizer = accelerator.prepare(model, optimizer)

    metric = TrainingMetrics(device=torch.cuda.current_device())
    global_step = 0
    T_per_frame = args.n_flare_tokens_per_frame
    S_steps = args.n_flare_steps
    K = T_per_frame * S_steps  # total flare tokens
    use_flare = bool(args.use_flare and K > 0)
    flare_layer_idx = args.flare_layer_index
    device = accelerator.device
    model.train()

    for epoch in range(args.n_epochs):
        # Required for proper shuffling each epoch when the dataloader is
        # not prepared by accelerate.
        train_sampler.set_epoch(epoch)

        from tqdm import tqdm
        it = (tqdm(dataloader, total=steps_per_epoch, desc=f"Epoch {epoch}")
              if accelerator.is_main_process else dataloader)

        for batch in it:
            # Manually move tensors to device since the train dataloader is
            # not prepared by accelerate (see comment above).
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            raw_model = accelerator.unwrap_model(model)

            inputs_embeds = raw_model.prepare_inputs_embeds(
                input_ids=batch["input_ids"],
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
            )

            n_slow_imgs = batch["n_slow_images"]
            grid_thw = batch.get("image_grid_thw")
            B = inputs_embeds.shape[0]
            if grid_thw is not None and grid_thw.shape[0] > n_slow_imgs:
                merge = getattr(raw_model.visual, "spatial_merge_size",
                                getattr(processor.image_processor, "merge_size", 2))
                n_imgs_per_sample = grid_thw.shape[0] // B
                n_slow_img_tokens = sum(
                    int(g[0] * (g[1] // merge) * (g[2] // merge))
                    for g in grid_thw[:n_imgs_per_sample][:n_slow_imgs]
                )
                slow_embeds, fast_embeds = split_slow_fast_embeds(
                    inputs_embeds, batch["input_ids"],
                    raw_model.image_token_id, n_slow_img_tokens)
            else:
                slow_embeds = inputs_embeds
                fast_embeds = inputs_embeds[:, :0]

            L_slow = slow_embeds.shape[1]
            
            torch.set_printoptions(profile="full")
            # print(batch["input_ids"])
            # print("---------------------")
            # print(inputs_embeds.shape)
            # print(slow_embeds.shape)
            # print(fast_embeds.shape)
            # input("pause")

            pos_ids, _ = raw_model.get_rope_index(
                input_ids=batch["input_ids"],
                image_grid_thw=batch.get("image_grid_thw"),
                attention_mask=batch["attention_mask"],
            )
            pos_ids = pos_ids[:, :, :L_slow]

            if use_flare:
                flare_q = raw_model.flare_queries.expand(B, -1, -1).to(device=slow_embeds.device, dtype=slow_embeds.dtype)
                slow_embeds_ext = torch.cat([slow_embeds, flare_q], dim=1)
                pos_ids = extend_position_ids_for_flare(pos_ids, K)
                L_latent = slow_embeds_ext.shape[1]  # L_slow + K
            else:
                slow_embeds_ext = slow_embeds
                L_latent = L_slow

            if args.use_robot_state and batch["state_raw"] is not None:
                state_vec = batch["state_raw"].to(slow_embeds.device,
                                                   dtype=slow_embeds.dtype)
                state_embeds = raw_model.state_embedder(state_vec).unsqueeze(1)
            else:
                state_embeds = torch.empty((B, 0, slow_embeds.shape[2]), device=slow_embeds.device, dtype=slow_embeds.dtype)
            n_state = state_embeds.shape[1]

            noisy_actions = raw_model.x_embedder(
                batch["noisy_actions"].to(slow_embeds.dtype))
            timesteps = raw_model.t_embedder(
                batch["timesteps"].to(slow_embeds.dtype)).unsqueeze(1)

            chunk = args.action_chunk
            target = batch["target"].to(slow_embeds.dtype)
            n_fast = fast_embeds.shape[1]
            r_target_norm = None  # set only when Paradigm C ran with tactile

            if args.use_tactile_refine_flow:  # Paradigm C: action-only flow + tactile residual flow
                # 1) L_flow — action expert single-step (tactile-blind sequence)
                full_embeds = torch.cat([
                    slow_embeds_ext,
                    fast_embeds, state_embeds, timesteps, noisy_actions,
                ], dim=1)
                L_total = full_embeds.shape[1]
                outputs = model.model(
                    inputs_embeds=full_embeds,
                    position_ids=pos_ids,
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                    output_hidden_states=use_flare,
                    latent_indexes=torch.arange(0, L_latent, device=full_embeds.device),
                    action_indexes=torch.arange(L_latent, L_total, device=full_embeds.device),
                    tactile_indexes=torch.arange(0, 0, device=full_embeds.device),
                )
                hidden = outputs.last_hidden_state
                act_pred_start = L_latent + n_fast + n_state + 1
                v_act = raw_model.final_layer(
                    hidden[:, act_pred_start: act_pred_start + chunk, :])
                loss_act = nn.MSELoss()(v_act, target)

                # 2) Â + cached_kv via no_grad full action flow (tactile-blind)
                has_any_tac = bool(args.use_tactile_vec or args.use_tactile_deform)
                if has_any_tac:
                    fe = fast_embeds if n_fast > 0 else None
                    se = state_embeds if n_state > 0 else None
                    ahat_noise = torch.randn_like(batch["noisy_actions"])
                    with torch.no_grad():
                        a_hat, cached_kv, n_action_in_cache = (
                            raw_model.forward_flow_action_only(
                                inputs_embeds=slow_embeds_ext,
                                position_ids=pos_ids,
                                noise=ahat_noise,
                                attention_mask=batch["attention_mask"],
                                state_embeds=se,
                                fast_embeds=fe,
                                num_steps=args.action_flow_train_steps,
                                refresh_clean_kv=True,
                            ))

                    # 3) L_refine — tactile residual flow single-step
                    norm_actions_gt = batch["norm_actions"].to(slow_embeds.dtype)
                    # Optional jitter on Â: if the action expert overfits and
                    # produces Â ≈ A_demo, the residual r_target collapses and
                    # the tactile head learns nothing.  Adding a small Gaussian
                    # perturbation guarantees a non-zero training signal even
                    # when the base policy is near-perfect.
                    a_hat_for_residual = a_hat.detach()
                    if args.tactile_residual_jitter > 0:
                        a_hat_for_residual = a_hat_for_residual + (
                            args.tactile_residual_jitter
                            * torch.randn_like(a_hat_for_residual))
                    r_target = norm_actions_gt - a_hat_for_residual
                    eps_r    = batch["eps_r"].to(slow_embeds.dtype)
                    tau_r    = batch["time_r"].to(slow_embeds.dtype)
                    tau_r_b  = tau_r[:, None, None]
                    r_tau    = (1 - tau_r_b) * r_target + tau_r_b * eps_r
                    v_target_r = eps_r - r_target

                    loss_tac = nn.MSELoss()(
                        raw_model.tactile_residual_train_step(
                            cached_kv=cached_kv,
                            latent_position_ids=pos_ids,
                            n_action_in_cache=n_action_in_cache,
                            base_chunk=a_hat_for_residual,
                            tactile_f6=batch.get("tactile_f6s_delayed"),
                            tactile_deform=batch.get("tactile_deforms_delayed"),
                            r_tau=r_tau,
                            tau=tau_r,
                        ),
                        v_target_r,
                    )
                    # Diagnostic: track residual magnitude.  Decaying toward 0
                    # means the tactile head has lost its training signal.
                    r_target_norm = r_target.float().abs().mean().detach()
                else:
                    loss_tac = 0.0

            elif is_stage1: # pretrain
                full_embeds = torch.cat([
                    slow_embeds_ext,
                    fast_embeds, state_embeds, timesteps, noisy_actions,
                ], dim=1)

                n_action = n_fast + n_state + 1 + chunk
                L_total = full_embeds.shape[1]
                latent_indexes  = torch.arange(0, L_latent, device=full_embeds.device)
                action_indexes  = torch.arange(L_latent, L_total, device=full_embeds.device)
                tactile_indexes = torch.arange(0, 0, device=full_embeds.device)

                outputs = model.model(
                    inputs_embeds=full_embeds,
                    position_ids=pos_ids,
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                    output_hidden_states=use_flare,
                    latent_indexes=latent_indexes,
                    action_indexes=action_indexes,
                    tactile_indexes=tactile_indexes,
                )
                hidden = outputs.last_hidden_state

                act_pred_start = L_latent + n_fast + n_state + 1
                v_act = raw_model.final_layer(hidden[:, act_pred_start: act_pred_start + chunk, :])
                loss_act = nn.MSELoss()(v_act, target)
                loss_tac = 0.0

            else: # mid/post training with tactile
                tac_parts = []
                if args.use_tactile_vec and batch["tactile_f6s"] is not None:
                    tac_parts.append(raw_model.tacf6_embedder(batch["tactile_f6s"].to(slow_embeds.dtype)))
                if args.use_tactile_deform and batch["tactile_deforms"] is not None:
                    deforms = batch["tactile_deforms"].to(slow_embeds.device, dtype=slow_embeds.dtype)
                    Bs, nf, C, H, W = deforms.shape
                    with torch.no_grad():
                        feats = raw_model.deform_encoder(deforms.view(-1, C, H, W))
                    feats = feats.view(Bs, nf, -1)
                    tac_parts.append(raw_model.deform_proj(feats.to(slow_embeds.dtype)))

                if tac_parts:
                    tactile_embeds = torch.cat(tac_parts, dim=1)
                else:
                    tactile_embeds = torch.empty((B, 0, slow_embeds.shape[2]), device=slow_embeds.device, dtype=slow_embeds.dtype)

                has_tactile = tactile_embeds.shape[1] > 0
                n_action = n_fast + n_state + 1 + chunk

                if has_tactile:
                    noisy_actions_tac = raw_model.x_embedder(batch["noisy_actions"].to(slow_embeds.dtype))
                    timesteps_tac = raw_model.t_embedder(batch["timesteps"].to(slow_embeds.dtype)).unsqueeze(1)
                    full_embeds = torch.cat([
                        slow_embeds_ext,
                        fast_embeds, state_embeds, timesteps, noisy_actions,
                        tactile_embeds, timesteps_tac, noisy_actions_tac,
                    ], dim=1)
                else:
                    full_embeds = torch.cat([
                        slow_embeds_ext,
                        fast_embeds, state_embeds, timesteps, noisy_actions,
                    ], dim=1)

                L_total = full_embeds.shape[1]
                latent_indexes = torch.arange(0, L_latent, device=full_embeds.device)
                action_indexes = torch.arange(L_latent, L_latent + n_action,
                                               device=full_embeds.device)
                if has_tactile:
                    tactile_indexes = torch.arange(L_latent + n_action, L_total,
                                                   device=full_embeds.device)
                else:
                    tactile_indexes = torch.arange(0, 0, device=full_embeds.device)
                    
                # print(latent_indexes)
                # print(action_indexes)
                # print(tactile_indexes)
                # input("pause")

                outputs = model.model(
                    inputs_embeds=full_embeds,
                    position_ids=pos_ids,
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                    output_hidden_states=use_flare,
                    latent_indexes=latent_indexes,
                    action_indexes=action_indexes,
                    tactile_indexes=tactile_indexes,
                )
                hidden = outputs.last_hidden_state

                act_pred_start = L_latent + n_fast + n_state + 1
                v_act = raw_model.final_layer(
                    hidden[:, act_pred_start: act_pred_start + chunk, :])
                loss_act = nn.MSELoss()(v_act, target)

                if has_tactile:
                    delta_v = raw_model.final_layer_tactile(hidden[:, -chunk:, :])
                    residual_target = target - v_act.detach()
                    loss_tac = nn.MSELoss()(delta_v, residual_target)
                else:
                    loss_tac = 0.0

            loss_flare = 0.0
            if use_flare and batch["flare_pixel_values"] is not None:
                # Extract flare hidden states from intermediate layer or last layer
                if flare_layer_idx == -1:
                    flare_source = hidden  # last layer (already normed)
                else:
                    # outputs.hidden_states is a tuple: (embed, layer0, layer1, ..., layerN)
                    # flare_layer_idx can be negative (e.g. -7 for ~3/4 depth)
                    all_hs = outputs.hidden_states
                    # Skip index 0 (embedding), so layer indices are 1..N
                    n_layers = len(all_hs) - 1  # exclude embedding
                    if flare_layer_idx < 0:
                        layer_i = n_layers + flare_layer_idx  # e.g. 28 + (-7) = 21
                    else:
                        layer_i = flare_layer_idx
                    layer_i = max(0, min(layer_i, n_layers - 1))
                    flare_source = all_hs[layer_i + 1]  # +1 to skip embedding

                flare_hidden = flare_source[:, L_slow: L_slow + K, :]  # [B, K, H]
                flare_pred = raw_model.flare_proj(flare_hidden)  # [B, K, H]

                f_pv = batch["flare_pixel_values"].to(device=flare_pred.device, dtype=flare_pred.dtype)
                f_thw = batch["flare_grid_thw"].to(device=flare_pred.device)

                with torch.no_grad():
                    vit_out = raw_model.visual(f_pv, grid_thw=f_thw)
                    features = vit_out[0] if isinstance(vit_out, (tuple, list)) else vit_out

                    # Adaptive pool each frame to T_per_frame tokens
                    merge = getattr(raw_model.visual, "spatial_merge_size", 2)
                    frame_feats = []
                    offset = 0
                    for g in f_thw:
                        n_tok = int(g[0] * (g[1] // merge) * (g[2] // merge))
                        frame_tokens = features[offset: offset + n_tok]  # [n_tok, H]
                        # Adaptive average pool: [n_tok, H] → [T_per_frame, H]
                        pooled = F.adaptive_avg_pool1d(
                            frame_tokens.unsqueeze(0).permute(0, 2, 1),  # [1, H, n_tok]
                            T_per_frame,
                        ).permute(0, 2, 1).squeeze(0)  # [T_per_frame, H]
                        frame_feats.append(pooled)
                        offset += n_tok
                    # frame_feats: list of B*S_steps tensors, each [T_per_frame, H]
                    # Reshape to [B, S_steps * T_per_frame, H] = [B, K, H]
                    flare_targets = torch.stack(frame_feats).view(B, K, -1)

                # Cosine similarity loss
                pred_norm = F.normalize(flare_pred, dim=-1)
                tgt_norm = F.normalize(flare_targets.detach(), dim=-1)
                loss_flare = (1.0 - (pred_norm * tgt_norm).sum(dim=-1)).mean()

            loss = loss_act
            if torch.is_tensor(loss_tac):
                tac_w = (args.tactile_refine_loss_weight if args.use_tactile_refine_flow
                         else args.tactile_loss_weight)
                loss = loss + tac_w * loss_tac
            if torch.is_tensor(loss_flare):
                loss = loss + args.flare_loss_weight * loss_flare

            metric.update(loss, loss_act, loss_tac, loss_flare)
            accelerator.backward(loss)

            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                m = metric.get_metric()
                if accelerator.is_main_process:
                    lr_now = lr_scheduler.get_last_lr()[0]
                    if hasattr(it, "set_postfix"):
                        it.set_postfix(epoch=epoch, step=global_step,
                                       loss=f"{m['total_loss']:.6f}",
                                       act=f"{m['action_loss']:.6f}",
                                       tac=f"{m['tactile_loss']:.6f}",
                                       fut=f"{m['flare_loss']:.6f}",
                                       lr=f"{lr_now:.2e}")
                    log_dict = {
                        "total_loss":   m["total_loss"],
                        "action_loss":  m["action_loss"],
                        "tactile_loss": m["tactile_loss"],
                        "flare_loss":  m["flare_loss"],
                        "lr": lr_now,
                    }
                    if r_target_norm is not None:
                        log_dict["refine/r_target_norm"] = float(r_target_norm)
                    wandb.log(log_dict, step=global_step)

            # ── Validation ──
            if (val_dataloader is not None
                    and getattr(args, "val_freq", 0) > 0
                    and (global_step + 1) % args.val_freq == 0):
                val_m = run_validation(
                    model, val_dataloader, accelerator, args,
                    is_stage1, use_flare, K, T_per_frame, flare_layer_idx)
                if accelerator.is_main_process:
                    accelerator.print(
                        f"  [Val step={global_step}] "
                        f"act={val_m['val/action_loss']:.6f} "
                        f"tac={val_m['val/tactile_loss']:.6f}")
                    wandb.log(val_m, step=global_step)

            global_step += 1

        if (epoch + 1) % args.save_freq == 0 or epoch == args.n_epochs - 1:
            accelerator.wait_for_everyone()
            save_checkpoint(model, processor, accelerator, args,
                            epoch, global_step, dataset.stats_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="qwen3vl_midtrain_tac_flare")
    parser.add_argument("--run_name", type=str, default="run_1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True,
                        help="Directory containing batch_*/pretrain_manifest.json "
                             "(produced by gen_pretrain_mecka_parallel.py).")
    parser.add_argument("--data_path", type=str, default="",
                        help="(Unused; kept for backwards compatibility.)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--max_ckpts", type=int, default=10)

    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--train_bsz_per_gpu", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.0)
    parser.add_argument("--warmup_rates", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--action_dim", type=int, default=62)
    parser.add_argument("--action_chunk", type=int, default=16)
    parser.add_argument("--use_robot_state", type=int, default=0)
    parser.add_argument("--use_tactile_vec", type=int, default=0)
    parser.add_argument("--use_tactile_deform", type=int, default=1)
    parser.add_argument("--deform_encoder_ckpt", type=str, default="")
    parser.add_argument("--tactile_intermediate_size", type=int, default=0)
    parser.add_argument("--training_stage", type=int, default=2, choices=[1, 2])
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--resume_source", type=str, default="pretrain",
                        choices=["pretrain", "midtrain"],
                        help="'pretrain': resumed ckpt did not train tactile (re-init); "
                             "'midtrain': resumed ckpt already trained tactile (keep).")
    parser.add_argument("--tactile_loss_weight", type=float, default=1.0)
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("W", "H"))

    # Paradigm C — action-only slow flow + tactile residual flow refinement
    parser.add_argument("--use_tactile_refine_flow", type=int, default=0,
                        help="1: tactile expert is trained as a residual-flow refinement "
                             "head over the action expert's clean chunk Â. 0: legacy "
                             "Paradigm A (delta_v residual on flow velocity).")
    parser.add_argument("--tactile_refine_loss_weight", type=float, default=1.0,
                        help="Weight on L_refine when use_tactile_refine_flow=1 "
                             "(replaces tactile_loss_weight for that paradigm).")
    parser.add_argument("--tactile_refine_noise_scale", type=float, default=0.1,
                        help="Initial noise magnitude for the residual flow at τ=1. "
                             "Smaller values match the small-residual regime.")
    parser.add_argument("--tactile_residual_jitter", type=float, default=0.0,
                        help="Std of Gaussian jitter added to Â before computing "
                             "r_target during training. Prevents the residual from "
                             "collapsing to 0 when the action expert overfits and "
                             "Â ≈ A_demo. 0.05 (5%% of normalized range) is a good "
                             "starting point if the refine/r_target_norm metric is "
                             "trending toward 0. 0 (default) disables.")
    parser.add_argument("--action_flow_train_steps", type=int, default=5,
                        help="Number of Euler steps for the no_grad action flow "
                             "that produces Â during training. 5 is a good "
                             "speed/accuracy trade-off; bump to 10 for evaluation.")
    parser.add_argument("--action_flow_eval_steps", type=int, default=10,
                        help="Number of Euler steps for action flow at deployment.")
    parser.add_argument("--tactile_delay_offsets", type=int, nargs="+",
                        default=[0, 4, 8, 12],
                        help="Delay offsets (frames) sampled uniformly when reading "
                             "tactile observations for L_refine.")

    # Flare
    parser.add_argument("--use_flare", type=int, default=1, help="Enable flare visual prediction for latent expert.")
    parser.add_argument("--n_flare_tokens_per_frame", type=int, default=4, help="Number of tokens per future frame.")
    parser.add_argument("--n_flare_steps", type=int, default=8, help="Number of future steps to predict.")
    parser.add_argument("--flare_loss_weight", type=float, default=0.5, help="Weight for flare prediction cosine loss.")
    parser.add_argument("--flare_frame_stride", type=int, default=2, help="Temporal stride for flare frame targets.")
    parser.add_argument("--flare_layer_index", type=int, default=-1, help="Layer to extract flare hidden states from (-1=last, e.g. -7 for ~3/4 depth).")
    parser.add_argument("--frame_stride", type=int, default=2)

    # Validation
    parser.add_argument("--val_ratio", type=float, default=0.0,
                        help="Fraction of samples for validation (0=disable)")
    parser.add_argument("--val_freq", type=int, default=0,
                        help="Run validation every N steps (0=disable)")
    parser.add_argument("--max_val_batches", type=int, default=50,
                        help="Max batches per validation run")

    args = parser.parse_args()

    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name, args.run_name)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)

