"""
EgoDex large-scale pretraining script for Qwen3-VL VLA (Stage 1) WITH flare.

Extends train_qwen3vl_pretrain_egodex.py with future visual prediction for the
latent expert, using the same flare mechanism as train_qwen3vl_flare.py.

Key difference from post-training flare: future frames are read from the SAME
ego_view.mp4 video by seeking to frame_t + (k+1)*flare_frame_stride, rather
than from separate image files.

When --use_flare 0, this script behaves identically to the base pretrain script.
"""

import os, sys

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import glob
import json
import math
import re
import shutil
import logging
import argparse
import warnings

import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import wandb
import PIL.Image
from datetime import timedelta

from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler as _DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator, DataLoaderConfiguration, InitProcessGroupKwargs
from transformers import AutoProcessor, set_seed

from qwen_vla import Qwen3VLVLAModel, extend_position_ids_for_flare, split_slow_fast_embeds

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


# ---------------------------------------------------------------------------
# Tracking-error noise injection: augment robot state with realistic noise
# sampled from the tracking error distribution computed during data generation.
# ---------------------------------------------------------------------------

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


class EpisodeGroupedSampler(_DistributedSampler):
    """
    Distributed sampler that groups frame indices by episode for I/O locality.

    Shuffles *episode* order for randomness, but emits each episode's frame
    indices contiguously so that per-worker episode caching is effective.
    Subclasses DistributedSampler so accelerate/DeepSpeed doesn't replace it.

    Memory-efficient: computes only this rank's indices via numpy cumsum +
    searchsorted, avoiding a full-dataset Python list (which would OOM at
    billions of transitions).
    """

    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, seed=0, drop_last=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=shuffle, seed=seed, drop_last=drop_last)
        self._cum_frames = dataset._cum_frames.copy()
        self._num_episodes = dataset._num_episodes
        # Per-episode frame counts (unshuffled)
        self._frame_counts = np.diff(self._cum_frames, prepend=0)
        # Per-episode original start index (unshuffled)
        self._orig_starts = np.zeros(self._num_episodes, dtype=np.int64)
        if self._num_episodes > 1:
            self._orig_starts[1:] = self._cum_frames[:-1]
        # Number of leading indices to drop from this rank's chunk on the
        # next __iter__ call (used for resume). Zero-cost: dropped indices
        # never reach the workers, so no data is loaded for them.
        self.start_index = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            ep_perm = torch.randperm(self._num_episodes, generator=g).numpy()
        else:
            ep_perm = np.arange(self._num_episodes)

        # Cumulative frame counts in *shuffled* episode order
        shuffled_fc = self._frame_counts[ep_perm]
        shuffled_cum = np.cumsum(shuffled_fc)
        total_frames = int(shuffled_cum[-1])

        # This rank's contiguous chunk in the flat index space
        chunk = self.num_samples
        rank_start = self.rank * chunk
        rank_end = rank_start + chunk
        actual_end = min(rank_end, total_frames)

        # Find which shuffled episodes overlap [rank_start, actual_end)
        ep_first = int(np.searchsorted(shuffled_cum, rank_start, side='right'))
        ep_last = int(np.searchsorted(shuffled_cum, actual_end, side='right'))
        ep_last = min(ep_last, self._num_episodes - 1)

        # Build only this rank's indices using numpy (not a 2B Python list)
        parts = []
        for i in range(ep_first, ep_last + 1):
            ep_idx = int(ep_perm[i])
            shuf_start = int(shuffled_cum[i - 1]) if i > 0 else 0
            shuf_end = int(shuffled_cum[i])

            # Clip to this rank's range
            t_start = max(0, rank_start - shuf_start)
            t_end = min(int(self._frame_counts[ep_idx]), actual_end - shuf_start)

            if t_start < t_end:
                base = int(self._orig_starts[ep_idx])
                parts.append(np.arange(base + t_start, base + t_end, dtype=np.int64))

        indices = np.concatenate(parts) if parts else np.array([], dtype=np.int64)

        # Padding: if rank's chunk extends beyond total_frames, wrap around
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

        indices = indices[:chunk]
        if self.start_index > 0:
            indices = indices[self.start_index:]
        return iter(indices.tolist())


class EgoDexPretrainFlareDataset(Dataset):
    """
    Map-style dataset with a compact cumsum index (same as base pretrain).

    Uses per-worker episode caching: bulk-reads all HDF5 data + video frames
    once per episode, then serves individual frames from memory.  Combined
    with EpisodeGroupedSampler this reduces per-sample I/O from ~200ms to
    ~0.1ms.

    Compared to the base EgoDexPretrainDataset, __getitem__ also returns
    future frames for flare visual prediction, read from the cached video.
    """

    def __init__(self, data_root: str, config, processor, accelerator):
        super().__init__()
        self.config = config
        self.processor = processor
        self.accelerator = accelerator

        _episode_dirs = []
        _frame_counts = []
        all_action_q01 = []
        all_action_q99 = []
        all_state_q01 = []
        all_state_q99 = []
        all_te_mean = []
        all_te_std = []

        manifest_paths = sorted(
            glob.glob(os.path.join(data_root, "*", "pretrain_manifest.json"))
        )
        if not manifest_paths:
            raise FileNotFoundError(
                f"No pretrain_manifest.json found under {data_root}/*/")

        for mp in manifest_paths:
            with open(mp, "r") as f:
                manifest = json.load(f)
            for ep in manifest["episodes"]:
                _episode_dirs.append(ep["episode_dir"])
                _frame_counts.append(ep["num_frames"])
            stats = manifest.get("statistics", {})
            if "action" in stats and "state" in stats:
                all_action_q01.append(np.array(stats["action"]["q01"], dtype=np.float32))
                all_action_q99.append(np.array(stats["action"]["q99"], dtype=np.float32))
                all_state_q01.append(np.array(stats["state"]["q01"], dtype=np.float32))
                all_state_q99.append(np.array(stats["state"]["q99"], dtype=np.float32))
            if "tracking_error" in stats:
                all_te_mean.append(np.array(stats["tracking_error"]["mean"], dtype=np.float32))
                all_te_std.append(np.array(stats["tracking_error"]["std"], dtype=np.float32))

        accelerator.print(f"Loaded {len(manifest_paths)} batch manifests")

        self.action_min = np.min(np.stack(all_action_q01), axis=0)
        self.action_max = np.max(np.stack(all_action_q99), axis=0)
        self.state_min = np.min(np.stack(all_state_q01), axis=0)
        self.state_max = np.max(np.stack(all_state_q99), axis=0)

        # Tracking error stats for state noise injection
        te_dim = (config.action_dim // 31) * 28
        if all_te_mean:
            self.te_mean = np.mean(np.stack(all_te_mean), axis=0)
            self.te_std  = np.mean(np.stack(all_te_std),  axis=0)
        else:
            self.te_mean = np.zeros(te_dim, dtype=np.float32)
            self.te_std  = np.zeros(te_dim, dtype=np.float32)

        self.action_mask = np.ones(config.action_dim, dtype=bool)
        self.state_mask = np.ones(config.action_dim, dtype=bool)

        if config.image_size:
            self.image_size = tuple(config.image_size)
        else:
            self.image_size = None

        # Flare config
        self.use_flare = bool(getattr(config, "use_flare", 0))
        self.n_flare_steps = getattr(config, "n_flare_steps", 0) if self.use_flare else 0
        self.flare_frame_stride = getattr(config, "flare_frame_stride", 4)

        # ── Build compact index via cumulative frame counts ──
        # np.searchsorted on this array maps flat idx → (ep_idx, frame_t)
        # in O(log N) with negligible memory (~8 bytes/episode vs ~64 bytes/frame).
        self._episode_dirs = _episode_dirs
        _frame_counts = np.array(_frame_counts, dtype=np.int64)
        self._cum_frames = np.cumsum(_frame_counts)
        self._total_transitions = int(self._cum_frames[-1])
        self._num_episodes = len(_episode_dirs)

        accelerator.print(f"EgoDex pretrain (flare): {self._num_episodes} episodes, "
                          f"{self._total_transitions} transitions, "
                          f"flare={self.n_flare_steps} steps x stride {self.flare_frame_stride}")

    def __len__(self):
        return self._total_transitions

    @property
    def total_transitions(self):
        return self._total_transitions

    def create_val_split(self, val_ratio=0.02, seed=42):
        """
        Split episodes into train/val. Returns a new dataset for validation.
        Modifies *self* in-place to keep only training episodes.
        """
        import copy
        rng = np.random.RandomState(seed)
        n_val = max(1, int(self._num_episodes * val_ratio))
        perm = rng.permutation(self._num_episodes)
        val_eps = sorted(perm[:n_val].tolist())
        train_eps = sorted(perm[n_val:].tolist())

        frame_counts = np.diff(self._cum_frames, prepend=0)

        val_ds = copy.copy(self)
        val_ds._episode_dirs = [self._episode_dirs[i] for i in val_eps]
        val_fc = frame_counts[val_eps]
        val_ds._cum_frames = np.cumsum(val_fc)
        val_ds._total_transitions = int(val_ds._cum_frames[-1])
        val_ds._num_episodes = len(val_eps)

        self._episode_dirs = [self._episode_dirs[i] for i in train_eps]
        train_fc = frame_counts[train_eps]
        self._cum_frames = np.cumsum(train_fc)
        self._total_transitions = int(self._cum_frames[-1])
        self._num_episodes = len(train_eps)

        self.accelerator.print(
            f"Train/Val split: {self._num_episodes} train eps "
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
    def _find_head_video(ep_dir: str) -> Optional[str]:
        candidate = os.path.join(ep_dir, "ego_view.mp4")
        if os.path.isfile(candidate):
            return candidate
        matches = glob.glob(os.path.join(ep_dir, "*head*.mp4"))
        return matches[0] if matches else None

    def _load_episode_cache(self, ep_idx: int):
        """
        Preload all HDF5 data + video frames for an episode in one shot.

        HDF5:  single open -> bulk read states[:] and action_chunks[:] -> close.
        Video: single open -> sequential cap.read() for all frames -> close.

        With EpisodeGroupedSampler, consecutive __getitem__ calls hit the same
        episode, so this is called once per episode per worker — amortizing
        the I/O cost to near-zero per frame.
        """
        ep_dir = self._episode_dirs[ep_idx]
        pretrain_h5 = os.path.join(ep_dir, "pretrain.hdf5")
        video_path = self._find_head_video(ep_dir)

        self._cache_ep_idx = ep_idx
        self._cache_h5 = None
        self._cache_frames = None

        # ── Bulk-read HDF5 ──
        if pretrain_h5 and os.path.isfile(pretrain_h5):
            try:
                with h5py.File(pretrain_h5, "r") as f:
                    self._cache_h5 = {
                        "states": f["states"][:],
                        "action_chunks": f["action_chunks"][:],
                        "language": f.attrs.get("language", ""),
                    }
            except Exception:
                pass

        # ── Sequential video decode (no random seeks) ──
        if video_path and os.path.isfile(video_path):
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if frames:
                self._cache_frames = frames

    def __getitem__(self, idx: int) -> Dict:
        # O(log N) lookup: flat idx -> (episode, frame) via cumulative sums
        ep_idx = int(np.searchsorted(self._cum_frames, idx, side="right"))
        frame_t = int(idx - (int(self._cum_frames[ep_idx - 1]) if ep_idx > 0 else 0))

        fallback_frame = np.zeros((288, 384, 3), dtype=np.uint8)
        fallback = {
            "frame": fallback_frame,
            "state": np.zeros(self.config.action_dim, dtype=np.float32),
            "action_chunk": np.zeros((self.config.action_chunk, self.config.action_dim), dtype=np.float32),
            "language": "",
            "flare_frames": [fallback_frame.copy() for _ in range(self.n_flare_steps)],
        }

        # ── Per-worker episode cache: load once, reuse for all frames ──
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

        # Current frame from cache
        if self._cache_frames is not None and frame_t < len(self._cache_frames):
            current_frame = self._cache_frames[frame_t]
        else:
            current_frame = fallback_frame

        # Future frames for flare (also from cache)
        num_cached = len(self._cache_frames) if self._cache_frames else 0
        flare_frames = []
        for k in range(self.n_flare_steps):
            future_t = frame_t + (k + 1) * self.flare_frame_stride
            future_t = min(future_t, num_cached - 1) if num_cached > 0 else -1
            if future_t >= 0 and self._cache_frames is not None:
                flare_frames.append(self._cache_frames[future_t])
            else:
                flare_frames.append(current_frame.copy() if current_frame is not fallback_frame
                                    else fallback_frame.copy())

        return {
            "frame": current_frame,
            "state": state,
            "action_chunk": action_chunk,
            "language": language,
            "flare_frames": flare_frames,
        }

    def collate_fn(self, batch: List[Dict]) -> Dict:
        cfg = self.config
        B = len(batch)

        # ── Actions ──
        actions = np.stack([x["action_chunk"] for x in batch], axis=0)
        norm_actions = self._normalize(actions, self.action_mask, self.action_min, self.action_max)
        norm_actions = torch.tensor(norm_actions, dtype=torch.bfloat16)

        # ── Flow matching noise ──
        device_cpu = norm_actions.device
        d = torch.distributions.Beta(
            torch.tensor(1.5, dtype=torch.float32, device=device_cpu),
            torch.tensor(1.0, dtype=torch.float32, device=device_cpu),
        )
        time = (d.sample((B,)) * 0.999 + 0.001).to(torch.bfloat16)
        t_ = time[:, None, None]
        noise = torch.randn_like(norm_actions)
        x_t = t_ * noise + (1 - t_) * norm_actions
        u_t = noise - norm_actions

        # ── State ──
        state_raw_list = []
        if cfg.use_robot_state:
            for x in batch:
                s = np.array(x["state"], dtype=np.float32)
                s = add_tracking_error_noise(s, self.te_mean, self.te_std,
                                             cfg.action_dim)
                ns = self._normalize(s, self.state_mask, self.state_min, self.state_max)
                state_raw_list.append(torch.tensor(ns, dtype=torch.bfloat16))

        # ── Single message: [slow_img | text | fast_img] ──
        all_input_ids = []
        all_pixel_values = []
        all_grid_thw = []
        n_slow_images = 1

        for x in batch:
            img = PIL.Image.fromarray(x["frame"])
            if self.image_size is not None:
                img = img.resize(self.image_size, PIL.Image.LANCZOS)

            content = [
                {"type": "image"},
                {"type": "text", "text": x.get("language", "")},
                {"type": "image"},
            ]
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inp = self.processor(
                text=text, images=[img, img],
                return_tensors="pt", padding=False,
            )
            all_input_ids.append(inp.input_ids[0])
            if "pixel_values" in inp and inp.pixel_values is not None:
                all_pixel_values.append(inp.pixel_values)
                all_grid_thw.append(inp.image_grid_thw)

        # ── Flare future frames ──
        flare_pixel_values = None
        flare_grid_thw = None

        if self.n_flare_steps > 0:
            flare_pil_imgs = []
            for x in batch:
                for ff in x["flare_frames"]:
                    pil_img = PIL.Image.fromarray(ff)
                    if self.image_size is not None:
                        pil_img = pil_img.resize(self.image_size, PIL.Image.LANCZOS)
                    flare_pil_imgs.append(pil_img)

            flare_inp = self.processor.image_processor(flare_pil_imgs, return_tensors="pt")
            flare_pixel_values = flare_inp.pixel_values.to(torch.bfloat16)
            flare_grid_thw = flare_inp.image_grid_thw

        # ── Padding ──
        pad_id = self.processor.tokenizer.pad_token_id or 0
        max_len = max(ids.shape[0] for ids in all_input_ids)

        padded_ids = []
        attention_ms = []
        for ids in all_input_ids:
            pad_len = max_len - ids.shape[0]
            padded_ids.append(F.pad(ids, (pad_len, 0), value=pad_id))
            attn = torch.ones(max_len, dtype=torch.long)
            if pad_len > 0:
                attn[:pad_len] = 0
            attention_ms.append(attn)

        input_ids = torch.stack(padded_ids)
        attention_mask = torch.stack(attention_ms)
        pixel_values = (torch.cat(all_pixel_values, dim=0) if all_pixel_values else None)
        image_grid_thw = (torch.cat(all_grid_thw, dim=0) if all_grid_thw else None)
        state_raw = torch.stack(state_raw_list) if state_raw_list else None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "n_slow_images": n_slow_images,
            "noisy_actions": x_t,
            "target": u_t,
            "timesteps": time,
            "state_raw": state_raw,
            "flare_pixel_values": flare_pixel_values,
            "flare_grid_thw": flare_grid_thw,
        }


def save_checkpoint(model, processor, accelerator, args, epoch, global_step, dataset,
                    save_full_state=True, lr_scheduler=None):
    """Save a checkpoint.

    Always writes `model.pt` + metadata (compatible with downstream eval).
    When save_full_state=True, additionally writes a `state/` subdir via
    `accelerator.save_state` containing optimizer, LR scheduler and RNG state
    (sharded in DeepSpeed format), plus `training_state.json` with the
    global_step / epoch needed to resume.
    """
    save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
    state_dir = os.path.join(save_dir, "state")

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

        with open(os.path.join(save_dir, "stats_data.json"), "w") as f:
            json.dump({"egodex": {
                "action": {
                    "mask": dataset.action_mask.tolist(),
                    "q01": dataset.action_min.tolist(),
                    "q99": dataset.action_max.tolist(),
                },
                "state": {
                    "mask": dataset.state_mask.tolist(),
                    "q01": dataset.state_min.tolist(),
                    "q99": dataset.state_max.tolist(),
                },
            }}, f, indent=2)

        with open(os.path.join(save_dir, "training_args.json"), "w") as f:
            json.dump({
                "model_path": args.model_path,
                "action_dim": args.action_dim,
                "action_chunk": args.action_chunk,
                "use_robot_state": args.use_robot_state,
                "training_stage": 1,
                "data_root": args.data_root,
                "use_flare": args.use_flare,
                "n_flare_tokens_per_frame": args.n_flare_tokens_per_frame,
                "n_flare_steps": args.n_flare_steps,
                "flare_layer_index": args.flare_layer_index,
            }, f, indent=2)

        with open(os.path.join(save_dir, "training_state.json"), "w") as f:
            json.dump({
                "epoch": int(epoch),
                "global_step": int(global_step),
                "n_epochs": int(args.n_epochs),
                "seed": int(args.seed),
                "train_bsz_per_gpu": int(args.train_bsz_per_gpu),
                "learning_rate": float(args.learning_rate),
                "warmup_rates": float(args.warmup_rates),
                "min_lr_ratio": float(args.min_lr_ratio),
            }, f, indent=2)

    accelerator.wait_for_everyone()

    # Collective: save optimizer / scheduler / RNG for exact resume.
    if save_full_state:
        accelerator.save_state(state_dir)
        # Accelerate's DeepSpeed path does NOT write `scheduler.bin` when the
        # LR scheduler was prepared separately from the model+optimizer (as
        # we do here, since num_training_steps depends on the prepared
        # world_size). Save it ourselves so resume can restore the cosine
        # curve position.
        if lr_scheduler is not None and accelerator.is_main_process:
            torch.save(lr_scheduler.state_dict(),
                       os.path.join(state_dir, "scheduler.bin"))

    accelerator.wait_for_everyone()
    logger.info(f"Checkpoint {epoch}-{global_step} saved"
                f"{' (with resumable state)' if save_full_state else ''}.")


def _resolve_resume(resume_checkpoint):
    """Normalize the --resume_checkpoint arg to (model_pt_path, ckpt_dir, state_dir_or_None).

    Accepts either a path to `model.pt` or a checkpoint directory.
    `state_dir` is returned only when `<ckpt_dir>/state/` exists, signaling
    the new-style full-state checkpoint.
    """
    if not resume_checkpoint:
        return None, None, None
    path = resume_checkpoint.rstrip("/")
    if os.path.isdir(path):
        ckpt_dir = path
        model_pt = os.path.join(ckpt_dir, "model.pt")
    else:
        model_pt = path
        ckpt_dir = os.path.dirname(path)
    state_dir = os.path.join(ckpt_dir, "state")
    if not os.path.isdir(state_dir):
        state_dir = None
    return model_pt, ckpt_dir, state_dir


def _parse_step_from_ckpt_dir(ckpt_dir):
    """Parse (epoch, global_step) from 'checkpoint-{epoch}-{step}' dir name, or (None, None)."""
    base = os.path.basename(os.path.normpath(ckpt_dir)) if ckpt_dir else ""
    m = re.match(r"checkpoint-(\d+)-(\d+)$", base)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


class TrainingMetrics:
    def __init__(self, device):
        self.n_step = 0
        self.action_loss = torch.tensor(0.0, device=device)
        self.flare_loss = torch.tensor(0.0, device=device)
        self.total_loss = torch.tensor(0.0, device=device)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    def update(self, total, action, flare=0.0):
        self.n_step += 1
        self.total_loss += total.item() if torch.is_tensor(total) else total
        self.action_loss += action.item() if torch.is_tensor(action) else action
        self.flare_loss += flare.item() if torch.is_tensor(flare) else flare

    def get_metric(self, reset=True):
        if dist.is_initialized():
            for t in [self.total_loss, self.action_loss, self.flare_loss]:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
        denom = self.world_size * max(self.n_step, 1)
        m = {
            "total_loss": self.total_loss.item() / denom,
            "action_loss": self.action_loss.item() / denom,
            "flare_loss": self.flare_loss.item() / denom,
        }
        if reset:
            self.n_step = 0
            for t in [self.total_loss, self.action_loss, self.flare_loss]:
                t.fill_(0)
        return m


@torch.no_grad()
def run_validation(model, val_dataloader, accelerator, raw_model_fn,
                   use_flare, K, T_per_frame, flare_layer_idx, args):
    """Run validation and return averaged metrics."""
    model.eval()
    val_loss_sum = torch.tensor(0.0, device=torch.cuda.current_device())
    val_act_sum = torch.tensor(0.0, device=torch.cuda.current_device())
    val_flare_sum = torch.tensor(0.0, device=torch.cuda.current_device())
    n_val = 0
    max_val_batches = getattr(args, "max_val_batches", 50)

    for i, batch in enumerate(val_dataloader):
        if i >= max_val_batches:
            break
        raw_model = raw_model_fn()

        inputs_embeds = raw_model.prepare_inputs_embeds(
            input_ids=batch["input_ids"],
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
        )

        n_slow_imgs = batch["n_slow_images"]
        grid_thw = batch.get("image_grid_thw")
        B = inputs_embeds.shape[0]
        if grid_thw is not None and grid_thw.shape[0] > n_slow_imgs:
            merge = getattr(raw_model.visual, "spatial_merge_size", 2)
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
        pos_ids, _ = raw_model.get_rope_index(
            input_ids=batch["input_ids"],
            image_grid_thw=batch.get("image_grid_thw"),
            attention_mask=batch["attention_mask"],
        )
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

        full_embeds = torch.cat([
            slow_embeds_ext, fast_embeds, state_embeds, timesteps, noisy_actions,
        ], dim=1)
        n_fast = fast_embeds.shape[1]
        L_total = full_embeds.shape[1]

        outputs = model.model(
            inputs_embeds=full_embeds,
            position_ids=pos_ids,
            attention_mask=batch["attention_mask"],
            use_cache=False,
            output_hidden_states=use_flare and (flare_layer_idx != -1),
            latent_indexes=torch.arange(0, L_latent, device=full_embeds.device),
            action_indexes=torch.arange(L_latent, L_total, device=full_embeds.device),
            tactile_indexes=torch.arange(0, 0, device=full_embeds.device),
        )
        hidden = outputs.last_hidden_state

        act_pred_start = L_latent + n_fast + n_state + 1
        v_act = raw_model.final_layer(hidden[:, act_pred_start: act_pred_start + chunk, :])
        loss_act = nn.MSELoss()(v_act, target)

        loss_flare = torch.tensor(0.0, device=loss_act.device)
        if use_flare and batch["flare_pixel_values"] is not None:
            if flare_layer_idx == -1:
                flare_source = hidden
            else:
                all_hs = outputs.hidden_states
                n_layers = len(all_hs) - 1
                li = (n_layers + flare_layer_idx) if flare_layer_idx < 0 else flare_layer_idx
                li = max(0, min(li, n_layers - 1))
                flare_source = all_hs[li + 1]

            flare_hidden = flare_source[:, L_slow: L_slow + K, :]
            flare_pred = raw_model.flare_proj(flare_hidden)
            f_pv = batch["flare_pixel_values"].to(device=flare_pred.device, dtype=flare_pred.dtype)
            f_thw = batch["flare_grid_thw"].to(device=flare_pred.device)
            vit_out = raw_model.visual(f_pv, grid_thw=f_thw)
            features = vit_out[0] if isinstance(vit_out, (tuple, list)) else vit_out
            merge = getattr(raw_model.visual, "spatial_merge_size", 2)
            frame_feats = []
            offset = 0
            for g in f_thw:
                n_tok = int(g[0] * (g[1] // merge) * (g[2] // merge))
                pooled = F.adaptive_avg_pool1d(
                    features[offset: offset + n_tok].unsqueeze(0).permute(0, 2, 1),
                    T_per_frame).permute(0, 2, 1).squeeze(0)
                frame_feats.append(pooled)
                offset += n_tok
            flare_targets = torch.stack(frame_feats).view(B, K, -1)
            pred_norm = F.normalize(flare_pred, dim=-1)
            tgt_norm = F.normalize(flare_targets, dim=-1)
            loss_flare = (1.0 - (pred_norm * tgt_norm).sum(dim=-1)).mean()

        loss = loss_act + args.flare_loss_weight * loss_flare
        val_loss_sum += loss.item()
        val_act_sum += loss_act.item()
        val_flare_sum += loss_flare.item()
        n_val += 1

    # All-reduce across ranks
    for t in [val_loss_sum, val_act_sum, val_flare_sum]:
        if dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    ws = dist.get_world_size() if dist.is_initialized() else 1
    denom = ws * max(n_val, 1)

    model.train()
    return {
        "val/total_loss": val_loss_sum.item() / denom,
        "val/action_loss": val_act_sum.item() / denom,
        "val/flare_loss": val_flare_sum.item() / denom,
    }


# ───────────────────────────────────────────────────────────────────
#  Main training loop
# ───────────────────────────────────────────────────────────────────
def train(args):
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False, even_batches=True),
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=60))],
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        wandb.init(project=args.experiment_name, name=args.run_name,
                   config=args, dir=args.log_dir)

    accelerator.state.deepspeed_plugin.deepspeed_config[
        "train_micro_batch_size_per_gpu"] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config[
        "train_batch_size"] = (args.train_bsz_per_gpu
                               * (dist.get_world_size() if dist.is_initialized() else 1)
                               * accelerator.gradient_accumulation_steps)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    model = Qwen3VLVLAModel.from_pretrained_qwen3vl(
        pretrained_path=args.model_path,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
        use_tactile_deform=False,
        use_robot_state=bool(args.use_robot_state),
        torch_dtype=torch.bfloat16,
        n_flare_tokens_per_frame=args.n_flare_tokens_per_frame if args.use_flare else 0,
        n_flare_steps=args.n_flare_steps if args.use_flare else 0,
        flare_layer_index=args.flare_layer_index,
    )

    model.initialize_vla_weights()
    accelerator.print("VLA weights initialized.")

    T_per_frame = args.n_flare_tokens_per_frame
    S_steps = args.n_flare_steps
    K = T_per_frame * S_steps
    use_flare = bool(args.use_flare and K > 0)
    flare_layer_idx = args.flare_layer_index

    if use_flare:
        accelerator.print(
            f"Flare: {S_steps} steps x {T_per_frame} tok/frame = {K} total, "
            f"layer_index={flare_layer_idx}, stride={args.flare_frame_stride}")

    # Resolve resume paths (supports either a model.pt file or a ckpt directory).
    resume_model_pt, resume_ckpt_dir, resume_state_dir = _resolve_resume(args.resume_checkpoint)

    # Init action expert from latent expert
    if not args.resume_checkpoint:
        named_params = dict(model.named_parameters())
        for name, param in model.named_parameters():
            if "_action" in name:
                base = name.replace("_action", "")
                if base in named_params:
                    param.data.copy_(named_params[base].data)
        accelerator.print("Action expert initialized from latent expert.")

    # Old-style resume (weights-only): load model weights now, before accelerator.prepare.
    # New-style (state/ present): model weights are restored by accelerator.load_state
    # *after* prepare — skip here so the later load takes effect.
    if args.resume_checkpoint and resume_state_dir is None:
        resume_sd = torch.load(resume_model_pt, map_location="cpu")
        if "state_dict" in resume_sd:
            resume_sd = resume_sd["state_dict"]
        missing, unexpected = model.load_state_dict(resume_sd, strict=False)
        accelerator.print(f"Resumed weights (old-style) from {resume_model_pt}: "
                          f"missing={len(missing)}, unexpected={len(unexpected)}")

    # Freeze vision + tactile, train latent + action + flare
    for name, param in model.named_parameters():
        if name.startswith("visual") or name.startswith("deform_encoder"):
            param.requires_grad = False
        elif "_tactile" in name or "final_layer_tactile" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Total: {total/1e9:.2f}B  Trainable: {trainable/1e9:.2f}B "
                      f"({trainable/total*100:.1f}%)")

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

    if getattr(args, "data_format", "json") == "lerobot":
        from qwen_vla.lerobot_dataset import TRexLeRobotDataset
        dataset = TRexLeRobotDataset(args, processor, accelerator)
    else:
        dataset = EgoDexPretrainFlareDataset(args.data_root, args, processor, accelerator)

    # Train/val split
    val_dataloader = None
    if args.val_ratio > 0:
        val_dataset = dataset.create_val_split(val_ratio=args.val_ratio, seed=args.seed)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.train_bsz_per_gpu,
            shuffle=False,
            drop_last=True,
            collate_fn=val_dataset.collate_fn,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=True,
        )

    # Episode-grouped sampler: clusters frame indices by episode so that
    # consecutive __getitem__ calls hit the same episode.  Combined with
    # per-worker episode caching, this eliminates per-frame HDF5 open/close
    # and video seek — reducing per-sample I/O from ~200ms to ~0.1ms.
    sampler = EpisodeGroupedSampler(
        dataset, shuffle=True, seed=args.seed, drop_last=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_bsz_per_gpu,
        sampler=sampler,
        drop_last=True,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # IMPORTANT: Do NOT prepare the training dataloader with accelerate.
    # Our EpisodeGroupedSampler already shards indices by rank.  accelerate's
    # prepare() wraps the batch_sampler in BatchSamplerShard which shards
    # *batches* a second time → each rank sees 1/N^2 of the data (the
    # "2137 steps" bug).  We prepare model + optimizer only, and move train
    # batches to device manually.
    if val_dataloader is not None:
        model, optimizer, val_dataloader = accelerator.prepare(
            model, optimizer, val_dataloader)
    else:
        model, optimizer = accelerator.prepare(model, optimizer)

    # steps_per_epoch from ground truth
    world_size = accelerator.num_processes
    train_samples = len(dataset)
    samples_per_rank = train_samples // world_size
    steps_per_epoch = samples_per_rank // args.train_bsz_per_gpu
    num_training_steps = steps_per_epoch * args.n_epochs // accelerator.gradient_accumulation_steps
    accelerator.print(f"Dataset: {train_samples} training samples, "
                      f"world_size={world_size}")
    accelerator.print(f"Estimated {steps_per_epoch} steps/epoch "
                      f"(samples_per_rank={samples_per_rank}), "
                      f"{num_training_steps} total training steps, "
                      f"warmup={int(args.warmup_rates * num_training_steps)}")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_rates * num_training_steps),
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # ── Resume: restore step counter / LR scheduler / (optionally) optimizer ──
    # Priority for resume_global_step:
    #   1) new-style `state/` present → read from training_state.json
    #   2) user-supplied --resume_step > 0 (one-time override for legacy checkpoints)
    #   3) 0 (treat as weights-only init, like the original behavior)
    # Dir-name parsing is NOT used automatically because `checkpoint-0-125000`
    # may mean "weights-only init from some other run", not "resume step 125000".
    resume_global_step = 0
    used_full_state = False
    scheduler_restored = False
    if args.resume_checkpoint:
        if resume_state_dir is not None:
            accelerator.print(f"Loading full resumable state from {resume_state_dir}")
            # PyTorch 2.6 flipped torch.load's default to weights_only=True,
            # but older DeepSpeed serializes classes (LossScaler, etc.) that
            # aren't on the safe allowlist. Force weights_only=False for the
            # duration of this call — the checkpoint was written by us and is
            # trusted.
            _orig_torch_load = torch.load
            def _torch_load_trusted(*a, **kw):
                kw["weights_only"] = False
                return _orig_torch_load(*a, **kw)
            torch.load = _torch_load_trusted
            try:
                accelerator.load_state(resume_state_dir)
            finally:
                torch.load = _orig_torch_load
            used_full_state = True
            state_json = os.path.join(resume_ckpt_dir, "training_state.json")
            if os.path.isfile(state_json):
                with open(state_json) as f:
                    tstate = json.load(f)
                resume_global_step = int(tstate.get("global_step", 0))
                accelerator.print(
                    f"Full resume: global_step={resume_global_step} from training_state.json")
            else:
                _, parsed = _parse_step_from_ckpt_dir(resume_ckpt_dir)
                resume_global_step = parsed or 0
                accelerator.print(
                    f"Full resume: no training_state.json; global_step from dir name={resume_global_step}")

            # Restore LR scheduler from our explicit scheduler.bin (Accelerate
            # + DeepSpeed does not save it when prepared separately).
            sched_path = os.path.join(resume_state_dir, "scheduler.bin")
            if os.path.isfile(sched_path):
                sched_sd = torch.load(sched_path, map_location="cpu",
                                      weights_only=False)
                lr_scheduler.load_state_dict(sched_sd)
                scheduler_restored = True
                accelerator.print(
                    f"Restored LR scheduler from scheduler.bin: "
                    f"lr={lr_scheduler.get_last_lr()[0]:.3e}")
            else:
                accelerator.print(
                    "scheduler.bin not found in state/ (legacy ckpt). "
                    "Will fast-forward the LR scheduler instead.")

        if args.resume_step > 0:
            resume_global_step = args.resume_step
            accelerator.print(f"--resume_step override: global_step={resume_global_step}")

        # Fast-forward fallback: covers two cases —
        #   (a) weights-only resume (no state/), or
        #   (b) full resume from a legacy ckpt that has state/ but no
        #       scheduler.bin (e.g. all 0423 checkpoints).
        if resume_global_step > 0 and not scheduler_restored:
            note = ("optimizer momentum NOT restored"
                    if not used_full_state
                    else "optimizer state IS restored, only the LR curve is approximated")
            accelerator.print(
                f"[approximate resume] Fast-forwarding LR scheduler by "
                f"{resume_global_step} steps ({note}).")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress "step before optimizer step" warnings
                for _ in range(resume_global_step):
                    lr_scheduler.step()
            accelerator.print(
                f"[approximate resume] LR now = "
                f"{lr_scheduler.get_last_lr()[0]:.3e}")

    metric = TrainingMetrics(device=torch.cuda.current_device())
    global_step = resume_global_step
    model.train()

    device = accelerator.device

    # Derive starting epoch and within-epoch offset from global_step.
    start_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
    step_in_epoch_offset = global_step - start_epoch * steps_per_epoch
    if start_epoch >= args.n_epochs:
        accelerator.print(
            f"Resume global_step={global_step} is at/past end of training "
            f"({args.n_epochs} epochs × {steps_per_epoch} steps). Nothing to do.")
        return

    if start_epoch > 0 or step_in_epoch_offset > 0:
        accelerator.print(
            f"Resuming at epoch={start_epoch}, step_in_epoch={step_in_epoch_offset}, "
            f"global_step={global_step}. "
            f"{'Will skip already-seen batches at the sampler (no data load).' if args.resume_skip_data else 'Data iterator restarts from beginning of epoch (some overlap).'}")

    for epoch in range(start_epoch, args.n_epochs):
        sampler.set_epoch(epoch)

        # Data-skip for resume: drop the already-consumed portion of this
        # epoch at the *sampler* level so workers never load those samples.
        # Previously we iterated and `continue`d, which still ran image
        # loading + collate for every skipped batch — slow enough that GPUs
        # stayed at 0% and the NCCL watchdog could kill the job. Setting
        # sampler.start_index just slices the index list before workers ever
        # see it, so skipping ~100k batches is effectively instantaneous.
        skip_n = (step_in_epoch_offset
                  if (args.resume_skip_data and epoch == start_epoch) else 0)
        sampler.start_index = skip_n * args.train_bsz_per_gpu
        epoch_total = max(0, steps_per_epoch - skip_n)
        if skip_n > 0:
            accelerator.print(
                f"Skipping first {skip_n} batches via sampler offset "
                f"(no data loaded). Epoch will run {epoch_total} steps.")

        from tqdm import tqdm
        desc = (f"Epoch {epoch} (resumed +{skip_n})"
                if skip_n > 0 else f"Epoch {epoch}")
        it = (tqdm(dataloader, total=epoch_total, desc=desc)
              if accelerator.is_main_process else dataloader)

        for batch in it:
            # Move batch to device (since dataloader is not prepared by accelerate)
            batch = {k: v.to(device) if torch.is_tensor(v) else v
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
                                getattr(dataset.processor.image_processor, "merge_size", 2))
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

            pos_ids, _ = raw_model.get_rope_index(
                input_ids=batch["input_ids"],
                image_grid_thw=batch.get("image_grid_thw"),
                attention_mask=batch["attention_mask"],
            )
            pos_ids = pos_ids[:, :, :L_slow]

            # ── Flare: append query tokens to latent expert ──
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
                state_embeds = torch.empty(
                    (B, 0, slow_embeds.shape[2]),
                    device=slow_embeds.device, dtype=slow_embeds.dtype)
            n_state = state_embeds.shape[1]

            noisy_actions = raw_model.x_embedder(
                batch["noisy_actions"].to(slow_embeds.dtype))
            timesteps = raw_model.t_embedder(
                batch["timesteps"].to(slow_embeds.dtype)).unsqueeze(1)

            chunk = args.action_chunk
            target = batch["target"].to(slow_embeds.dtype)

            # Stage 1: latent + action expert only (no tactile)
            full_embeds = torch.cat([
                slow_embeds_ext,
                fast_embeds, state_embeds, timesteps, noisy_actions,
            ], dim=1)

            n_fast = fast_embeds.shape[1]
            n_action = n_fast + n_state + 1 + chunk
            L_total = full_embeds.shape[1]
            latent_indexes = torch.arange(0, L_latent, device=full_embeds.device)
            action_indexes = torch.arange(L_latent, L_total, device=full_embeds.device)
            tactile_indexes = torch.arange(0, 0, device=full_embeds.device)

            if global_step == 0 and accelerator.is_main_process:
                print(f"\n[Layout] slow={L_slow} flare={K} fast={n_fast} "
                      f"state={n_state} t=1 act={chunk} total={L_total}")

            outputs = model.model(
                inputs_embeds=full_embeds,
                position_ids=pos_ids,
                attention_mask=batch["attention_mask"],
                use_cache=False,
                output_hidden_states=use_flare and (flare_layer_idx != -1),
                latent_indexes=latent_indexes,
                action_indexes=action_indexes,
                tactile_indexes=tactile_indexes,
            )
            hidden = outputs.last_hidden_state

            act_pred_start = L_latent + n_fast + n_state + 1
            v_act = raw_model.final_layer(
                hidden[:, act_pred_start: act_pred_start + chunk, :])
            loss_act = nn.MSELoss()(v_act, target)

            # ── Flare loss ──
            loss_flare = 0.0
            if use_flare and batch["flare_pixel_values"] is not None:
                if flare_layer_idx == -1:
                    flare_source = hidden
                else:
                    all_hs = outputs.hidden_states
                    n_layers = len(all_hs) - 1
                    li = (n_layers + flare_layer_idx) if flare_layer_idx < 0 else flare_layer_idx
                    li = max(0, min(li, n_layers - 1))
                    flare_source = all_hs[li + 1]

                flare_hidden = flare_source[:, L_slow: L_slow + K, :]
                flare_pred = raw_model.flare_proj(flare_hidden)

                f_pv = batch["flare_pixel_values"].to(device=flare_pred.device, dtype=flare_pred.dtype)
                f_thw = batch["flare_grid_thw"].to(device=flare_pred.device)

                with torch.no_grad():
                    vit_out = raw_model.visual(f_pv, grid_thw=f_thw)
                    features = vit_out[0] if isinstance(vit_out, (tuple, list)) else vit_out

                    merge = getattr(raw_model.visual, "spatial_merge_size", 2)
                    frame_feats = []
                    offset = 0
                    for g in f_thw:
                        n_tok = int(g[0] * (g[1] // merge) * (g[2] // merge))
                        frame_tokens = features[offset: offset + n_tok]
                        pooled = F.adaptive_avg_pool1d(
                            frame_tokens.unsqueeze(0).permute(0, 2, 1),
                            T_per_frame,
                        ).permute(0, 2, 1).squeeze(0)
                        frame_feats.append(pooled)
                        offset += n_tok
                    flare_targets = torch.stack(frame_feats).view(B, K, -1)

                pred_norm = F.normalize(flare_pred, dim=-1)
                tgt_norm = F.normalize(flare_targets.detach(), dim=-1)
                loss_flare = (1.0 - (pred_norm * tgt_norm).sum(dim=-1)).mean()

            loss = loss_act
            if torch.is_tensor(loss_flare):
                loss = loss + args.flare_loss_weight * loss_flare

            metric.update(loss, loss_act, loss_flare)
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
                                       flare=f"{m['flare_loss']:.6f}",
                                       lr=f"{lr_now:.2e}")
                    wandb.log({
                        "total_loss": m["total_loss"],
                        "action_loss": m["action_loss"],
                        "flare_loss": m["flare_loss"],
                        "lr": lr_now,
                        "epoch": epoch,
                    }, step=global_step)

            # ── Validation ──
            if (val_dataloader is not None
                    and args.val_freq > 0
                    and (global_step + 1) % args.val_freq == 0):
                val_m = run_validation(
                    model, val_dataloader, accelerator,
                    lambda: accelerator.unwrap_model(model),
                    use_flare, K, T_per_frame, flare_layer_idx, args)
                if accelerator.is_main_process:
                    accelerator.print(
                        f"  [Val step={global_step}] "
                        f"loss={val_m['val/total_loss']:.6f} "
                        f"act={val_m['val/action_loss']:.6f} "
                        f"flare={val_m['val/flare_loss']:.6f}")
                    wandb.log(val_m, step=global_step)

            # ── Step-level checkpoint ──
            if (args.save_steps > 0
                    and (global_step + 1) % args.save_steps == 0):
                accelerator.wait_for_everyone()
                save_checkpoint(model, processor, accelerator, args,
                                epoch, global_step + 1, dataset,
                                save_full_state=bool(args.save_full_state),
                                lr_scheduler=lr_scheduler)

            global_step += 1

        if (epoch + 1) % args.save_freq == 0 or epoch == args.n_epochs - 1:
            accelerator.wait_for_everyone()
            save_checkpoint(model, processor, accelerator, args,
                            epoch, global_step, dataset,
                            save_full_state=bool(args.save_full_state),
                            lr_scheduler=lr_scheduler)

    accelerator.print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="qwen3vl_egodex_pretrain_flare")
    parser.add_argument("--run_name", type=str, default="run_1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="",
                        help="EgoDex episode root (required for --data_format json).")
    parser.add_argument("--data_format", type=str, default="json", choices=["json", "lerobot"])
    parser.add_argument("--lerobot_root", type=str, default="",
                        help="LeRobot dataset dir (required when --data_format lerobot).")
    parser.add_argument("--lerobot_repo_id", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--max_ckpts", type=int, default=5)

    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=2)
    parser.add_argument("--train_bsz_per_gpu", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.0)
    parser.add_argument("--warmup_rates", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--action_dim", type=int, default=62)
    parser.add_argument("--action_chunk", type=int, default=16)
    parser.add_argument("--use_robot_state", type=int, default=0)
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("W", "H"))

    parser.add_argument("--resume_checkpoint", type=str, default="",
                        help="Path to a model.pt file or to a checkpoint dir. "
                             "If the dir contains a `state/` subfolder, full "
                             "optimizer+scheduler+RNG state is restored.")
    parser.add_argument("--resume_step", type=int, default=0,
                        help="Explicit step to resume at (overrides any saved "
                             "value). Use this with a weights-only checkpoint "
                             "(e.g. the legacy checkpoint-0-30000) to fast-"
                             "forward the LR scheduler and wandb step counter.")
    parser.add_argument("--resume_skip_data", type=int, default=0,
                        help="If 1, fast-forward the dataloader past the "
                             "already-consumed portion of the first epoch on "
                             "resume. If 0, data iteration restarts from the "
                             "beginning of the epoch (some overlap).")
    parser.add_argument("--save_full_state", type=int, default=1,
                        help="If 1 (default), every checkpoint also writes a "
                             "`state/` subdir with optimizer+scheduler+RNG "
                             "state for exact resume. Set to 0 to save disk.")
    parser.add_argument("--save_steps", type=int, default=0,
                        help="Save checkpoint every N steps (0=disable, epoch-only)")

    # Validation
    parser.add_argument("--val_ratio", type=float, default=0.02,
                        help="Fraction of episodes to hold out for validation (0=disable)")
    parser.add_argument("--val_freq", type=int, default=1000,
                        help="Run validation every N training steps (0=disable)")
    parser.add_argument("--max_val_batches", type=int, default=50,
                        help="Max batches per validation run")

    # Flare
    parser.add_argument("--use_flare", type=int, default=1)
    parser.add_argument("--n_flare_tokens_per_frame", type=int, default=4)
    parser.add_argument("--n_flare_steps", type=int, default=8)
    parser.add_argument("--flare_loss_weight", type=float, default=0.5)
    parser.add_argument("--flare_frame_stride", type=int, default=4)
    parser.add_argument("--flare_layer_index", type=int, default=-1)

    args = parser.parse_args()

    if args.data_format == "lerobot":
        if not args.lerobot_root:
            parser.error("--data_format lerobot requires --lerobot_root")
    elif not args.data_root:
        parser.error("--data_format json requires --data_root")

    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name, args.run_name)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)
