"""
EgoDex pretraining with Scene FLARE + Tactile FLARE.

Two auxiliary prediction tasks for the latent expert:
  1. Scene FLARE (existing): predict future full-frame ViT features → scene dynamics
  2. Tactile FLARE (new):    predict current fingertip-crop ViT features → contact awareness

Data assumptions:
  - Each episode has ego_view.mp4, pretrain.hdf5, metadata.json (standard)
  - Each EgoDex episode has fingertip_coords.npy [T, 10, 2] precomputed by
    precompute_fingertip_coords.py (10 = left 5 + right 5 fingertip pixel coords)
  - Episodes without fingertip_coords.npy still train normally; tactile flare
    loss is simply skipped for those samples.

When --use_tactile_flare 0, behaves identically to pretrain_egodex_flare.
When --use_scene_flare 0 AND --use_tactile_flare 0, behaves identically to base pretrain.
"""

import os, sys

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import glob
import json
import math
import shutil
import logging
import argparse
from datetime import timedelta

import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import wandb
import PIL.Image

from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
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


# ───────────────────────────────────────────────────────────────────
#  Dataset
# ───────────────────────────────────────────────────────────────────

class EgoDexTactileFlareDataset(Dataset):
    """
    Map-style dataset with flat (episode_idx, frame_t) index.

    Returns per sample:
      - frame:            current RGB frame (np.ndarray)
      - state:            robot state (np.ndarray)
      - action_chunk:     action chunk (np.ndarray)
      - language:         task description (str)
      - flare_frames:     list of S future RGB frames (for scene flare)
      - fingertip_coords: (10, 2) pixel coords or None (for tactile flare)
    """

    def __init__(self, data_root: str, config, processor, accelerator):
        super().__init__()
        self.config = config
        self.processor = processor
        self.accelerator = accelerator

        self.episodes = []
        all_action_q01, all_action_q99 = [], []
        all_state_q01, all_state_q99 = [], []
        all_te_mean, all_te_std = [], []

        manifest_paths = sorted(
            glob.glob(os.path.join(data_root, "*", "pretrain_manifest.json")))
        if not manifest_paths:
            raise FileNotFoundError(
                f"No pretrain_manifest.json found under {data_root}/*/")

        for mp in manifest_paths:
            with open(mp, "r") as f:
                manifest = json.load(f)
            self.episodes.extend(manifest["episodes"])
            stats = manifest.get("statistics", {})
            if "action" in stats and "state" in stats:
                all_action_q01.append(np.array(stats["action"]["q01"], dtype=np.float32))
                all_action_q99.append(np.array(stats["action"]["q99"], dtype=np.float32))
                all_state_q01.append(np.array(stats["state"]["q01"], dtype=np.float32))
                all_state_q99.append(np.array(stats["state"]["q99"], dtype=np.float32))
            if "tracking_error" in stats:
                all_te_mean.append(np.array(stats["tracking_error"]["mean"], dtype=np.float32))
                all_te_std.append(np.array(stats["tracking_error"]["std"], dtype=np.float32))

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

        # Scene flare config
        self.use_scene_flare = bool(getattr(config, "use_scene_flare", 0))
        self.scene_flare_steps = getattr(config, "scene_flare_steps", 0) if self.use_scene_flare else 0
        self.scene_flare_stride = getattr(config, "scene_flare_frame_stride", 4)

        # Tactile flare config
        self.use_tactile_flare = bool(getattr(config, "use_tactile_flare", 0))
        self.tactile_crop_size = getattr(config, "tactile_flare_crop_size", 96)

        # Cache for fingertip coords (lightweight npy, lazy-loaded per episode)
        self._ftip_coords_cache = {}  # ep_idx -> np.ndarray [T, 10, 2] or None

        # Build flat index
        self._index = []
        for ep_idx, ep_info in enumerate(self.episodes):
            for t in range(ep_info["num_frames"]):
                self._index.append((ep_idx, t))

        accelerator.print(
            f"Dataset: {len(self.episodes)} episodes, {len(self._index)} transitions\n"
            f"  Scene flare: {self.scene_flare_steps} steps x stride {self.scene_flare_stride}\n"
            f"  Tactile flare: crop={self.tactile_crop_size}px, enabled={self.use_tactile_flare}")

    def _get_fingertip_coords(self, ep_idx, frame_t):
        """Load precomputed fingertip 2D coords. Returns (10, 2) or None."""
        if ep_idx not in self._ftip_coords_cache:
            ep_dir = self.episodes[ep_idx]["episode_dir"]
            npy_path = os.path.join(ep_dir, "fingertip_coords.npy")
            if os.path.isfile(npy_path):
                self._ftip_coords_cache[ep_idx] = np.load(npy_path)
            else:
                self._ftip_coords_cache[ep_idx] = None
        arr = self._ftip_coords_cache[ep_idx]
        if arr is None:
            return None
        t = min(frame_t, arr.shape[0] - 1)
        return arr[t]

    @staticmethod
    def _crop_fingertips(frame, coords, crop_size):
        """
        Crop 10 fingertip patches from an already-loaded frame.
        frame: (H, W, 3) uint8, coords: (10, 2) pixel coords.
        Returns (10, crop_size, crop_size, 3) uint8.
        """
        img_h, img_w = frame.shape[:2]
        half = crop_size // 2
        crops = np.zeros((10, crop_size, crop_size, 3), dtype=np.uint8)
        for i in range(10):
            cx, cy = int(coords[i, 0]), int(coords[i, 1])
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(img_w, cx + half), min(img_h, cy + half)
            if x2 > x1 and y2 > y1:
                patch = frame[y1:y2, x1:x2]
                crops[i, :patch.shape[0], :patch.shape[1]] = patch
        return crops

    def __len__(self):
        return len(self._index)

    def create_val_split(self, val_ratio=0.02, seed=42):
        """Split episodes into train/val. Returns val dataset."""
        import copy
        rng = np.random.RandomState(seed)
        n_ep = len(self.episodes)
        n_val = max(1, int(n_ep * val_ratio))
        perm = rng.permutation(n_ep)
        val_ep_set = set(perm[:n_val].tolist())
        train_ep_set = set(perm[n_val:].tolist())

        val_ds = copy.copy(self)
        val_ds.is_val = True
        val_ds._fingertip_cache = {}
        val_ds._index = [(ei, t) for ei, t in self._index if ei in val_ep_set]

        self._index = [(ei, t) for ei, t in self._index if ei in train_ep_set]
        self.accelerator.print(
            f"Train/Val: {len(train_ep_set)} train eps ({len(self._index)} frames), "
            f"{len(val_ep_set)} val eps ({len(val_ds._index)} frames)")
        return val_ds

    @staticmethod
    def _normalize(values, mask, vmin, vmax):
        return np.where(
            mask,
            np.clip(2 * (values - vmin) / (vmax - vmin + 1e-8) - 1, -1, 1),
            values,
        )

    @staticmethod
    def _find_head_video(ep_dir):
        candidate = os.path.join(ep_dir, "ego_view.mp4")
        if os.path.isfile(candidate):
            return candidate
        matches = glob.glob(os.path.join(ep_dir, "*head*.mp4"))
        return matches[0] if matches else None

    def _read_video_frames(self, video_path, frame_indices):
        """Read frames by seeking once then decoding forward sequentially."""
        if not frame_indices:
            return []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return [None] * len(frame_indices)
            need_set = set(frame_indices)
            collected = {}
            first, last = min(frame_indices), max(frame_indices)
            if last - first > 200:
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        collected[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, first)
                pos = first
                while pos <= last:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if pos in need_set:
                        collected[pos] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pos += 1
            cap.release()
            return [collected.get(idx) for idx in frame_indices]
        except Exception:
            return [None] * len(frame_indices)

    def __getitem__(self, idx: int) -> Dict:
        ep_idx, frame_t = self._index[idx]
        ep_info = self.episodes[ep_idx]
        ep_dir = ep_info["episode_dir"]
        num_frames = ep_info["num_frames"]
        pretrain_h5 = os.path.join(ep_dir, "pretrain.hdf5")
        video_path = self._find_head_video(ep_dir)

        fallback_frame = np.zeros((288, 384, 3), dtype=np.uint8)
        S = self.scene_flare_steps
        fallback = {
            "frame": fallback_frame,
            "state": np.zeros(self.config.action_dim, dtype=np.float32),
            "action_chunk": np.zeros((self.config.action_chunk, self.config.action_dim),
                                      dtype=np.float32),
            "language": "",
            "flare_frames": [fallback_frame.copy() for _ in range(S)],
            "fingertip_coords": None,
        }

        if video_path is None or not os.path.isfile(pretrain_h5):
            return fallback

        try:
            with h5py.File(pretrain_h5, "r") as f:
                state = f["states"][frame_t]
                action_chunk = f["action_chunks"][frame_t]
                language = f.attrs.get("language", "")
        except Exception:
            return fallback

        # Build frame indices: current + future (for scene flare)
        frame_indices = [frame_t]
        for k in range(S):
            frame_indices.append(min(frame_t + (k + 1) * self.scene_flare_stride,
                                     num_frames - 1))

        all_frames = self._read_video_frames(video_path, frame_indices)
        current_frame = all_frames[0] if all_frames[0] is not None else fallback_frame
        flare_frames = []
        for k in range(S):
            ff = all_frames[k + 1]
            flare_frames.append(ff if ff is not None else current_frame.copy())

        # Fingertip coords for tactile flare (crops are cut in collate_fn from frame)
        ftip_coords = None
        if self.use_tactile_flare:
            ftip_coords = self._get_fingertip_coords(ep_idx, frame_t)

        return {
            "frame": current_frame,
            "state": state,
            "action_chunk": action_chunk,
            "language": language,
            "flare_frames": flare_frames,
            "fingertip_coords": ftip_coords,
        }

    def collate_fn(self, batch: List[Dict]) -> Dict:
        cfg = self.config
        B = len(batch)

        # ── Actions + flow matching ──
        actions = np.stack([x["action_chunk"] for x in batch])
        norm_actions = self._normalize(actions, self.action_mask,
                                        self.action_min, self.action_max)
        norm_actions = torch.tensor(norm_actions, dtype=torch.bfloat16)

        d = torch.distributions.Beta(
            torch.tensor(1.5, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32))
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

        # ── VLM input: [slow_img | text | fast_img] ──
        all_input_ids, all_pixel_values, all_grid_thw = [], [], []
        n_slow_images = 1

        for x in batch:
            img = PIL.Image.fromarray(x["frame"])
            if self.image_size is not None:
                img = img.resize(self.image_size, PIL.Image.LANCZOS)
            content = [{"type": "image"},
                       {"type": "text", "text": x.get("language", "")},
                       {"type": "image"}]
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inp = self.processor(text=text, images=[img, img],
                                 return_tensors="pt", padding=False)
            all_input_ids.append(inp.input_ids[0])
            if "pixel_values" in inp and inp.pixel_values is not None:
                all_pixel_values.append(inp.pixel_values)
                all_grid_thw.append(inp.image_grid_thw)

        # ── Scene flare future frames ──
        scene_flare_pv, scene_flare_thw = None, None
        if self.scene_flare_steps > 0:
            flare_pil = []
            for x in batch:
                for ff in x["flare_frames"]:
                    pil_img = PIL.Image.fromarray(ff)
                    if self.image_size is not None:
                        pil_img = pil_img.resize(self.image_size, PIL.Image.LANCZOS)
                    flare_pil.append(pil_img)
            flare_inp = self.processor.image_processor(flare_pil, return_tensors="pt")
            scene_flare_pv = flare_inp.pixel_values.to(torch.bfloat16)
            scene_flare_thw = flare_inp.image_grid_thw

        # ── Tactile flare: crop fingertips from already-loaded frame ──
        tactile_flare_pv, tactile_flare_thw = None, None
        has_tactile_flare = []

        if self.use_tactile_flare:
            crop_size = self.tactile_crop_size
            crop_pil = []
            for x in batch:
                coords = x["fingertip_coords"]  # (10, 2) or None
                if coords is None:
                    has_tactile_flare.append(False)
                    for _ in range(10):
                        crop_pil.append(PIL.Image.new("RGB", (crop_size, crop_size)))
                else:
                    has_tactile_flare.append(True)
                    # Crop from the frame already in memory — no extra I/O
                    crops = self._crop_fingertips(x["frame"], coords, crop_size)
                    for f_i in range(10):
                        crop_pil.append(PIL.Image.fromarray(crops[f_i]))

            if crop_pil:
                crop_inp = self.processor.image_processor(crop_pil, return_tensors="pt")
                tactile_flare_pv = crop_inp.pixel_values.to(torch.bfloat16)
                tactile_flare_thw = crop_inp.image_grid_thw

        # ── Padding ──
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

        return {
            "input_ids": torch.stack(padded_ids),
            "attention_mask": torch.stack(attention_ms),
            "pixel_values": torch.cat(all_pixel_values, dim=0) if all_pixel_values else None,
            "image_grid_thw": torch.cat(all_grid_thw, dim=0) if all_grid_thw else None,
            "n_slow_images": n_slow_images,
            "noisy_actions": x_t,
            "target": u_t,
            "timesteps": time,
            "state_raw": torch.stack(state_raw_list) if state_raw_list else None,
            # Scene flare
            "scene_flare_pixel_values": scene_flare_pv,
            "scene_flare_grid_thw": scene_flare_thw,
            # Tactile flare
            "tactile_flare_pixel_values": tactile_flare_pv,
            "tactile_flare_grid_thw": tactile_flare_thw,
            "has_tactile_flare": has_tactile_flare,
        }


# ───────────────────────────────────────────────────────────────────
#  Checkpoint / Metrics
# ───────────────────────────────────────────────────────────────────

def save_checkpoint(model, processor, accelerator, args, epoch, global_step, dataset):
    save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
    if accelerator.is_main_process:
        ckpts = [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint-")]
        if args.max_ckpts > 0 and len(ckpts) >= args.max_ckpts:
            oldest = min(ckpts, key=lambda f: os.path.getctime(
                os.path.join(args.output_dir, f)))
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
                "action": {"mask": dataset.action_mask.tolist(),
                           "q01": dataset.action_min.tolist(),
                           "q99": dataset.action_max.tolist()},
                "state":  {"mask": dataset.state_mask.tolist(),
                           "q01": dataset.state_min.tolist(),
                           "q99": dataset.state_max.tolist()},
            }}, f, indent=2)
        with open(os.path.join(save_dir, "training_args.json"), "w") as f:
            json.dump({
                "model_path": args.model_path,
                "action_dim": args.action_dim,
                "action_chunk": args.action_chunk,
                "use_robot_state": args.use_robot_state,
                "training_stage": 1,
                "use_scene_flare": args.use_scene_flare,
                "n_flare_tokens_per_frame": args.scene_flare_tokens_per_frame,
                "n_flare_steps": args.scene_flare_steps,
                "flare_layer_index": args.flare_layer_index,
                "use_tactile_flare": args.use_tactile_flare,
                "n_tactile_flare_tokens": 10,
                "tactile_flare_crop_size": args.tactile_flare_crop_size,
            }, f, indent=2)
    accelerator.wait_for_everyone()


class TrainingMetrics:
    def __init__(self, device):
        self.n_step = 0
        self.total_loss = torch.tensor(0.0, device=device)
        self.action_loss = torch.tensor(0.0, device=device)
        self.scene_flare_loss = torch.tensor(0.0, device=device)
        self.tactile_flare_loss = torch.tensor(0.0, device=device)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    def update(self, total, action, scene_flare=0.0, tactile_flare=0.0):
        self.n_step += 1
        for attr, val in [("total_loss", total), ("action_loss", action),
                          ("scene_flare_loss", scene_flare),
                          ("tactile_flare_loss", tactile_flare)]:
            v = val.item() if torch.is_tensor(val) else val
            getattr(self, attr).add_(v)

    def get_metric(self, reset=True):
        tensors = [self.total_loss, self.action_loss,
                   self.scene_flare_loss, self.tactile_flare_loss]
        if dist.is_initialized():
            for t in tensors:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
        denom = self.world_size * max(self.n_step, 1)
        m = {
            "total_loss": self.total_loss.item() / denom,
            "action_loss": self.action_loss.item() / denom,
            "scene_flare_loss": self.scene_flare_loss.item() / denom,
            "tactile_flare_loss": self.tactile_flare_loss.item() / denom,
        }
        if reset:
            self.n_step = 0
            for t in tensors:
                t.fill_(0)
        return m


# ───────────────────────────────────────────────────────────────────
#  Flare loss computation (shared between train and val)
# ───────────────────────────────────────────────────────────────────

def compute_scene_flare_loss(raw_model, hidden, outputs, batch, B, L_slow, K_scene,
                              T_per_frame, flare_layer_idx):
    """Compute cosine-similarity loss for scene FLARE tokens."""
    if K_scene == 0 or batch["scene_flare_pixel_values"] is None:
        return 0.0

    if flare_layer_idx == -1:
        flare_source = hidden
    else:
        all_hs = outputs.hidden_states
        n_layers = len(all_hs) - 1
        li = (n_layers + flare_layer_idx) if flare_layer_idx < 0 else flare_layer_idx
        li = max(0, min(li, n_layers - 1))
        flare_source = all_hs[li + 1]

    flare_hidden = flare_source[:, L_slow: L_slow + K_scene, :]
    flare_pred = raw_model.flare_proj(flare_hidden)

    f_pv = batch["scene_flare_pixel_values"].to(device=flare_pred.device, dtype=flare_pred.dtype)
    f_thw = batch["scene_flare_grid_thw"].to(device=flare_pred.device)

    with torch.no_grad():
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
        flare_targets = torch.stack(frame_feats).view(B, K_scene, -1)

    pred_norm = F.normalize(flare_pred, dim=-1)
    tgt_norm = F.normalize(flare_targets.detach(), dim=-1)
    return (1.0 - (pred_norm * tgt_norm).sum(dim=-1)).mean()


def compute_tactile_flare_loss(raw_model, hidden, outputs, batch, B, L_slow,
                                K_scene, K_tactile, flare_layer_idx):
    """Compute cosine-similarity loss for tactile FLARE tokens (fingertip crops)."""
    if K_tactile == 0 or batch["tactile_flare_pixel_values"] is None:
        return 0.0

    # Check which samples have valid fingertip data
    has_ftip = batch["has_tactile_flare"]
    if not any(has_ftip):
        return 0.0

    # Tactile flare tokens sit after scene flare tokens
    tac_start = L_slow + K_scene
    if flare_layer_idx == -1:
        flare_source = hidden
    else:
        all_hs = outputs.hidden_states
        n_layers = len(all_hs) - 1
        li = (n_layers + flare_layer_idx) if flare_layer_idx < 0 else flare_layer_idx
        li = max(0, min(li, n_layers - 1))
        flare_source = all_hs[li + 1]

    tac_hidden = flare_source[:, tac_start: tac_start + K_tactile, :]
    tac_pred = raw_model.tactile_flare_proj(tac_hidden)  # [B, 10, H]

    # Encode fingertip crops through frozen ViT
    f_pv = batch["tactile_flare_pixel_values"].to(device=tac_pred.device, dtype=tac_pred.dtype)
    f_thw = batch["tactile_flare_grid_thw"].to(device=tac_pred.device)

    with torch.no_grad():
        vit_out = raw_model.visual(f_pv, grid_thw=f_thw)
        features = vit_out[0] if isinstance(vit_out, (tuple, list)) else vit_out

        # Global average pool each crop to 1 token
        merge = getattr(raw_model.visual, "spatial_merge_size", 2)
        crop_feats = []
        offset = 0
        for g in f_thw:
            n_tok = int(g[0] * (g[1] // merge) * (g[2] // merge))
            crop_feats.append(features[offset: offset + n_tok].mean(dim=0))
            offset += n_tok
        # crop_feats: B*10 vectors → [B, 10, H]
        tac_targets = torch.stack(crop_feats).view(B, K_tactile, -1)

    # Mask out samples without fingertip data
    pred_norm = F.normalize(tac_pred, dim=-1)
    tgt_norm = F.normalize(tac_targets.detach(), dim=-1)
    cos_sim = (pred_norm * tgt_norm).sum(dim=-1)  # [B, 10]

    # Build mask [B, 10]
    mask = torch.zeros(B, K_tactile, device=cos_sim.device, dtype=cos_sim.dtype)
    for i, has in enumerate(has_ftip):
        if has:
            mask[i] = 1.0

    if mask.sum() == 0:
        return 0.0
    loss = ((1.0 - cos_sim) * mask).sum() / mask.sum()
    return loss


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

    # ── Flare token counts ──
    T_per_frame = args.scene_flare_tokens_per_frame
    S_steps = args.scene_flare_steps
    K_scene = (T_per_frame * S_steps) if args.use_scene_flare else 0
    K_tactile = 10 if args.use_tactile_flare else 0  # 1 token per fingertip
    K_total = K_scene + K_tactile
    flare_layer_idx = args.flare_layer_index

    model = Qwen3VLVLAModel.from_pretrained_qwen3vl(
        pretrained_path=args.model_path,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
        use_tactile_deform=False,
        use_robot_state=bool(args.use_robot_state),
        torch_dtype=torch.bfloat16,
        n_flare_tokens_per_frame=T_per_frame if args.use_scene_flare else 0,
        n_flare_steps=S_steps if args.use_scene_flare else 0,
        flare_layer_index=flare_layer_idx,
    )

    # Add tactile flare components to the model
    H = model.config.hidden_size
    if K_tactile > 0:
        model.tactile_flare_queries = nn.Parameter(
            torch.randn(1, K_tactile, H) * 0.02)
        model.tactile_flare_proj = nn.Sequential(
            nn.Linear(H, H), nn.GELU(), nn.Linear(H, H))

    model.initialize_vla_weights()

    # Also xavier-init tactile flare proj
    if K_tactile > 0:
        for mm in model.tactile_flare_proj.modules():
            if isinstance(mm, nn.Linear):
                nn.init.xavier_uniform_(mm.weight)
                if mm.bias is not None:
                    nn.init.zeros_(mm.bias)

    accelerator.print(
        f"FLARE: scene={K_scene} tokens, tactile={K_tactile} tokens, "
        f"total={K_total}, layer={flare_layer_idx}")

    # Init action expert from latent expert
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
        model_sd = model.state_dict()
        filtered_sd = {k: v for k, v in resume_sd.items()
                       if k not in model_sd or model_sd[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
        accelerator.print(f"Resumed: missing={len(missing)}, unexpected={len(unexpected)}")

    # Freeze vision + tactile expert, train latent + action + flare
    for name, param in model.named_parameters():
        if name.startswith("visual") or name.startswith("deform_encoder"):
            param.requires_grad = False
        elif "_tactile" in name and "flare" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Total: {total/1e9:.2f}B  Trainable: {trainable/1e9:.2f}B")

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

    dataset = EgoDexTactileFlareDataset(args.data_root, args, processor, accelerator)

    val_dataloader = None
    if args.val_ratio > 0:
        val_dataset = dataset.create_val_split(val_ratio=args.val_ratio, seed=args.seed)
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.train_bsz_per_gpu, shuffle=False,
            drop_last=True, collate_fn=val_dataset.collate_fn,
            num_workers=max(1, args.num_workers // 2), pin_memory=True)

    dataloader = DataLoader(
        dataset, batch_size=args.train_bsz_per_gpu, shuffle=True, drop_last=True,
        collate_fn=dataset.collate_fn, num_workers=args.num_workers, pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False)

    if val_dataloader is not None:
        model, optimizer, dataloader, val_dataloader = accelerator.prepare(
            model, optimizer, dataloader, val_dataloader)
    else:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    steps_per_epoch = len(dataloader)
    num_training_steps = steps_per_epoch * args.n_epochs // accelerator.gradient_accumulation_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_rates * num_training_steps),
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio)
    lr_scheduler = accelerator.prepare(lr_scheduler)

    accelerator.print(f"{steps_per_epoch} steps/epoch, {num_training_steps} total")

    metric = TrainingMetrics(device=torch.cuda.current_device())
    global_step = 0
    use_scene = bool(args.use_scene_flare and K_scene > 0)
    use_tactile = bool(args.use_tactile_flare and K_tactile > 0)
    model.train()

    for epoch in range(args.n_epochs):
        if hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch)

        from tqdm import tqdm
        it = (tqdm(dataloader, total=steps_per_epoch, desc=f"Epoch {epoch}")
              if accelerator.is_main_process else dataloader)

        for batch in it:
            raw_model = accelerator.unwrap_model(model)

            # ── Embed inputs ──
            inputs_embeds = raw_model.prepare_inputs_embeds(
                input_ids=batch["input_ids"],
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"))

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
                attention_mask=batch["attention_mask"])
            pos_ids = pos_ids[:, :, :L_slow]

            # ── Append FLARE query tokens to latent expert ──
            parts = [slow_embeds]
            if use_scene:
                scene_q = raw_model.flare_queries.expand(B, -1, -1).to(
                    device=slow_embeds.device, dtype=slow_embeds.dtype)
                parts.append(scene_q)
            if use_tactile:
                tac_q = raw_model.tactile_flare_queries.expand(B, -1, -1).to(
                    device=slow_embeds.device, dtype=slow_embeds.dtype)
                parts.append(tac_q)
            slow_embeds_ext = torch.cat(parts, dim=1)
            pos_ids = extend_position_ids_for_flare(pos_ids, K_total)
            L_latent = slow_embeds_ext.shape[1]

            # ── Action tokens ──
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

            full_embeds = torch.cat([
                slow_embeds_ext, fast_embeds, state_embeds, timesteps, noisy_actions,
            ], dim=1)
            L_total = full_embeds.shape[1]

            if global_step == 0 and accelerator.is_main_process:
                print(f"\n[Layout] slow={L_slow} scene_flare={K_scene} "
                      f"tac_flare={K_tactile} fast={n_fast} state={n_state} "
                      f"t=1 act={chunk} total={L_total}")

            outputs = model.model(
                inputs_embeds=full_embeds,
                position_ids=pos_ids,
                attention_mask=batch["attention_mask"],
                use_cache=False,
                output_hidden_states=(K_total > 0 and flare_layer_idx != -1),
                latent_indexes=torch.arange(0, L_latent, device=full_embeds.device),
                action_indexes=torch.arange(L_latent, L_total, device=full_embeds.device),
                tactile_indexes=torch.arange(0, 0, device=full_embeds.device),
            )
            hidden = outputs.last_hidden_state

            # ── Action loss ──
            act_start = L_latent + n_fast + n_state + 1
            v_act = raw_model.final_layer(hidden[:, act_start: act_start + chunk, :])
            loss_act = nn.MSELoss()(v_act, target)

            # ── Scene FLARE loss ──
            loss_scene = compute_scene_flare_loss(
                raw_model, hidden, outputs, batch, B, L_slow,
                K_scene, T_per_frame, flare_layer_idx) if use_scene else 0.0

            # ── Tactile FLARE loss ──
            loss_tactile = compute_tactile_flare_loss(
                raw_model, hidden, outputs, batch, B, L_slow,
                K_scene, K_tactile, flare_layer_idx) if use_tactile else 0.0

            # ── Total ──
            loss = loss_act
            if torch.is_tensor(loss_scene):
                loss = loss + args.scene_flare_loss_weight * loss_scene
            if torch.is_tensor(loss_tactile):
                loss = loss + args.tactile_flare_loss_weight * loss_tactile

            metric.update(loss, loss_act, loss_scene, loss_tactile)
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
                        it.set_postfix(
                            step=global_step,
                            loss=f"{m['total_loss']:.5f}",
                            act=f"{m['action_loss']:.5f}",
                            sf=f"{m['scene_flare_loss']:.5f}",
                            tf=f"{m['tactile_flare_loss']:.5f}",
                            lr=f"{lr_now:.2e}")
                    wandb.log({**m, "lr": lr_now, "epoch": epoch}, step=global_step)

            global_step += 1

        if (epoch + 1) % args.save_freq == 0 or epoch == args.n_epochs - 1:
            accelerator.wait_for_everyone()
            save_checkpoint(model, processor, accelerator, args,
                            epoch, global_step, dataset)

    accelerator.print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="qwen3vl_egodex_pretrain_tactile_flare")
    parser.add_argument("--run_name", type=str, default="run_1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
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
    parser.add_argument("--resume_checkpoint", type=str, default="")

    # Validation
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--val_freq", type=int, default=1000)
    parser.add_argument("--max_val_batches", type=int, default=50)

    # Scene FLARE (predict future frames)
    parser.add_argument("--use_scene_flare", type=int, default=1)
    parser.add_argument("--scene_flare_tokens_per_frame", type=int, default=4)
    parser.add_argument("--scene_flare_steps", type=int, default=8)
    parser.add_argument("--scene_flare_frame_stride", type=int, default=4)
    parser.add_argument("--scene_flare_loss_weight", type=float, default=0.5)

    # Tactile FLARE (predict fingertip crops)
    parser.add_argument("--use_tactile_flare", type=int, default=1)
    parser.add_argument("--tactile_flare_crop_size", type=int, default=96)
    parser.add_argument("--tactile_flare_loss_weight", type=float, default=0.5)

    # Shared
    parser.add_argument("--flare_layer_index", type=int, default=-1)

    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name, args.run_name)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
