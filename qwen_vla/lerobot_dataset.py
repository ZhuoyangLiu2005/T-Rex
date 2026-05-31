"""TRexLeRobotDataset — load a LeRobot v3.0 dataset and emit the EXACT batch
contract the T-Rex trainers already consume (drop-in for SftDataset).

The model and the cascaded-flow training loop are unchanged: this dataset +
`collate_fn` reproduce every key `SftDataset.collate_fn` returns
(input_ids/pixel_values/noisy_actions/target/timesteps/tactile_*/flare_*/...),
sourcing frames from LeRobot instead of a JSON.

Design:
  * Temporal windows (action chunk is baked, but the F6 history and FLARE future
    head frames) come from LeRobot `delta_timestamps`.
  * Normalization uses the q01/q99 `meta/trex_norm_stats.json` sidecar so it is
    byte-identical to the JSON pipeline.
  * The LeRobot import is lazy and `collate_fn` consumes plain item dicts, so this
    module imports (and the collate is testable) without lerobot installed.

Co-training across sources is handled at *conversion* time (merge several raw
roots into one LeRobot dataset), so the loader only needs the single-dataset API.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Dict, List, Optional

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from utils.lerobot_common import (
    ACTION_DIM, STATS_KEY,
    KEY_HEAD, KEY_WRIST_R, KEY_WRIST_L, KEY_STATE, KEY_ACTION, KEY_ACTION_ABS,
    KEY_TACF6, DEFORM_KEYS, load_norm_stats,
)

import cv2  # for the tracking-error-noise rotation conversions


# ── state-noise helpers (mirror scripts/train.py) ────────────────────────────
def _rot6d_to_mat(rot6d):
    col1, col2 = rot6d[:3], rot6d[3:6]
    return np.column_stack([col1, col2, np.cross(col1, col2)])


def _arm9d_to_axis_angle(arm_9d):
    R = _rot6d_to_mat(arm_9d[3:9])
    aa, _ = cv2.Rodrigues(R)
    return np.concatenate([arm_9d[:3], aa.flatten()])


def _axis_angle_to_arm9d(arm_aa):
    R, _ = cv2.Rodrigues(arm_aa[3:6])
    return np.concatenate([arm_aa[:3], R[:, 0], R[:, 1]])


def add_tracking_error_noise(state, te_mean, te_std, action_dim):
    noisy = state.copy()
    for arm_idx in range(action_dim // 31):
        off = arm_idx * 31
        arm_aa = _arm9d_to_axis_angle(state[off:off + 9])
        te_off = arm_idx * 28
        noise_arm = np.random.normal(te_mean[te_off:te_off + 6], te_std[te_off:te_off + 6]).astype(np.float32)
        noise_hand = np.random.normal(te_mean[te_off + 6:te_off + 28], te_std[te_off + 6:te_off + 28]).astype(np.float32)
        noisy[off:off + 9] = _axis_angle_to_arm9d(arm_aa + noise_arm)
        noisy[off + 9:off + 31] = state[off + 9:off + 31] + noise_hand
    return noisy


def _normalize(values, mask, vmin, vmax):
    return np.where(mask, np.clip(2 * (values - vmin) / (vmax - vmin + 1e-8) - 1, -1, 1), values)


class TRexLeRobotDataset(torch.utils.data.Dataset):
    def __init__(self, config, processor, accelerator, episodes: Optional[List[int]] = None,
                 _ds=None, _stats=None):
        self.config = config
        self.processor = processor
        self.accelerator = accelerator

        root = config.lerobot_root
        self.root = root
        repo_id = getattr(config, "lerobot_repo_id", "") or os.path.basename(root.rstrip("/"))

        # fps + feature presence from info.json (no lerobot needed for this).
        with open(os.path.join(root, "meta", "info.json")) as f:
            info = json.load(f)
        self.fps = int(info["fps"])
        feats = info["features"]
        self.has_wrist   = KEY_WRIST_R in feats and KEY_WRIST_L in feats
        self.has_tactile = KEY_TACF6 in feats
        self.has_deform  = DEFORM_KEYS[0] in feats

        # flow / flare / tactile config (mirror SftDataset)
        self.image_size = tuple(config.image_size) if getattr(config, "image_size", None) else None
        self.use_flare = bool(getattr(config, "use_flare", 0))
        self.n_flare_steps = int(getattr(config, "n_flare_steps", 0)) if self.use_flare else 0
        self.flare_stride = int(getattr(config, "flare_frame_stride", 1))
        self.use_tactile_vec = bool(getattr(config, "use_tactile_vec", 0))
        self.use_tactile_deform = bool(getattr(config, "use_tactile_deform", 0))
        self.use_tactile_vqvae = bool(getattr(config, "use_tactile_vqvae", 0))
        self.use_robot_state = bool(getattr(config, "use_robot_state", 0))
        self.vqvae_window = int(getattr(config, "vqvae_window", 16))
        self.action_dim = int(getattr(config, "action_dim", ACTION_DIM))

        # ── normalization stats (q01/q99 sidecar) ──
        self.stats_data = _stats if _stats is not None else load_norm_stats(root)
        block = self.stats_data[next(iter(self.stats_data))]
        def _arr(key, sub):
            return np.array(block[key][sub])
        self.action_mask = _arr("action", "mask"); self.action_min = _arr("action", "q01"); self.action_max = _arr("action", "q99")
        self.state_mask  = _arr("state", "mask");  self.state_min  = _arr("state", "q01");  self.state_max  = _arr("state", "q99")
        if self.has_tactile:
            self.tacf6_mask = _arr("tactile_f6", "mask"); self.tacf6_min = _arr("tactile_f6", "q01"); self.tacf6_max = _arr("tactile_f6", "q99")
        te = block.get("tracking_error", {})
        te_dim = (self.action_dim // 31) * 28
        self.te_mean = np.array(te.get("mean", np.zeros(te_dim)), dtype=np.float32)
        self.te_std  = np.array(te.get("std",  np.zeros(te_dim)), dtype=np.float32)

        # ── build the underlying LeRobotDataset (lazy import) ──
        if _ds is not None:
            self.ds = _ds
        else:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            self.ds = LeRobotDataset(
                repo_id, root=root, episodes=episodes,
                delta_timestamps=self._build_delta_timestamps())
        accelerator.print(f"[LeRobot] {repo_id}: {len(self.ds)} frames, fps={self.fps}, "
                          f"wrist={self.has_wrist}, tactile={self.has_tactile}")

    # head: index 0 = current slow image; 1..S = FLARE future frames.
    def _head_offsets(self):
        offs = [0.0]
        for k in range(self.n_flare_steps):
            offs.append((k + 1) * self.flare_stride / self.fps)
        return offs

    def _f6_offsets(self):
        W = self.vqvae_window
        return [(i - (W - 1)) / self.fps for i in range(W)]   # [-(W-1)/fps ... 0]

    def _build_delta_timestamps(self) -> Dict[str, list]:
        dt = {KEY_HEAD: self._head_offsets()}
        if self.has_tactile and (self.use_tactile_vec or self.use_tactile_vqvae):
            dt[KEY_TACF6] = self._f6_offsets()
        return dt

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]   # LeRobot item dict (tensors + task + *_is_pad)

    def create_val_split(self, val_ratio=0.05, seed=42):
        """Split by episode index; returns a val TRexLeRobotDataset (shares stats)."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        n_ep = self.ds.meta.total_episodes
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_ep)
        n_val = max(1, int(n_ep * val_ratio))
        val_eps = sorted(perm[:n_val].tolist())
        train_eps = sorted(perm[n_val:].tolist())
        repo_id = getattr(self.config, "lerobot_repo_id", "") or os.path.basename(self.root.rstrip("/"))
        dt = self._build_delta_timestamps()
        val_ds = LeRobotDataset(repo_id, root=self.root, episodes=val_eps, delta_timestamps=dt)
        self.ds = LeRobotDataset(repo_id, root=self.root, episodes=train_eps, delta_timestamps=dt)
        self.accelerator.print(f"[LeRobot] train/val split: {len(train_eps)}/{len(val_eps)} episodes")
        val = copy.copy(self)
        val.ds = val_ds
        return val

    # ── image helpers ──
    def _img_to_pil(self, img_t: torch.Tensor) -> PIL.Image.Image:
        """LeRobot video tensor [3,H,W] float[0,1] → resized RGB PIL."""
        arr = (img_t.detach().float().clamp(0, 1) * 255.0).to(torch.uint8)
        arr = arr.permute(1, 2, 0).cpu().numpy()           # HWC
        img = PIL.Image.fromarray(arr, mode="RGB")
        if self.image_size is not None:
            img = img.resize(self.image_size, PIL.Image.LANCZOS)
        return img

    def collate_fn(self, batch: List[Dict]) -> Dict:
        cfg = self.config
        B = len(batch)
        device_cpu = torch.device("cpu")

        # ── actions + flow matching ──
        actions = np.stack([np.asarray(x[KEY_ACTION], dtype=np.float32) for x in batch], axis=0)  # [B,16,62]
        norm_actions = torch.tensor(
            _normalize(actions, self.action_mask, self.action_min, self.action_max),
            dtype=torch.bfloat16)
        beta = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
        time = (beta.sample((B,)) * 0.999 + 0.001).to(torch.bfloat16)
        t_ = time[:, None, None]
        noise = torch.randn_like(norm_actions)
        x_t = t_ * noise + (1 - t_) * norm_actions
        u_t = noise - norm_actions
        time_r = (beta.sample((B,)) * 0.999 + 0.001).to(torch.bfloat16)
        eps_r = torch.randn_like(norm_actions)

        # ── tactile ──
        norm_tacf6 = None
        tactile_f6_history_tensor = None
        if self.has_tactile and KEY_TACF6 in batch[0]:
            f6_hist = torch.stack([x[KEY_TACF6].float() for x in batch], dim=0)   # [B,W,10,6]
            if self.use_tactile_vqvae:
                tactile_f6_history_tensor = f6_hist                              # raw, model normalizes
            if self.use_tactile_vec:
                cur = f6_hist[:, -1].reshape(B, -1).numpy()                       # current frame [B,60]
                norm_tacf6 = torch.tensor(
                    _normalize(cur, self.tacf6_mask, self.tacf6_min, self.tacf6_max).reshape(B, -1, 6),
                    dtype=torch.bfloat16)

        deforms_tensor = None
        if self.use_tactile_deform and self.has_deform:
            # each deform key is [3,H,W] float[0,1]; take channel 0 → [H,W].
            per_sample = []
            for x in batch:
                fingers = [x[k][0] for k in DEFORM_KEYS]      # 10 × [H,W]
                per_sample.append(torch.stack(fingers, dim=0))  # [10,H,W]
            deforms_tensor = torch.stack(per_sample, dim=0).unsqueeze(2).float()  # [B,10,1,H,W]

        # ── robot state ──
        state_raw = None
        if self.use_robot_state:
            sl = []
            for x in batch:
                s = np.asarray(x[KEY_STATE], dtype=np.float32)
                s = add_tracking_error_noise(s, self.te_mean, self.te_std, self.action_dim)
                sl.append(torch.tensor(_normalize(s, self.state_mask, self.state_min, self.state_max),
                                       dtype=torch.bfloat16))
            state_raw = torch.stack(sl)

        # ── images → Qwen processor (slow=head[0], fast=[wrist_right, wrist_left]) ──
        all_input_ids, all_pixel_values, all_grid_thw = [], [], []
        n_slow_images = 1
        for x in batch:
            head_seq = x[KEY_HEAD]                      # [1+S, 3, H, W]
            pil_slow = [self._img_to_pil(head_seq[0])]
            pil_fast = []
            if self.has_wrist:
                pil_fast = [self._img_to_pil(x[KEY_WRIST_R]), self._img_to_pil(x[KEY_WRIST_L])]
            all_pil = pil_slow + pil_fast
            content = [{"type": "image"} for _ in pil_slow]
            content.append({"type": "text", "text": x.get("task", "")})
            content += [{"type": "image"} for _ in pil_fast]
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
            inp = self.processor(text=text, images=all_pil, return_tensors="pt", padding=False)
            all_input_ids.append(inp.input_ids[0])
            if "pixel_values" in inp and inp.pixel_values is not None:
                all_pixel_values.append(inp.pixel_values)
                all_grid_thw.append(inp.image_grid_thw)

        # ── FLARE future head frames ──
        flare_pixel_values = flare_grid_thw = None
        if self.n_flare_steps > 0:
            flare_pil = []
            for x in batch:
                head_seq = x[KEY_HEAD]                               # [1+S,3,H,W]
                is_pad = x.get(f"{KEY_HEAD}_is_pad")
                for k in range(self.n_flare_steps):
                    fi = 1 + k
                    if is_pad is not None and bool(is_pad[fi]):
                        flare_pil.append(self._img_to_pil(head_seq[0]))   # fall back to current
                    else:
                        flare_pil.append(self._img_to_pil(head_seq[fi]))
            finp = self.processor.image_processor(flare_pil, return_tensors="pt")
            flare_pixel_values = finp.pixel_values.to(torch.bfloat16)
            flare_grid_thw = finp.image_grid_thw

        # ── pad input_ids (left-pad, like SftDataset) ──
        pad_id = self.processor.tokenizer.pad_token_id or 0
        max_len = max(ids.shape[0] for ids in all_input_ids)
        padded_ids, attn_ms = [], []
        for ids in all_input_ids:
            pad = max_len - ids.shape[0]
            padded_ids.append(F.pad(ids, (pad, 0), value=pad_id))
            a = torch.ones(max_len, dtype=torch.long)
            if pad > 0:
                a[:pad] = 0
            attn_ms.append(a)
        input_ids = torch.stack(padded_ids)
        attention_mask = torch.stack(attn_ms)
        pixel_values = torch.cat(all_pixel_values, dim=0) if all_pixel_values else None
        image_grid_thw = torch.cat(all_grid_thw, dim=0) if all_grid_thw else None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "n_slow_images": n_slow_images,
            "noisy_actions": x_t,
            "target": u_t,
            "timesteps": time,
            "norm_actions": norm_actions,
            "tactile_f6s": norm_tacf6,
            "tactile_deforms": deforms_tensor,
            "tactile_f6s_delayed": norm_tacf6,            # delay_k=0 (parity with posttrain JSON path)
            "tactile_deforms_delayed": deforms_tensor,
            "tactile_codes": None,                        # embedded VQ-VAE encodes the history
            "tactile_f6_history": tactile_f6_history_tensor,
            "time_r": time_r,
            "eps_r": eps_r,
            "state_raw": state_raw,
            "flare_pixel_values": flare_pixel_values,
            "flare_grid_thw": flare_grid_thw,
        }
