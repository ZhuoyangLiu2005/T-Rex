"""
Qwen3-VL MoT training with flare prediction + per-modality Tactile CFG + tactile memory.

Extends train_qwen3vl_flare.py with:
  --cfg_drop_force P       : probability of replacing F6 tokens with null (default 0.15)
  --cfg_drop_deform P      : probability of replacing deform tokens with null (default 0.15)
  --use_learnable_null 1   : use learnable null embeddings instead of zero-masking
  --tactile_cfg_scale W    : guidance scale at inference (default 1.0, logged only)
  --tactile_history_len T  : number of tactile timesteps to feed (default 1 = no memory)
  --tactile_history_stride : frame stride between history steps (default 1)

Per-modality CFG + learnable null
---------------------------------
Force and deform are dropped independently, each replaced by a learned "null
embedding" specific to that modality. Zero-masking (previous default) puts the
tactile expert in an arbitrary "pretend no signal" state that is in-distribution
for the embedder; a learnable null is a cleaner absent-signal marker.
  tac_null_f6      : [1, 1, H]  learned null for the force modality
  tac_null_deform  : [1, 1, H]  learned null for the deform modality

At inference, two independent guidance scales can be applied:
  v_guided = v_uncond + W_f*(v_cond_force - v_uncond) + W_d*(v_cond_deform - v_uncond)

Tactile memory (compressed, constant MoT seq length)
-----------------------------------------------------
When `tactile_history_len > 1`, the dataset builds a T-frame history window
per sample. But history is NOT concatenated into the MoT sequence -- naïve
concat would T× the tactile token count and pay quadratic attention cost at
every layer × every denoise step. Instead we compress history OUTSIDE the
MoT with a `TacTemporalPool` (one per modality):

  embedder -> [B, T, n_fingers, H] -> TacTemporalPool -> [B, n_fingers, H]

The pool runs finger-major internally (per-finger independent attention
pool over T, giving the right inductive bias for slip / contact onset /
release) and mixes history into the current frame via a zero-init gated
residual, so memory starts disabled and turns on as training learns to
use it. The MoT sees the same `n_fingers` tokens per modality regardless
of T -- no attention-mask surgery, no latency hit.

Final tactile sequence layout (unchanged from no-memory case):
  [ f6_finger0..N-1,    deform_finger0..N-1 ]

Unified loss (inherited from CFG script)
----------------------------------------
  v_final = v_act + delta_v,  loss = MSE(v_final, target).
Gradients flow through both experts jointly.
"""

import os, sys

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

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

from typing import List, Dict
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator
from transformers import AutoProcessor, set_seed
from datasets import Dataset as HFDataset

from qwen_vla import Qwen3VLVLAModel, extend_position_ids_for_flare, split_slow_fast_embeds
import cv2

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


class TacTemporalPool(nn.Module):
    """Summarize tactile history [B, T, n_fingers, H] -> [B, n_fingers, H].

    Rationale: naïvely concatenating T history tokens inflates the MoT
    sequence length by T× per modality and adds quadratic attention cost
    at every layer × every denoise step. We instead pool outside the MoT
    so the MoT sees the same number of tokens as the no-memory case.

    Structure: per-finger independent attention pool (finger axis treated
    as batch), giving the correct temporal-per-finger inductive bias for
    slip / contact onset / release. A single learned query dot-products
    over the time-stamped history.

    Gating: `out = current + tanh(gate) * history`, `gate` zero-init.
    Memory starts fully disabled and the model learns to mix it in when
    it helps -- matches the residual-tactile philosophy already used for
    `final_layer_tactile`.
    """

    def __init__(self, hidden_size: int, t_max: int):
        super().__init__()
        self.H = hidden_size
        self.t_max = t_max
        self.time_embed = nn.Parameter(
            (torch.randn(t_max, hidden_size) * 0.02).to(torch.bfloat16))
        self.query = nn.Parameter(
            (torch.randn(hidden_size) * 0.02).to(torch.bfloat16))
        self.gate = nn.Parameter(torch.zeros(hidden_size, dtype=torch.bfloat16))
        self.scale = hidden_size ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, n_fingers, H]; current frame is x[:, -1].
        B, T, nf, H = x.shape
        current = x[:, -1]  # [B, nf, H]
        if T == 1:
            return current
        tpe = self.time_embed[:T].to(dtype=x.dtype).view(1, T, 1, H)
        x = x + tpe
        q = self.query.to(dtype=x.dtype).view(1, 1, 1, H)
        # Finger-major internally: per-finger softmax over time.
        logits = (x * q).sum(-1) * self.scale            # [B, T, nf]
        attn = torch.softmax(logits, dim=1)               # softmax over T
        history = (x * attn.unsqueeze(-1)).sum(1)         # [B, nf, H]
        gate = torch.tanh(self.gate.to(dtype=x.dtype)).view(1, 1, H)
        return current + gate * history


def _rot6d_to_mat(rot6d):
    """6D rotation (first two columns) -> 3x3 rotation matrix."""
    col1 = rot6d[:3]
    col2 = rot6d[3:6]
    col3 = np.cross(col1, col2)
    return np.column_stack([col1, col2, col3])


def _arm9d_to_axis_angle(arm_9d):
    """[trans(3), rot6d(6)] -> [trans(3), axis_angle(3)]."""
    R = _rot6d_to_mat(arm_9d[3:9])
    aa, _ = cv2.Rodrigues(R)
    return np.concatenate([arm_9d[:3], aa.flatten()])


def _axis_angle_to_arm9d(arm_aa):
    """[trans(3), axis_angle(3)] -> [trans(3), rot6d(6)]."""
    R, _ = cv2.Rodrigues(arm_aa[3:6])
    return np.concatenate([arm_aa[:3], R[:, 0], R[:, 1]])


def add_tracking_error_noise(state, te_mean, te_std, action_dim):
    """Add tracking-error noise to robot state.

    Supports single-arm (action_dim=31, te=28D) and bimanual (action_dim=62, te=56D).
    Per arm (28D tracking error = 3 xyz + 3 axis-angle + 22 hand):
      1. Convert arm 9D [trans, rot6d] -> [trans, axis_angle]
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


class SftDataset(Dataset):
    def __init__(self, config, processor, accelerator):
        self.config = config
        self.processor = processor
        self.accelerator = accelerator

        self.hf_dataset = HFDataset.from_json(
            config.data_path, keep_in_memory=False,
        )

        stats_path = config.data_path.replace(".json", "_statistics.json")
        with open(stats_path, "r") as f:
            self.stats_data = json.load(f)
        self.dataset_name = next(iter(self.stats_data))

        def _arr(key, sub):
            return np.array(self.stats_data[self.dataset_name][key][sub])

        self.action_mask = _arr("action", "mask")
        self.action_min  = _arr("action", "q01")
        self.action_max  = _arr("action", "q99")
        self.state_mask  = _arr("state",  "mask")
        self.state_min   = _arr("state",  "q01")
        self.state_max   = _arr("state",  "q99")
        self.tacf6_mask  = _arr("tactile_f6", "mask")
        self.tacf6_min   = _arr("tactile_f6", "q01")
        self.tacf6_max   = _arr("tactile_f6", "q99")

        # Tracking error stats for state noise injection (used when use_robot_state=1)
        # Dimension: 28D per arm (6 arm + 22 hand), scales with n_arms
        te_dim = (config.action_dim // 31) * 28
        te_stats = self.stats_data[self.dataset_name].get("tracking_error", {})
        self.te_mean = np.array(te_stats.get("mean", np.zeros(te_dim)), dtype=np.float32)
        self.te_std  = np.array(te_stats.get("std",  np.zeros(te_dim)), dtype=np.float32)

        self.img_dir = os.path.dirname(config.data_path)

        if config.image_size:
            self.image_size = tuple(config.image_size)
        else:
            self.image_size = None

        # For flare frame loading: build per-episode frame lists
        # so we can look up flare frames by index offset.
        self._build_episode_index()

        accelerator.print(f"Dataset size: {len(self.hf_dataset)}")

    def _build_episode_index(self):
        """Group samples by episode to enable flare frame lookup."""
        # Each sample has image paths like '.../taskname/episodename/imageXX_view.png'
        # We group by episode dir and sort by frame index within each episode.
        # For simplicity, we just store the list of slow image paths per sample index.
        # Flare frames are loaded from the JSON data directly
        pass  # Flare frames are loaded from the JSON data directly

    def __len__(self):
        return len(self.hf_dataset)

    @staticmethod
    def _episode_prefix(sample):
        """Episode identifier = directory of the first slow image path."""
        paths = sample.get("input_image_slow", [])
        if not paths:
            return ""
        return os.path.dirname(paths[0])

    def __getitem__(self, idx):
        sample = dict(self.hf_dataset[idx])
        T = max(getattr(self.config, "tactile_history_len", 1), 1)
        stride = max(getattr(self.config, "tactile_history_stride", 1), 1)

        # Always emit a history window of length T (T=1 = no memory).
        # Ordering: oldest past first, current last.
        cur_prefix = self._episode_prefix(sample)
        f6_hist = []
        deform_paths_hist = []
        for t in range(T - 1, 0, -1):
            past_idx = idx - t * stride
            past_sample = None
            if past_idx >= 0:
                try:
                    cand = dict(self.hf_dataset[past_idx])
                    if self._episode_prefix(cand) == cur_prefix:
                        past_sample = cand
                except Exception:
                    pass
            src = past_sample if past_sample is not None else sample
            f6_hist.append(src.get("tactile_f6"))
            deform_paths_hist.append(list(src.get("tactile_image_deform", []) or []))
        f6_hist.append(sample.get("tactile_f6"))
        deform_paths_hist.append(list(sample.get("tactile_image_deform", []) or []))

        sample["_tactile_f6_hist"] = f6_hist
        sample["_tactile_deform_hist"] = deform_paths_hist
        return sample

    @staticmethod
    def _normalize(values, mask, vmin, vmax):
        return np.where(
            mask,
            np.clip(2 * (values - vmin) / (vmax - vmin + 1e-8) - 1, -1, 1),
            values,
        )

    def _open(self, rel_path):
        img = PIL.Image.open(os.path.join(self.img_dir, rel_path)).convert("RGB")
        if self.image_size is not None:
            img = img.resize(self.image_size, PIL.Image.LANCZOS)
        return img

    def _open_gray(self, path):
        full = path if os.path.isabs(path) else os.path.join(self.img_dir, path)
        img = PIL.Image.open(full).convert("L")
        return np.array(img, dtype=np.float32) / 255.0

    def _beta_sample(self, n, device):
        d = torch.distributions.Beta(
            torch.tensor(1.5, dtype=torch.float32, device=device),
            torch.tensor(1.0, dtype=torch.float32, device=device),
        )
        return (d.sample((n,)) * 0.999 + 0.001).to(torch.bfloat16)

    def _load_flare_frame(self, sample, k, K, frame_stride):
        slow_path = sample.get("input_image_slow", [""])[0]
        if not slow_path:
            return None

        # Extract frame index from path: image{idx}_{view}.png
        match = re.search(r'image(\d+)_', os.path.basename(slow_path))
        if not match:
            return None

        current_idx = int(match.group(1))
        flare_idx = current_idx + (k + 1) * frame_stride
        flare_path = re.sub(r'image\d+_', f'image{flare_idx}_', slow_path)

        full_path = os.path.join(self.img_dir, flare_path) if not os.path.isabs(flare_path) else flare_path
        if os.path.exists(full_path):
            return self._open(flare_path)
        else:
            # Out of episode range -> use last available or return None
            return None

    def collate_fn(self, batch: List[Dict]) -> Dict:
        cfg = self.config
        B = len(batch)

        actions = np.array([x["action"] for x in batch], dtype=np.float32)
        actions = actions.reshape(B, -1, cfg.action_dim)
        norm_actions = self._normalize(actions, self.action_mask, self.action_min, self.action_max)
        norm_actions = torch.tensor(norm_actions, dtype=torch.bfloat16)

        device_cpu = norm_actions.device
        time = self._beta_sample(B, device_cpu)
        t_ = time[:, None, None]
        noise = torch.randn_like(norm_actions)
        x_t = t_ * noise + (1 - t_) * norm_actions
        u_t = noise - norm_actions

        # Unified temporal shape: [B, T, n_fingers, ...] (T=1 means no memory).
        T_hist = max(getattr(cfg, "tactile_history_len", 1), 1)

        norm_tacf6 = None
        if cfg.use_tactile_vec:
            # Each sample carries a history list of length T of flat [n_fingers*6] arrays.
            tacf6_hist = np.array(
                [x["_tactile_f6_hist"] for x in batch], dtype=np.float32
            ).reshape(B, T_hist, -1)
            flat = tacf6_hist.reshape(B * T_hist, -1)
            normed = self._normalize(flat, self.tacf6_mask, self.tacf6_min, self.tacf6_max)
            norm_tacf6 = torch.tensor(
                normed.reshape(B, T_hist, -1, 6), dtype=torch.bfloat16
            )  # [B, T, n_fingers, 6]

        deforms_tensor = None
        if cfg.use_tactile_deform:
            deforms = []  # [B, T, n_fingers, H, W]
            for x in batch:
                frames = []  # [T, n_fingers, H, W]
                for paths_at_t in x["_tactile_deform_hist"]:
                    imgs = [self._open_gray(p) for p in paths_at_t]
                    frames.append(imgs)
                deforms.append(frames)
            deforms_tensor = torch.tensor(np.array(deforms)).unsqueeze(3)
            # Shape: [B, T, n_fingers, 1, H, W]

        state_raw_list = []
        if cfg.use_robot_state:
            for x in batch:
                state_raw = np.array(x["state_fast"], dtype=np.float32)
                state_raw = add_tracking_error_noise(state_raw,
                                                     self.te_mean, self.te_std,
                                                     cfg.action_dim)
                norm_state = self._normalize(state_raw, self.state_mask,
                                     self.state_min, self.state_max)
                state_raw_list.append(torch.tensor(norm_state, dtype=torch.bfloat16))

        all_input_ids = []
        all_pixel_values = []
        all_grid_thw = []
        n_slow_images = 0

        for x in batch:
            slow_imgs = x.get("input_image_slow", [])
            fast_imgs = x.get("input_image_fast", [])
            n_slow_images = len(slow_imgs)

            pil_slow = [self._open(p) for p in slow_imgs]
            pil_fast = [self._open(p) for p in fast_imgs]
            all_pil = pil_slow + pil_fast # a list, including all images

            content = []
            for _ in pil_slow:
                content.append({"type": "image"})
            content.append({"type": "text", "text": x.get("input_prompt", "")})
            for _ in pil_fast:
                content.append({"type": "image"})

            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inp = self.processor(
                text=text,
                images=all_pil if all_pil else None,
                return_tensors="pt", padding=False,
            )
            all_input_ids.append(inp.input_ids[0])
            if "pixel_values" in inp and inp.pixel_values is not None:
                all_pixel_values.append(inp.pixel_values)
                all_grid_thw.append(inp.image_grid_thw)

        n_flare_steps = cfg.n_flare_steps if cfg.use_flare else 0
        flare_stride = cfg.frame_stride
        flare_pixel_values = None
        flare_grid_thw = None

        if n_flare_steps > 0: # use flare
            flare_pil_imgs = []
            for x in batch:
                for k in range(n_flare_steps):
                    flare_images = self._load_flare_frame(x, k, n_flare_steps, flare_stride)
                    if flare_images is None:
                        slow_paths = x.get("input_image_slow", [])
                        flare_images = self._open(slow_paths[0]) if slow_paths else PIL.Image.new("RGB", self.image_size or (384, 288))
                    flare_pil_imgs.append(flare_images)

            flare_inp = self.processor.image_processor(flare_pil_imgs, return_tensors="pt")
            flare_pixel_values = flare_inp.pixel_values.to(torch.bfloat16)
            flare_grid_thw = flare_inp.image_grid_thw

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
            "tactile_f6s": norm_tacf6,
            "tactile_deforms": deforms_tensor,
            "state_raw": state_raw,
            "flare_pixel_values": flare_pixel_values, # [B*S patches, C] or None (S = n_flare_steps)
            "flare_grid_thw": flare_grid_thw, # [B*S, 3] or None
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
                "cfg_drop_force": args.cfg_drop_force,
                "cfg_drop_deform": args.cfg_drop_deform,
                "use_learnable_null": args.use_learnable_null,
                "tactile_cfg_scale": args.tactile_cfg_scale,
                "tactile_history_len": args.tactile_history_len,
                "tactile_history_stride": args.tactile_history_stride,
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


def train(args):
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
    model.initialize_vla_weights()
    if args.use_flare:
        accelerator.print(
            f"Flare alignment: {args.n_flare_steps} steps x {args.n_flare_tokens_per_frame} tok/frame "
            f"= {model.n_flare_tokens} total tokens, layer_index={args.flare_layer_index}"
        )

    is_stage1 = (args.training_stage == 1)
    has_any_tactile = bool(args.use_tactile_vec or args.use_tactile_deform)
    freeze_tactile = is_stage1 or (not has_any_tactile)

    # ── Learnable CFG null embeddings (per modality) + temporal memory pool ──
    # Registered on the outer VLA wrapper so they are saved/loaded with the
    # checkpoint. Names start with "tac_" -- the freeze logic below groups
    # them with the tactile expert (stage-2 only).
    H_hidden = model.config.hidden_size
    T_hist = max(args.tactile_history_len, 1)
    if args.use_tactile_vec and args.use_learnable_null:
        model.tac_null_f6 = nn.Parameter(
            (torch.randn(1, 1, H_hidden) * 0.02).to(torch.bfloat16))
    if args.use_tactile_deform and args.use_learnable_null:
        model.tac_null_deform = nn.Parameter(
            (torch.randn(1, 1, H_hidden) * 0.02).to(torch.bfloat16))
    if T_hist > 1:
        # One temporal pool per modality (independent params, matching pool
        # structure but modality-specific dynamics: deform ≠ force over time).
        if args.use_tactile_vec:
            model.tac_pool_f6 = TacTemporalPool(H_hidden, T_hist)
        if args.use_tactile_deform:
            model.tac_pool_deform = TacTemporalPool(H_hidden, T_hist)

    if has_any_tactile:
        accelerator.print(
            f"Tactile CFG: drop(force)={args.cfg_drop_force}, "
            f"drop(deform)={args.cfg_drop_deform}, "
            f"null={'learnable' if args.use_learnable_null else 'zero'}, "
            f"cfg_scale={args.tactile_cfg_scale} (inference only)")
        if args.tactile_history_len > 1:
            accelerator.print(
                f"Tactile memory: history_len={args.tactile_history_len}, "
                f"stride={args.tactile_history_stride}")

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

    if not freeze_tactile:
        for name, param in model.named_parameters():
            if "_tactile" in name:
                if param.ndim >= 2:
                    nn.init.xavier_uniform_(param)
                elif param.ndim == 1:
                    nn.init.zeros_(param)
        # NOTE: No zero-init of final_layer_tactile -- CFG dropout handles
        # regularization. Xavier init gives the tactile expert strong gradients
        # from step 1, preventing gradient starvation.

    for name, param in model.named_parameters():
        if name.startswith("visual") or name.startswith("deform_encoder"):
            param.requires_grad = False
        elif ("_tactile" in name or "final_layer_tactile" in name
              or name.startswith("tac_")):  # tac_null_*, tac_pool_*
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

    dataset = SftDataset(args, processor, accelerator)
    dataloader = DataLoader(
        dataset, batch_size=args.train_bsz_per_gpu, shuffle=True,
        collate_fn=dataset.collate_fn, num_workers=4, pin_memory=True,
    )

    num_training_steps = (
        int(len(dataloader) * args.n_epochs)
        // accelerator.gradient_accumulation_steps
        // dist.get_world_size()
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_rates * num_training_steps),
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    metric = TrainingMetrics(device=torch.cuda.current_device())
    global_step = 0
    T_per_frame = args.n_flare_tokens_per_frame
    S_steps = args.n_flare_steps
    K = T_per_frame * S_steps  # total flare tokens
    use_flare = bool(args.use_flare and K > 0)
    flare_layer_idx = args.flare_layer_index
    model.train()

    for epoch in range(args.n_epochs):
        from tqdm import tqdm
        it = (tqdm(dataloader, total=len(dataloader))
              if accelerator.is_main_process else dataloader)

        for batch in it:
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

            if is_stage1: # pretrain
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

                def _cfg_drop(tok, p, null_param_name):
                    """Per-sample drop → learnable null (or zero) on [B, nf, H]."""
                    if p <= 0:
                        return tok
                    Bs = tok.shape[0]
                    drop = (torch.rand(Bs, device=tok.device) < p).view(Bs, 1, 1)
                    if args.use_learnable_null and hasattr(raw_model, null_param_name):
                        null_emb = getattr(raw_model, null_param_name).to(
                            dtype=tok.dtype).view(1, 1, -1)
                    else:
                        null_emb = torch.zeros(
                            1, 1, tok.shape[-1],
                            device=tok.device, dtype=tok.dtype)
                    return torch.where(drop, null_emb.expand_as(tok), tok)

                # ── F6 force tokens ─────────────────────────────────────────
                # Shape contract entering MoT: [B, n_fingers, H] (constant in T).
                if args.use_tactile_vec and batch["tactile_f6s"] is not None:
                    f6 = batch["tactile_f6s"].to(
                        slow_embeds.device, dtype=slow_embeds.dtype)  # [B,T,nf,6]
                    f6_tok = raw_model.tacf6_embedder(f6)              # [B,T,nf,H]
                    # Temporal memory: compress T → 1 (per-finger attention pool).
                    if f6_tok.shape[1] > 1 and hasattr(raw_model, "tac_pool_f6"):
                        f6_tok = raw_model.tac_pool_f6(f6_tok)         # [B,nf,H]
                    else:
                        f6_tok = f6_tok[:, -1]                         # current only
                    f6_tok = _cfg_drop(f6_tok, args.cfg_drop_force, "tac_null_f6")
                    tac_parts.append(f6_tok)

                # ── Deform tokens ───────────────────────────────────────────
                if args.use_tactile_deform and batch["tactile_deforms"] is not None:
                    deforms = batch["tactile_deforms"].to(
                        slow_embeds.device, dtype=slow_embeds.dtype)   # [B,T,nf,1,H,W]
                    Bs, Ts, nf_d, C, Hh, Ww = deforms.shape
                    with torch.no_grad():
                        feats = raw_model.deform_encoder(deforms.view(-1, C, Hh, Ww))
                    feats = feats.view(Bs, Ts, nf_d, -1)
                    d_tok = raw_model.deform_proj(feats.to(slow_embeds.dtype))  # [B,T,nf,H]
                    if d_tok.shape[1] > 1 and hasattr(raw_model, "tac_pool_deform"):
                        d_tok = raw_model.tac_pool_deform(d_tok)       # [B,nf,H]
                    else:
                        d_tok = d_tok[:, -1]
                    d_tok = _cfg_drop(d_tok, args.cfg_drop_deform, "tac_null_deform")
                    tac_parts.append(d_tok)

                if tac_parts:
                    tactile_embeds = torch.cat(tac_parts, dim=1)
                else:
                    tactile_embeds = torch.empty(
                        (B, 0, slow_embeds.shape[2]),
                        device=slow_embeds.device, dtype=slow_embeds.dtype)

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
                    v_final = v_act + delta_v
                    # Unified loss: both experts jointly predict the velocity
                    loss_act = nn.MSELoss()(v_final, target)
                    # Separate tactile loss for monitoring (not used in backward)
                    with torch.no_grad():
                        loss_tac = nn.MSELoss()(delta_v, target - v_act.detach())
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
                        # Adaptive average pool: [n_tok, H] -> [T_per_frame, H]
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
            # loss_tac is monitoring-only with CFG; the unified loss_act already
            # includes the tactile contribution via v_final = v_act + delta_v
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
                    wandb.log({
                        "total_loss":   m["total_loss"],
                        "action_loss":  m["action_loss"],
                        "tactile_loss": m["tactile_loss"],
                        "flare_loss":  m["flare_loss"],
                        "lr": lr_now,
                    }, step=global_step)

            global_step += 1

        if (epoch + 1) % args.save_freq == 0 or epoch == args.n_epochs - 1:
            accelerator.wait_for_everyone()
            save_checkpoint(model, processor, accelerator, args,
                            epoch, global_step, dataset.stats_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="qwen3vl_mot_flare_cfg")
    parser.add_argument("--run_name", type=str, default="run_1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="")
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

    parser.add_argument("--action_dim", type=int, default=31)
    parser.add_argument("--action_chunk", type=int, default=8)
    parser.add_argument("--use_robot_state", type=int, default=0)
    parser.add_argument("--use_tactile_vec", type=int, default=0)
    parser.add_argument("--use_tactile_deform", type=int, default=1)
    parser.add_argument("--deform_encoder_ckpt", type=str, default="")
    parser.add_argument("--tactile_intermediate_size", type=int, default=0)
    parser.add_argument("--training_stage", type=int, default=2, choices=[1, 2])
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--tactile_loss_weight", type=float, default=1.0)
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("W", "H"))

    # Flare
    parser.add_argument("--use_flare", type=int, default=1, help="Enable flare visual prediction for latent expert.")
    parser.add_argument("--n_flare_tokens_per_frame", type=int, default=4, help="Number of tokens per future frame.")
    parser.add_argument("--n_flare_steps", type=int, default=8, help="Number of future steps to predict.")
    parser.add_argument("--flare_loss_weight", type=float, default=0.5, help="Weight for flare prediction cosine loss.")
    parser.add_argument("--flare_frame_stride", type=int, default=2, help="Temporal stride for flare frame targets.")
    parser.add_argument("--flare_layer_index", type=int, default=-1, help="Layer to extract flare hidden states from (-1=last, e.g. -7 for ~3/4 depth).")
    parser.add_argument("--frame_stride", type=int, default=2)

    # Per-modality Tactile CFG with learnable null
    parser.add_argument("--cfg_drop_force", type=float, default=0.15,
                        help="Probability of replacing F6 tokens with null (per-sample).")
    parser.add_argument("--cfg_drop_deform", type=float, default=0.15,
                        help="Probability of replacing deform tokens with null (per-sample).")
    parser.add_argument("--use_learnable_null", type=int, default=1,
                        help="1 = learnable null embedding per modality; 0 = zero-mask.")
    parser.add_argument("--tactile_cfg_scale", type=float, default=1.0,
                        help="Guidance scale at inference (logged only, not used in training).")

    # Tactile memory (temporal history window)
    parser.add_argument("--tactile_history_len", type=int, default=1,
                        help="Number of tactile timesteps to feed (1 = no memory).")
    parser.add_argument("--tactile_history_stride", type=int, default=1,
                        help="Frame stride between history steps.")

    args = parser.parse_args()

    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name, args.run_name)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)
