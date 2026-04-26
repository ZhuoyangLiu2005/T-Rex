"""
Qwen3-VL MoT training with tactile auxiliary objectives.

Design (differs from train_qwen3vl_tflare_gate.py):

  - Tactile tokens (F6 + deform, CURRENT FRAME ONLY — no history) live
    INSIDE the action block — action expert reads them directly via
    causal attention. No duplication in the tactile block.

  - Tactile block contains only learnable QUERY tokens routed to the
    tactile expert:
        tflare_q    (K_tac tokens)   — future tactile prediction
        contact_q   (n_fingers)      — per-finger contact classification
        force_q     (n_fingers)      — per-finger force regression
    All three query sets share the tactile expert's Q/K/V/MLP weights and
    read tactile info from the action block via joint attention.

  - Action expert still produces v_act (62-d) and is the sole action head.
    No gate, no v_tac, no solo loss.

Sequence layout:
  [ slow | flare_q | fast | state | tac_f6 | tac_deform | t | x_t |
    tflare_q | contact_q | force_q ]
  └── latent ──┘    └────── action block ────────────────┘
  └──── tactile block ──────┘

Losses:
  L = MSE(v_act, target_action)
    + w_vflare  · visual_flare_cosine
    + w_tflare  · tactile_flare_cosine
    + w_contact · BCE(contact_logits, contact_labels)
    + w_force   · MSE(force_pred, force_labels)
"""

import os, sys, copy

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


# ── Rotation helpers (unchanged) ────────────────────────────────────────────

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
    n_arms = action_dim // 31
    for arm_idx in range(n_arms):
        offset = arm_idx * 31
        arm_9d = state[offset:offset + 9]
        hand_22d = state[offset + 9:offset + 31]
        arm_aa = _arm9d_to_axis_angle(arm_9d)
        te_off = arm_idx * 28
        noise_arm  = np.random.normal(te_mean[te_off:te_off + 6],
                                      te_std[te_off:te_off + 6]).astype(np.float32)
        noise_hand = np.random.normal(te_mean[te_off + 6:te_off + 28],
                                      te_std[te_off + 6:te_off + 28]).astype(np.float32)
        noisy[offset:offset + 9]    = _axis_angle_to_arm9d(arm_aa + noise_arm)
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


# ── Dataset ─────────────────────────────────────────────────────────────────

class SftDataset(Dataset):
    def __init__(self, config, processor, accelerator):
        self.config = config
        self.processor = processor
        self.accelerator = accelerator

        self.hf_dataset = HFDataset.from_json(config.data_path, keep_in_memory=False)

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

        te_dim = (config.action_dim // 31) * 28
        te_stats = self.stats_data[self.dataset_name].get("tracking_error", {})
        self.te_mean = np.array(te_stats.get("mean", np.zeros(te_dim)), dtype=np.float32)
        self.te_std  = np.array(te_stats.get("std",  np.zeros(te_dim)), dtype=np.float32)

        self.img_dir = os.path.dirname(config.data_path)
        self.image_size = tuple(config.image_size) if config.image_size else None

        accelerator.print(f"Dataset size: {len(self.hf_dataset)}")

    def __len__(self):
        return len(self.hf_dataset)

    @staticmethod
    def _episode_prefix(sample):
        paths = sample.get("input_image_slow", [])
        return os.path.dirname(paths[0]) if paths else ""

    def __getitem__(self, idx):
        sample = dict(self.hf_dataset[idx])
        cur_prefix = self._episode_prefix(sample)

        # No tactile history — action expert sees only the CURRENT-frame F6
        # and CURRENT-frame deform. Tactile is read directly from
        # sample["tactile_f6"] and sample["tactile_image_deform"] in collate_fn.

        # ── Future tactile for tflare: F6 only (no deform future I/O) ───────
        if getattr(self.config, "use_tactile_flare", 0):
            S = self.config.n_tfl_steps
            stride = self.config.tactile_flare_stride
            tflare_f6 = []
            for k in range(S):
                fut_idx = idx + (k + 1) * stride
                fut_sample = None
                if 0 <= fut_idx < len(self.hf_dataset):
                    try:
                        cand = dict(self.hf_dataset[fut_idx])
                        if self._episode_prefix(cand) == cur_prefix:
                            fut_sample = cand
                    except Exception:
                        pass
                src = fut_sample if fut_sample is not None else sample
                tflare_f6.append(src.get("tactile_f6"))
            sample["_tflare_f6"] = tflare_f6

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
        match = re.search(r'image(\d+)_', os.path.basename(slow_path))
        if not match:
            return None
        current_idx = int(match.group(1))
        flare_idx = current_idx + (k + 1) * frame_stride
        flare_path = re.sub(r'image\d+_', f'image{flare_idx}_', slow_path)
        full_path = os.path.join(self.img_dir, flare_path) if not os.path.isabs(flare_path) else flare_path
        if os.path.exists(full_path):
            try:
                return self._open(flare_path)
            except (PIL.UnidentifiedImageError, OSError):
                return None
        return None

    def collate_fn(self, batch: List[Dict]) -> Dict:
        cfg = self.config
        B = len(batch)

        actions = np.array([x["action"] for x in batch], dtype=np.float32)
        actions = actions.reshape(B, -1, cfg.action_dim)
        norm_actions = self._normalize(actions, self.action_mask, self.action_min, self.action_max)
        norm_actions = torch.tensor(norm_actions, dtype=torch.bfloat16)

        time = self._beta_sample(B, norm_actions.device)
        t_ = time[:, None, None]
        noise = torch.randn_like(norm_actions)
        x_t = t_ * noise + (1 - t_) * norm_actions
        u_t = noise - norm_actions

        # ── Tactile history (current + T-1 past frames) ────────────────────
        # Current-frame F6 only — no history.
        tactile_f6_current = None
        if cfg.use_tactile_vec:
            f6_raw = np.array([x["tactile_f6"] for x in batch], dtype=np.float32)
            f6_raw = f6_raw.reshape(B, -1, 6)  # [B, n_fingers, 6]
            normed = self._normalize(
                f6_raw.reshape(B, -1), self.tacf6_mask, self.tacf6_min, self.tacf6_max)
            tactile_f6_current = torch.tensor(
                normed.reshape(B, -1, 6), dtype=torch.bfloat16)

        tactile_deform_current = None
        if cfg.use_tactile_deform:
            # Only CURRENT-frame deform: n_fingers PNGs per sample
            # (was T × n_fingers with history; now 1 × n_fingers).
            dfms = []
            for x in batch:
                paths = list(x.get("tactile_image_deform", []) or [])
                imgs = [self._open_gray(p) for p in paths]
                dfms.append(imgs)
            tactile_deform_current = torch.tensor(np.array(dfms)).unsqueeze(2)
            # [B, n_fingers, 1, H, W]

        # ── Per-finger contact + force labels from CURRENT raw F6 ───────────
        raw_f6_current = np.array([x["tactile_f6"] for x in batch], dtype=np.float32)
        raw_f6_current = raw_f6_current.reshape(B, -1, 6)  # [B, n_fingers, 6]
        force_mag = np.linalg.norm(raw_f6_current[..., :3], axis=-1)  # [B, n_fingers]
        contact_labels = (force_mag > cfg.contact_force_threshold).astype(np.float32)
        force_labels = (force_mag / max(cfg.force_scale, 1e-6)).astype(np.float32)
        contact_labels = torch.tensor(contact_labels, dtype=torch.bfloat16)
        force_labels   = torch.tensor(force_labels,   dtype=torch.bfloat16)

        state_raw_list = []
        if cfg.use_robot_state:
            for x in batch:
                state_raw = np.array(x["state_fast"], dtype=np.float32)
                state_raw = add_tracking_error_noise(state_raw, self.te_mean, self.te_std, cfg.action_dim)
                norm_state = self._normalize(state_raw, self.state_mask, self.state_min, self.state_max)
                state_raw_list.append(torch.tensor(norm_state, dtype=torch.bfloat16))

        all_input_ids, all_pixel_values, all_grid_thw = [], [], []
        n_slow_images = 0
        for x in batch:
            slow_imgs = x.get("input_image_slow", [])
            fast_imgs = x.get("input_image_fast", [])
            n_slow_images = len(slow_imgs)

            pil_slow = [self._open(p) for p in slow_imgs]
            pil_fast = [self._open(p) for p in fast_imgs]
            all_pil = pil_slow + pil_fast

            content = []
            for _ in pil_slow: content.append({"type": "image"})
            content.append({"type": "text", "text": x.get("input_prompt", "")})
            for _ in pil_fast: content.append({"type": "image"})

            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inp = self.processor(text=text, images=all_pil if all_pil else None,
                                 return_tensors="pt", padding=False)
            all_input_ids.append(inp.input_ids[0])
            if "pixel_values" in inp and inp.pixel_values is not None:
                all_pixel_values.append(inp.pixel_values)
                all_grid_thw.append(inp.image_grid_thw)

        # ── Visual FLARE ────────────────────────────────────────────────────
        n_flare_steps = cfg.n_flare_steps if cfg.use_flare else 0
        flare_stride = cfg.frame_stride
        flare_pixel_values, flare_grid_thw = None, None
        if n_flare_steps > 0:
            flare_pil_imgs = []
            for x in batch:
                for k in range(n_flare_steps):
                    fimg = self._load_flare_frame(x, k, n_flare_steps, flare_stride)
                    if fimg is None:
                        slow_paths = x.get("input_image_slow", [])
                        fimg = self._open(slow_paths[0]) if slow_paths else \
                               PIL.Image.new("RGB", self.image_size or (384, 288))
                    flare_pil_imgs.append(fimg)
            finp = self.processor.image_processor(flare_pil_imgs, return_tensors="pt")
            flare_pixel_values = finp.pixel_values.to(torch.bfloat16)
            flare_grid_thw = finp.image_grid_thw

        # ── Tactile-FLARE future: F6 only (no deform future I/O) ────────────
        tflare_f6_tensor = None
        if cfg.use_tactile_flare and cfg.n_tfl_steps > 0:
            S = cfg.n_tfl_steps
            if cfg.use_tactile_vec:
                f6_fut = np.array([x["_tflare_f6"] for x in batch], dtype=np.float32)
                f6_fut = f6_fut.reshape(B, S, -1, 6)
                flat = f6_fut.reshape(B * S, -1)
                normed = self._normalize(flat, self.tacf6_mask, self.tacf6_min, self.tacf6_max)
                tflare_f6_tensor = torch.tensor(
                    normed.reshape(B, S, -1, 6), dtype=torch.bfloat16)
            # F6 only; deform-future path removed.

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
            "tactile_f6s_current": tactile_f6_current,         # [B, nf, 6] or None
            "tactile_deforms_current": tactile_deform_current, # [B, nf, 1, H, W] or None
            "state_raw": state_raw,
            "flare_pixel_values": flare_pixel_values,
            "flare_grid_thw": flare_grid_thw,
            "tflare_f6": tflare_f6_tensor,
            "contact_labels": contact_labels,            # [B, n_fingers]
            "force_labels": force_labels,                # [B, n_fingers]
        }


# ── Checkpoint ──────────────────────────────────────────────────────────────

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
                # tac_aux specific
                "use_tactile_flare": args.use_tactile_flare,
                "n_tfl_tokens_per_step": args.n_tfl_tokens_per_step,
                "n_tfl_steps": args.n_tfl_steps,
                "tactile_flare_stride": args.tactile_flare_stride,
                "tflare_loss_weight": args.tflare_loss_weight,
                "n_fingers": args.n_fingers,
                "contact_loss_weight": args.contact_loss_weight,
                "force_loss_weight": args.force_loss_weight,
                "contact_force_threshold": args.contact_force_threshold,
                "force_scale": args.force_scale,
            }, f, indent=2)

        with open(os.path.join(save_dir, "stats_data.json"), "w") as f:
            json.dump(stats_data, f, indent=2)

    accelerator.wait_for_everyone()
    logger.info(f"Checkpoint {epoch}-{global_step} saved.")


# ── Metrics ─────────────────────────────────────────────────────────────────

class TrainingMetrics:
    _FIELDS = ("total_loss", "action_loss", "flare_loss", "tflare_loss",
               "contact_loss", "force_loss", "contact_acc", "force_mae")

    def __init__(self, device):
        self.n_step = 0
        for f in self._FIELDS:
            setattr(self, f, torch.tensor(0.0, device=device))
        self.world_size = dist.get_world_size()

    def update(self, **kwargs):
        self.n_step += 1
        for f, v in kwargs.items():
            if v is None or not hasattr(self, f):
                continue
            v = v.item() if torch.is_tensor(v) else v
            getattr(self, f).add_(v)

    def get_metric(self, reset=True):
        for f in self._FIELDS:
            dist.all_reduce(getattr(self, f), op=dist.ReduceOp.SUM)
        denom = self.world_size * max(self.n_step, 1)
        m = {f: getattr(self, f).item() / denom for f in self._FIELDS}
        if reset:
            self.n_step = 0
            for f in self._FIELDS:
                getattr(self, f).fill_(0)
        return m


# ── Training loop ───────────────────────────────────────────────────────────

def train(args):
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        wandb.init(project=args.experiment_name, name=args.run_name,
                   config=args, dir=args.log_dir)

    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config["train_batch_size"] = (
        args.train_bsz_per_gpu * dist.get_world_size() * accelerator.gradient_accumulation_steps)
    accelerator.print(
        f"[Diag] gas={accelerator.gradient_accumulation_steps}, "
        f"ws={dist.get_world_size()}, bsz/gpu={args.train_bsz_per_gpu}")

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

    is_stage1 = (args.training_stage == 1)
    has_any_tactile = bool(args.use_tactile_vec or args.use_tactile_deform)
    freeze_tactile = is_stage1 or (not has_any_tactile)

    # ── Register new modules on the outer wrapper ──────────────────────────
    H = model.config.hidden_size
    n_fingers = args.n_fingers

    # Tactile-FLARE queries + projection (unchanged from tflare_gate)
    if args.use_tactile_flare and has_any_tactile:
        K_tac = args.n_tfl_tokens_per_step * args.n_tfl_steps
        model.tactile_flare_queries = nn.Parameter(
            (torch.randn(1, K_tac, H) * 0.02).to(torch.bfloat16))
        tfl_proj = nn.Sequential(
            nn.Linear(H, H), nn.GELU(), nn.Linear(H, H),
        )
        for m_ in tfl_proj.modules():
            if isinstance(m_, nn.Linear):
                nn.init.xavier_uniform_(m_.weight)
                if m_.bias is not None:
                    nn.init.zeros_(m_.bias)
        model.tactile_flare_proj = tfl_proj.to(torch.bfloat16)

    # Contact + force queries and heads (NEW)
    if has_any_tactile:
        model.contact_queries = nn.Parameter(
            (torch.randn(1, n_fingers, H) * 0.02).to(torch.bfloat16))
        model.force_queries = nn.Parameter(
            (torch.randn(1, n_fingers, H) * 0.02).to(torch.bfloat16))

        contact_head = nn.Linear(H, 1)
        force_head   = nn.Linear(H, 1)
        nn.init.xavier_uniform_(contact_head.weight); nn.init.zeros_(contact_head.bias)
        nn.init.xavier_uniform_(force_head.weight);   nn.init.zeros_(force_head.bias)
        model.contact_head = contact_head.to(torch.bfloat16)
        model.force_head   = force_head.to(torch.bfloat16)

    # ── Action-expert init (no-resume only) ────────────────────────────────
    if not args.resume_checkpoint:
        named_params = dict(model.named_parameters())
        for name, param in model.named_parameters():
            if "_action" in name:
                base = name.replace("_action", "")
                if base in named_params:
                    param.data.copy_(named_params[base].data)
        accelerator.print("Action expert initialized from latent expert.")

    # ── Resume from Stage-1 checkpoint ─────────────────────────────────────
    if args.resume_checkpoint:
        ckpt_path = args.resume_checkpoint
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "model.pt")
        resume_sd = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in resume_sd:
            resume_sd = resume_sd["state_dict"]
        model_sd = model.state_dict()
        filtered_sd, skipped = {}, []
        for k, v in resume_sd.items():
            if k in model_sd and model_sd[k].shape != v.shape:
                skipped.append(k)
            else:
                filtered_sd[k] = v
        if skipped:
            accelerator.print(f"Skipped {len(skipped)} keys with shape mismatch (e.g. {skipped[0]})")
        missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
        accelerator.print(f"Resumed: missing={len(missing)}, unexpected={len(unexpected)}")

    # ── Snapshot tactile target encoders (AFTER resume, BEFORE reinit) ─────
    if args.use_tactile_flare and has_any_tactile:
        model.target_tacf6_embedder = copy.deepcopy(model.tacf6_embedder).to(torch.bfloat16)
        model.target_tacf6_embedder.eval()
        for p in model.target_tacf6_embedder.parameters():
            p.requires_grad = False
        if args.use_tactile_deform:
            model.target_deform_proj = copy.deepcopy(model.deform_proj).to(torch.bfloat16)
            model.target_deform_proj.eval()
            for p in model.target_deform_proj.parameters():
                p.requires_grad = False
        accelerator.print("[Tactile-FLARE] Target encoders snapshotted and frozen.")

    # ── Stage-2 reinit of tactile-expert backbone params ───────────────────
    if not freeze_tactile:
        for name, param in model.named_parameters():
            if "_tactile" in name and "target_" not in name:
                if param.ndim >= 2:
                    nn.init.xavier_uniform_(param)
                elif param.ndim == 1:
                    nn.init.zeros_(param)
        # final_layer_tactile is unused in this design; leave its weights alone.

    # ── requires_grad ──────────────────────────────────────────────────────
    for name, param in model.named_parameters():
        if (name.startswith("visual") or name.startswith("deform_encoder")
                or name.startswith("target_")):
            param.requires_grad = False
        elif "_tactile" in name or "final_layer_tactile" in name:
            param.requires_grad = (not freeze_tactile)
        else:
            param.requires_grad = True
    # Unused final_layer_tactile stays disabled to avoid wasted optimizer state.
    for name, param in model.named_parameters():
        if name.startswith("final_layer_tactile"):
            param.requires_grad = False

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
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        # prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
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
    K = T_per_frame * S_steps
    use_flare = bool(args.use_flare and K > 0)
    K_tac = (args.n_tfl_tokens_per_step * args.n_tfl_steps) if args.use_tactile_flare else 0
    use_tflare = bool(args.use_tactile_flare and K_tac > 0 and has_any_tactile)
    flare_layer_idx = args.flare_layer_index

    accelerator.print(
        f"use_flare={use_flare} K_vis={K} | "
        f"use_tflare={use_tflare} K_tac={K_tac} | "
        f"n_fingers={n_fingers} has_tac={has_any_tactile}")

    model.train()

    for epoch in range(args.n_epochs):
        from tqdm import tqdm
        it = (tqdm(dataloader, total=len(dataloader))
              if accelerator.is_main_process else dataloader)

        for batch in it:
            raw_model = accelerator.unwrap_model(model)
            device = torch.cuda.current_device()
            dtype = torch.bfloat16

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
                attention_mask=batch["attention_mask"],
            )
            pos_ids = pos_ids[:, :, :L_slow]

            if use_flare:
                flare_q = raw_model.flare_queries.expand(B, -1, -1).to(device=device, dtype=dtype)
                slow_embeds_ext = torch.cat([slow_embeds, flare_q], dim=1)
                pos_ids = extend_position_ids_for_flare(pos_ids, K)
                L_latent = slow_embeds_ext.shape[1]
            else:
                slow_embeds_ext = slow_embeds
                L_latent = L_slow

            if args.use_robot_state and batch["state_raw"] is not None:
                state_vec = batch["state_raw"].to(device=device, dtype=dtype)
                state_embeds = raw_model.state_embedder(state_vec).unsqueeze(1)
            else:
                state_embeds = torch.empty((B, 0, H), device=device, dtype=dtype)
            n_state = state_embeds.shape[1]

            noisy_actions = raw_model.x_embedder(batch["noisy_actions"].to(dtype))
            timesteps = raw_model.t_embedder(batch["timesteps"].to(dtype)).unsqueeze(1)
            chunk = args.action_chunk
            target = batch["target"].to(dtype)
            n_fast = fast_embeds.shape[1]

            # ── Encode CURRENT-frame tactile (action-block tokens) ─────────
            tac_f6_tok = None
            tac_deform_tok = None
            if has_any_tactile:
                if args.use_tactile_vec and batch["tactile_f6s_current"] is not None:
                    f6_cur = batch["tactile_f6s_current"].to(device=device, dtype=dtype)  # [B, nf, 6]
                    tac_f6_tok = raw_model.tacf6_embedder(f6_cur)                         # [B, nf, H]
                if args.use_tactile_deform and batch["tactile_deforms_current"] is not None:
                    # Current-frame deform only — no history, no pool.
                    df_cur = batch["tactile_deforms_current"].to(device=device, dtype=dtype)
                    Bs, nf_d, C, Hh, Ww = df_cur.shape                      # [B, nf, 1, H, W]
                    with torch.no_grad():
                        df_feats = raw_model.deform_encoder(df_cur.view(-1, C, Hh, Ww))
                    df_feats = df_feats.view(Bs, nf_d, -1)
                    tac_deform_tok = raw_model.deform_proj(df_feats.to(dtype))  # [B, nf, H]

            n_tac_f6 = tac_f6_tok.shape[1] if tac_f6_tok is not None else 0
            n_tac_def = tac_deform_tok.shape[1] if tac_deform_tok is not None else 0
            n_tac_input = n_tac_f6 + n_tac_def

            # ── Assemble action block: [fast | state | tac_f6 | tac_deform | t | x_t] ──
            act_parts = [fast_embeds, state_embeds]
            if tac_f6_tok is not None:     act_parts.append(tac_f6_tok)
            if tac_deform_tok is not None: act_parts.append(tac_deform_tok)
            act_parts.extend([timesteps, noisy_actions])
            action_block = torch.cat(act_parts, dim=1)
            n_action = action_block.shape[1]

            # ── Assemble tactile block: [tflare_q | contact_q | force_q] ───
            tac_block_parts = []
            if use_tflare:
                tac_block_parts.append(
                    raw_model.tactile_flare_queries.expand(B, -1, -1).to(device=device, dtype=dtype))
            if has_any_tactile:
                tac_block_parts.append(
                    raw_model.contact_queries.expand(B, -1, -1).to(device=device, dtype=dtype))
                tac_block_parts.append(
                    raw_model.force_queries.expand(B, -1, -1).to(device=device, dtype=dtype))
            if tac_block_parts:
                tactile_block = torch.cat(tac_block_parts, dim=1)
                n_tactile = tactile_block.shape[1]
            else:
                tactile_block = None
                n_tactile = 0

            # ── Full sequence + routing ────────────────────────────────────
            if tactile_block is not None:
                full_embeds = torch.cat([slow_embeds_ext, action_block, tactile_block], dim=1)
            else:
                full_embeds = torch.cat([slow_embeds_ext, action_block], dim=1)
            L_total = full_embeds.shape[1]

            latent_indexes  = torch.arange(0, L_latent, device=device)
            action_indexes  = torch.arange(L_latent, L_latent + n_action, device=device)
            tactile_indexes = (torch.arange(L_latent + n_action, L_total, device=device)
                               if tactile_block is not None else torch.arange(0, 0, device=device))

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

            # ── Action prediction ──────────────────────────────────────────
            act_chunk_start = L_latent + n_fast + n_state + n_tac_input + 1
            h_act_chunk = hidden[:, act_chunk_start:act_chunk_start + chunk, :]
            v_act = raw_model.final_layer(h_act_chunk)
            loss_act = nn.MSELoss()(v_act, target)

            # ── Tactile-block head extractions ─────────────────────────────
            loss_tflare  = torch.tensor(0.0, device=device)
            loss_contact = torch.tensor(0.0, device=device)
            loss_force   = torch.tensor(0.0, device=device)
            contact_acc_val = torch.tensor(0.0, device=device)
            force_mae_val   = torch.tensor(0.0, device=device)

            if tactile_block is not None:
                tac_start = L_latent + n_action
                cursor = tac_start

                # Tactile-FLARE
                if use_tflare:
                    h_tflare = hidden[:, cursor:cursor + K_tac, :]
                    cursor += K_tac

                    tgt_parts = []
                    if args.use_tactile_vec and batch["tflare_f6"] is not None:
                        f6_fut = batch["tflare_f6"].to(device=device, dtype=dtype)
                        with torch.no_grad():
                            f6_tgt = raw_model.target_tacf6_embedder(f6_fut)
                        tgt_parts.append(f6_tgt)
                    # Note: tflare target is F6-only (no future deform) to keep
                    # I/O low; target_deform_proj is not used here anymore.

                    if tgt_parts:
                        tgt_all = torch.cat(tgt_parts, dim=2)
                        B_, S_, nf_t, H_ = tgt_all.shape
                        flat = tgt_all.view(B_ * S_, nf_t, H_).permute(0, 2, 1)
                        pooled = F.adaptive_avg_pool1d(flat.float(), args.n_tfl_tokens_per_step)
                        pooled = (pooled.permute(0, 2, 1).to(dtype)
                                  .view(B_, S_ * args.n_tfl_tokens_per_step, H_).detach())
                        pred = raw_model.tactile_flare_proj(h_tflare)
                        pred_n = F.normalize(pred.float(), dim=-1)
                        tgt_n  = F.normalize(pooled.float(), dim=-1)
                        loss_tflare = (1.0 - (pred_n * tgt_n).sum(dim=-1)).mean()

                # Contact classification
                h_contact = hidden[:, cursor:cursor + n_fingers, :]
                cursor += n_fingers
                contact_logits = raw_model.contact_head(h_contact).squeeze(-1)  # [B, nf]
                contact_targets = batch["contact_labels"].to(device=device, dtype=torch.float32)
                loss_contact = F.binary_cross_entropy_with_logits(
                    contact_logits.float(), contact_targets, reduction="mean")
                with torch.no_grad():
                    pred_bin = (contact_logits.float() > 0).float()
                    contact_acc_val = (pred_bin == contact_targets).float().mean()

                # Force regression
                h_force = hidden[:, cursor:cursor + n_fingers, :]
                cursor += n_fingers
                force_pred = raw_model.force_head(h_force).squeeze(-1)  # [B, nf]
                force_targets = batch["force_labels"].to(device=device, dtype=dtype)
                loss_force = nn.MSELoss()(force_pred, force_targets)
                with torch.no_grad():
                    force_mae_val = (force_pred.float() - force_targets.float()).abs().mean()

            # ── Visual FLARE (unchanged) ───────────────────────────────────
            loss_flare = torch.tensor(0.0, device=device)
            if use_flare and batch["flare_pixel_values"] is not None:
                if flare_layer_idx == -1:
                    flare_source = hidden
                else:
                    all_hs = outputs.hidden_states
                    n_layers = len(all_hs) - 1
                    layer_i = (n_layers + flare_layer_idx) if flare_layer_idx < 0 else flare_layer_idx
                    layer_i = max(0, min(layer_i, n_layers - 1))
                    flare_source = all_hs[layer_i + 1]

                flare_hidden = flare_source[:, L_slow:L_slow + K, :]
                flare_pred = raw_model.flare_proj(flare_hidden)
                f_pv  = batch["flare_pixel_values"].to(device=device, dtype=dtype)
                f_thw = batch["flare_grid_thw"].to(device=device)

                with torch.no_grad():
                    vit_out = raw_model.visual(f_pv, grid_thw=f_thw)
                    features = vit_out[0] if isinstance(vit_out, (tuple, list)) else vit_out
                    merge = getattr(raw_model.visual, "spatial_merge_size", 2)
                    frame_feats, offset = [], 0
                    for g_thw in f_thw:
                        n_tok = int(g_thw[0] * (g_thw[1] // merge) * (g_thw[2] // merge))
                        frame_tokens = features[offset:offset + n_tok]
                        pooled = F.adaptive_avg_pool1d(
                            frame_tokens.unsqueeze(0).permute(0, 2, 1), T_per_frame,
                        ).permute(0, 2, 1).squeeze(0)
                        frame_feats.append(pooled)
                        offset += n_tok
                    flare_targets = torch.stack(frame_feats).view(B, K, -1)

                pred_n = F.normalize(flare_pred, dim=-1)
                tgt_n  = F.normalize(flare_targets.detach(), dim=-1)
                loss_flare = (1.0 - (pred_n * tgt_n).sum(dim=-1)).mean()

            # ── Total loss ─────────────────────────────────────────────────
            loss = loss_act
            if loss_flare.requires_grad:
                loss = loss + args.flare_loss_weight * loss_flare
            if loss_tflare.requires_grad:
                loss = loss + args.tflare_loss_weight * loss_tflare
            if loss_contact.requires_grad:
                loss = loss + args.contact_loss_weight * loss_contact
            if loss_force.requires_grad:
                loss = loss + args.force_loss_weight * loss_force

            metric.update(
                total_loss=loss, action_loss=loss_act,
                flare_loss=loss_flare, tflare_loss=loss_tflare,
                contact_loss=loss_contact, force_loss=loss_force,
                contact_acc=contact_acc_val, force_mae=force_mae_val,
            )
            accelerator.backward(loss)

            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if (global_step + 1) % max(1, args.log_freq) == 0:
                m = metric.get_metric()
                if accelerator.is_main_process:
                    lr_now = lr_scheduler.get_last_lr()[0]
                    if hasattr(it, "set_postfix"):
                        it.set_postfix(
                            epoch=epoch, step=global_step,
                            loss=f"{m['total_loss']:.4f}",
                            act=f"{m['action_loss']:.4f}",
                            vfl=f"{m['flare_loss']:.4f}",
                            tfl=f"{m['tflare_loss']:.4f}",
                            con=f"{m['contact_loss']:.3f}",
                            frc=f"{m['force_loss']:.3f}",
                            acc=f"{m['contact_acc']:.3f}",
                            lr=f"{lr_now:.2e}",
                        )
                    wandb.log({
                        "total_loss":    m["total_loss"],
                        "action_loss":   m["action_loss"],
                        "flare_loss":    m["flare_loss"],
                        "tflare_loss":   m["tflare_loss"],
                        "contact_loss":  m["contact_loss"],
                        "force_loss":    m["force_loss"],
                        "contact_acc":   m["contact_acc"],
                        "force_mae":     m["force_mae"],
                        "lr": lr_now,
                    }, step=global_step)

            global_step += 1

        if (epoch + 1) % args.save_freq == 0 or epoch == args.n_epochs - 1:
            accelerator.wait_for_everyone()
            save_checkpoint(model, processor, accelerator, args,
                            epoch, global_step, dataset.stats_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="qwen3vl_mot_tac_aux")
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
    parser.add_argument("--log_freq", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=12,
                        help="DataLoader worker count. Tac_aux opens 1280 PNGs/batch "
                             "at T=8,S=8,B=8,nf=10 — 4 workers cannot keep up, raise to 12–16.")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="Batches each worker prefetches ahead.")

    parser.add_argument("--action_dim", type=int, default=31)
    parser.add_argument("--action_chunk", type=int, default=8)
    parser.add_argument("--use_robot_state", type=int, default=0)
    parser.add_argument("--use_tactile_vec", type=int, default=0)
    parser.add_argument("--use_tactile_deform", type=int, default=1)
    parser.add_argument("--deform_encoder_ckpt", type=str, default="")
    parser.add_argument("--tactile_intermediate_size", type=int, default=0)
    parser.add_argument("--training_stage", type=int, default=2, choices=[1, 2])
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("W", "H"))

    # Visual FLARE
    parser.add_argument("--use_flare", type=int, default=1)
    parser.add_argument("--n_flare_tokens_per_frame", type=int, default=4)
    parser.add_argument("--n_flare_steps", type=int, default=8)
    parser.add_argument("--flare_loss_weight", type=float, default=0.5)
    parser.add_argument("--flare_frame_stride", type=int, default=4)
    parser.add_argument("--flare_layer_index", type=int, default=-1)
    parser.add_argument("--frame_stride", type=int, default=2)

    # Tactile-FLARE
    parser.add_argument("--use_tactile_flare", type=int, default=1)
    parser.add_argument("--n_tfl_tokens_per_step", type=int, default=4)
    parser.add_argument("--n_tfl_steps", type=int, default=4)
    parser.add_argument("--tactile_flare_stride", type=int, default=2)
    parser.add_argument("--tflare_loss_weight", type=float, default=0.5)

    # Tactile auxiliary heads
    parser.add_argument("--n_fingers", type=int, default=10,
                        help="Number of tactile fingers (bimanual=10, single=5).")
    parser.add_argument("--contact_loss_weight", type=float, default=0.5)
    parser.add_argument("--force_loss_weight", type=float, default=0.3)
    parser.add_argument("--contact_force_threshold", type=float, default=0.5,
                        help="Raw |F_xyz| threshold for binary contact labels.")
    parser.add_argument("--force_scale", type=float, default=2.0,
                        help="Divisor for force_labels to keep MSE well-scaled.")

    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name, args.run_name)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)
