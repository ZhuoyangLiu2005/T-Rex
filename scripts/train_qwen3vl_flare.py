"""
Qwen3-VL MoT training with flare visual prediction for the latent expert.

Extends train_qwen3vl.py with:
  --use_flare 1                   : enable flare query tokens + cosine similarity loss
  --n_flare_tokens_per_frame T    : number of tokens per future frame (default=4)
  --n_flare_steps S               : number of future steps to predict (default=8)
  --flare_loss_weight w           : weight for flare prediction loss (default=0.5)
  --flare_frame_stride s          : temporal stride for flare frame targets (default=frame_stride)
  --flare_layer_index L           : which layer to extract flare hidden states from
                                    (default=-1, i.e. last layer; use e.g. -7 for ~3/4 depth)

  Total flare tokens = n_flare_tokens_per_frame * n_flare_steps.

When --use_flare 0, this script behaves identically to train_qwen3vl.py.
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
        # Flare frames are loaded by offsetting into the video/image sequence.
        pass  # Flare frames are loaded from the JSON data directly

    def __len__(self):
        return len(self.hf_dataset)

    def create_val_split(self, val_ratio=0.05, seed=42):
        """Split the dataset into train/val. Returns a new SftDataset for val."""
        import copy
        n = len(self.hf_dataset)
        n_val = max(1, int(n * val_ratio))
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        val_indices = sorted(perm[:n_val].tolist())
        train_indices = sorted(perm[n_val:].tolist())

        val_ds = copy.copy(self)
        val_ds.hf_dataset = self.hf_dataset.select(val_indices)

        self.hf_dataset = self.hf_dataset.select(train_indices)

        self.accelerator.print(
            f"Train/Val split: {len(self.hf_dataset)} train, "
            f"{len(val_ds.hf_dataset)} val samples")
        return val_ds

    def __getitem__(self, idx):
        return self.hf_dataset[idx]

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
            try:
                return self._open(flare_path)
            except (PIL.UnidentifiedImageError, OSError):
                return None
        else:
            # Out of episode range → use last available or return None
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

        norm_tacf6 = None
        if cfg.use_tactile_vec:
            tacf6 = np.array([x["tactile_f6"] for x in batch], dtype=np.float32)
            tacf6 = tacf6.reshape(B, -1)
            norm_tacf6 = self._normalize(tacf6, self.tacf6_mask,
                                         self.tacf6_min, self.tacf6_max)
            norm_tacf6 = torch.tensor(norm_tacf6.reshape(B, -1, 6), dtype=torch.bfloat16)

        deforms_tensor = None
        if cfg.use_tactile_deform:
            deforms = []
            for x in batch:
                imgs = [self._open_gray(p) for p in x.get("tactile_image_deform", [])]
                deforms.append(imgs)
            deforms_tensor = torch.tensor(np.array(deforms)).unsqueeze(2)

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
        flare_stride = cfg.flare_frame_stride
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

        # Paradigm C: in post-training the JSON records don't expose frame
        # indices for delayed tactile lookup, so we reuse the current tactile
        # as the "delayed" input.  The asymmetry between midtrain (variable
        # delay_k) and post-training (delay_k=0) is intentional — midtrain
        # teaches the model to be robust to delays; post-training fine-tunes
        # to the small in-lab set.
        time_r = self._beta_sample(B, device_cpu)
        eps_r  = torch.randn_like(norm_actions)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "n_slow_images": n_slow_images,
            "noisy_actions": x_t,
            "target": u_t,
            "timesteps": time,
            "norm_actions": norm_actions,            # for L_refine residual target
            "tactile_f6s": norm_tacf6,
            "tactile_deforms": deforms_tensor,
            "tactile_f6s_delayed": norm_tacf6,       # same as current (delay_k=0)
            "tactile_deforms_delayed": deforms_tensor,
            "time_r": time_r,
            "eps_r": eps_r,
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
            eps_r = batch["eps_r"].to(slow_embeds.dtype)
            tau_r = batch["time_r"].to(slow_embeds.dtype)
            tau_r_b = tau_r[:, None, None]
            r_tau = (1 - tau_r_b) * r_target + tau_r_b * eps_r
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
    model.initialize_vla_weights(skip_tactile_zero_init=bool(args.use_tactile_refine_flow))
    if args.use_flare:
        accelerator.print(
            f"Flare alignment: {args.n_flare_steps} steps × {args.n_flare_tokens_per_frame} tok/frame "
            f"= {model.n_flare_tokens} total tokens, layer_index={args.flare_layer_index}"
        )

    is_stage1 = (args.training_stage == 1)
    has_any_tactile = bool(args.use_tactile_vec or args.use_tactile_deform)
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

    dataset = SftDataset(args, processor, accelerator)

    val_dataloader = None
    if getattr(args, "val_ratio", 0) > 0:
        val_dataset = dataset.create_val_split(
            val_ratio=args.val_ratio, seed=args.seed)
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.train_bsz_per_gpu, shuffle=False,
            drop_last=True, collate_fn=val_dataset.collate_fn,
            num_workers=2, pin_memory=True)

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

    if val_dataloader is not None:
        model, optimizer, dataloader, val_dataloader = accelerator.prepare(
            model, optimizer, dataloader, val_dataloader)
    else:
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
                # 1) L_flow — action expert single-step (tactile-blind)
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

                    norm_actions_gt = batch["norm_actions"].to(slow_embeds.dtype)
                    # Optional jitter on Â: prevents r_target from collapsing
                    # to 0 when the action expert overfits the small in-lab
                    # dataset.  See refine/r_target_norm wandb metric.
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

    parser.add_argument("--experiment_name", type=str, default="qwen3vl_mot_flare")
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
    parser.add_argument("--resume_source", type=str, default="pretrain",
                        choices=["pretrain", "midtrain"],
                        help="'pretrain': resumed ckpt did not train tactile (re-init); "
                             "'midtrain': resumed ckpt already trained tactile (keep).")
    parser.add_argument("--tactile_loss_weight", type=float, default=1.0)
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("W", "H"))

    # action-only slow flow + tactile residual flow refinement
    parser.add_argument("--use_tactile_refine_flow", type=int, default=0,
                        help="1: tactile expert is trained as a residual-flow refinement "
                             "head over the action expert's clean chunk Â. 0: legacy "
                             "Paradigm A (delta_v residual on flow velocity).")
    parser.add_argument("--tactile_refine_loss_weight", type=float, default=1.0, help="Weight on L_refine when use_tactile_refine_flow=1.")
    parser.add_argument("--tactile_refine_noise_scale", type=float, default=0.1, help="Initial noise magnitude for the residual flow at τ=1.")
    parser.add_argument("--action_flow_train_steps", type=int, default=5, help="Number of Euler steps for the no_grad action flow that produces Â during training.")
    parser.add_argument("--tactile_residual_jitter", type=float, default=0.0,
                        help="Std of Gaussian jitter added to Â before computing "
                             "r_target during training. Prevents the residual from "
                             "collapsing to 0 when the action expert overfits and "
                             "Â ≈ A_demo. 0.05 (5%% of normalized range) is a good "
                             "starting point if the refine/r_target_norm metric is "
                             "trending toward 0. 0 (default) disables.")

    # Flare
    parser.add_argument("--use_flare", type=int, default=1, help="Enable flare visual prediction for latent expert.")
    parser.add_argument("--n_flare_tokens_per_frame", type=int, default=4, help="Number of tokens per future frame.")
    parser.add_argument("--n_flare_steps", type=int, default=8, help="Number of future steps to predict.")
    parser.add_argument("--flare_loss_weight", type=float, default=0.5, help="Weight for flare prediction cosine loss.")
    parser.add_argument("--flare_frame_stride", type=int, default=2, help="Temporal stride for flare frame targets.")
    parser.add_argument("--flare_layer_index", type=int, default=-1, help="Layer to extract flare hidden states from (-1=last, e.g. -7 for ~3/4 depth).")

    # Validation
    parser.add_argument("--val_ratio", type=float, default=0.0, help="Fraction of samples for validation (0=disable)")
    parser.add_argument("--val_freq", type=int, default=0, help="Run validation every N steps (0=disable)")
    parser.add_argument("--max_val_batches", type=int, default=50, help="Max batches per validation run")

    args = parser.parse_args()

    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name, args.run_name)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)

