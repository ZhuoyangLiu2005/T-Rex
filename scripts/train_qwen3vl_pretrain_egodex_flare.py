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
import shutil
import logging
import argparse

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
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator, DataLoaderConfiguration, InitProcessGroupKwargs
from transformers import AutoProcessor, set_seed

from qwen_vla import Qwen3VLVLAModel, extend_position_ids_for_flare

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                     min_lr_ratio=0.0, num_cycles=0.5):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
        return (1 - min_lr_ratio) * cosine + min_lr_ratio
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


class EgoDexPretrainFlareDataset(Dataset):
    """
    Map-style dataset with a flat (episode_idx, frame_t) index.

    Compared to the base EgoDexPretrainDataset, __getitem__ also returns
    future frames for flare visual prediction, read from the same video.
    """

    def __init__(self, data_root: str, config, processor, accelerator):
        super().__init__()
        self.config = config
        self.processor = processor
        self.accelerator = accelerator

        self.episodes = []
        all_action_q01 = []
        all_action_q99 = []
        all_state_q01 = []
        all_state_q99 = []

        manifest_paths = sorted(
            glob.glob(os.path.join(data_root, "*", "pretrain_manifest.json"))
        )
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

        accelerator.print(f"Loaded {len(manifest_paths)} batch manifests")

        self.action_min = np.min(np.stack(all_action_q01), axis=0)
        self.action_max = np.max(np.stack(all_action_q99), axis=0)
        self.state_min = np.min(np.stack(all_state_q01), axis=0)
        self.state_max = np.max(np.stack(all_state_q99), axis=0)

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

        # Build flat index, optionally splitting train/val by episode
        self.val_ratio = getattr(config, "val_ratio", 0.0)
        self.is_val = False  # set by create_val_split()

        self._index = []
        for ep_idx, ep_info in enumerate(self.episodes):
            num_frames = ep_info["num_frames"]
            for t in range(num_frames):
                self._index.append((ep_idx, t))

        self._total_transitions = len(self._index)
        accelerator.print(f"EgoDex pretrain (flare): {len(self.episodes)} episodes, "
                          f"{self._total_transitions} transitions, "
                          f"flare={self.n_flare_steps} steps x stride {self.flare_frame_stride}")

    def create_val_split(self, val_ratio=0.02, seed=42):
        """
        Split episodes into train/val. Returns a new dataset object for val.
        Modifies self in-place to keep only train episodes.
        """
        rng = np.random.RandomState(seed)
        n_ep = len(self.episodes)
        n_val = max(1, int(n_ep * val_ratio))
        perm = rng.permutation(n_ep)
        val_ep_indices = set(perm[:n_val].tolist())
        train_ep_indices = set(perm[n_val:].tolist())

        # Build val dataset (shallow copy with different index)
        import copy
        val_ds = copy.copy(self)
        val_ds.is_val = True
        val_ds._index = [(ep_idx, t) for ep_idx, t in self._index if ep_idx in val_ep_indices]
        val_ds._total_transitions = len(val_ds._index)

        # Trim self to train only
        self._index = [(ep_idx, t) for ep_idx, t in self._index if ep_idx in train_ep_indices]
        self._total_transitions = len(self._index)

        self.accelerator.print(
            f"Train/Val split: {len(train_ep_indices)} train eps ({self._total_transitions} frames), "
            f"{len(val_ep_indices)} val eps ({val_ds._total_transitions} frames)")
        return val_ds

    def __len__(self):
        return len(self._index)

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

    def _read_video_frames(self, video_path, frame_indices):
        """
        Read multiple frames by seeking once then decoding forward sequentially.
        Falls back to per-frame seeking if sequential read fails.
        """
        if not frame_indices:
            return []

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return [None] * len(frame_indices)

            need_set = set(frame_indices)
            collected = {}
            first = min(frame_indices)
            last = max(frame_indices)

            # If the range is too large (>200 frames), fall back to per-frame seek
            # to avoid decoding hundreds of unneeded frames
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
        fallback = {
            "frame": fallback_frame,
            "state": np.zeros(self.config.action_dim, dtype=np.float32),
            "action_chunk": np.zeros((self.config.action_chunk, self.config.action_dim), dtype=np.float32),
            "language": "",
            "flare_frames": [fallback_frame.copy() for _ in range(self.n_flare_steps)],
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

        # Build list of frame indices to read: [current, future_1, ..., future_S]
        frame_indices = [frame_t]
        for k in range(self.n_flare_steps):
            future_t = frame_t + (k + 1) * self.flare_frame_stride
            # Clamp to last frame if out of range
            future_t = min(future_t, num_frames - 1)
            frame_indices.append(future_t)

        # Read all frames in one VideoCapture session
        all_frames = self._read_video_frames(video_path, frame_indices)

        current_frame = all_frames[0]
        if current_frame is None:
            current_frame = fallback_frame

        flare_frames = []
        for k in range(self.n_flare_steps):
            ff = all_frames[k + 1]
            if ff is None:
                ff = current_frame.copy()
            flare_frames.append(ff)

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


def save_checkpoint(model, processor, accelerator, args, epoch, global_step, dataset):
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

    accelerator.wait_for_everyone()
    logger.info(f"Checkpoint {epoch}-{global_step} saved.")


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
            fast_grid = grid_thw[n_slow_imgs: n_imgs_per_sample]
            n_fast_tokens = sum(int(g[0] * (g[1] // merge) * (g[2] // merge)) for g in fast_grid)
            slow_embeds = inputs_embeds[:, :-n_fast_tokens]
            fast_embeds = inputs_embeds[:, -n_fast_tokens:]
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
        resume_sd = torch.load(args.resume_checkpoint, map_location="cpu")
        if "state_dict" in resume_sd:
            resume_sd = resume_sd["state_dict"]
        missing, unexpected = model.load_state_dict(resume_sd, strict=False)
        accelerator.print(f"Resumed: missing={len(missing)}, unexpected={len(unexpected)}")

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

    dataloader = DataLoader(
        dataset,
        batch_size=args.train_bsz_per_gpu,
        shuffle=True,
        drop_last=True,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    if val_dataloader is not None:
        model, optimizer, dataloader, val_dataloader = accelerator.prepare(
            model, optimizer, dataloader, val_dataloader)
    else:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    steps_per_epoch = len(dataloader)
    num_training_steps = steps_per_epoch * args.n_epochs // accelerator.gradient_accumulation_steps
    accelerator.print(f"Estimated {steps_per_epoch} steps/epoch, "
                      f"{num_training_steps} total training steps")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_rates * num_training_steps),
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    metric = TrainingMetrics(device=torch.cuda.current_device())
    global_step = 0
    model.train()

    for epoch in range(args.n_epochs):
        if hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch)

        from tqdm import tqdm
        it = (tqdm(dataloader, total=steps_per_epoch, desc=f"Epoch {epoch}")
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
                                getattr(dataset.processor.image_processor, "merge_size", 2))
                n_imgs_per_sample = grid_thw.shape[0] // B
                fast_grid = grid_thw[n_slow_imgs: n_imgs_per_sample]
                n_fast_tokens = sum(
                    int(g[0] * (g[1] // merge) * (g[2] // merge))
                    for g in fast_grid
                )
                slow_embeds = inputs_embeds[:, :-n_fast_tokens]
                fast_embeds = inputs_embeds[:, -n_fast_tokens:]
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

            global_step += 1

        if (epoch + 1) % args.save_freq == 0 or epoch == args.n_epochs - 1:
            accelerator.wait_for_everyone()
            save_checkpoint(model, processor, accelerator, args,
                            epoch, global_step, dataset)

    accelerator.print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="qwen3vl_egodex_pretrain_flare")
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

    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name, args.run_name)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)
