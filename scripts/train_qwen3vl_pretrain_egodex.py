"""
EgoDex large-scale pretraining script for Qwen3-VL VLA (Stage 1, no tactile).

Key design:
  1. Map-style Dataset with a flat (episode, frame) index for balanced multi-node
     sharding via DistributedSampler — every rank processes exactly the same
     number of steps per epoch, preventing NCCL deadlocks.
  2. Reads pretrain.hdf5 (states, action_chunks) + ego_view.mp4 per episode.
  3. Bimanual 62D: left_wrist_9d(9) + left_hand_22d(22) + right_wrist_9d(9) + right_hand_22d(22).
  4. Single ego-view duplicated as both slow and fast image inputs.
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

from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator, DataLoaderConfiguration
from transformers import AutoProcessor, set_seed

from qwen_vla import Qwen3VLVLAModel
from janus.models.action_tokenizer import ActionTokenizer

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

class EgoDexPretrainDataset(Dataset):
    """
    Map-style dataset with a flat (episode_idx, frame_t) index.

    DistributedSampler shards this index evenly across ranks, guaranteeing
    every rank processes the same number of samples per epoch — no NCCL hangs.
    """

    def __init__(self, data_root: str, config, processor, accelerator):
        super().__init__()
        self.config = config
        self.processor = processor
        self.accelerator = accelerator
        self.action_tokenizer = ActionTokenizer(processor.tokenizer)

        # Discover all batch manifests under data_root
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

        # Aggregate stats: element-wise min of q01, max of q99 across batches
        self.action_min = np.min(np.stack(all_action_q01), axis=0)
        self.action_max = np.max(np.stack(all_action_q99), axis=0)
        self.state_min = np.min(np.stack(all_state_q01), axis=0)
        self.state_max = np.max(np.stack(all_state_q99), axis=0)

        self.action_mask = np.ones(config.action_dim, dtype=bool)
        self.state_mask = np.ones(config.action_dim, dtype=bool)

        if config.image_size:
            self.image_size = tuple(config.image_size)  # (W, H)
        else:
            self.image_size = None

        # ── Build flat index: list of (episode_idx, frame_t) ──
        # Trust manifest num_frames; bad episodes handled in __getitem__
        self._index = []
        for ep_idx, ep_info in enumerate(self.episodes):
            num_frames = ep_info["num_frames"]
            for t in range(num_frames):
                self._index.append((ep_idx, t))

        self._total_transitions = len(self._index)
        accelerator.print(f"EgoDex pretrain: {len(self.episodes)} episodes, "
                          f"{self._total_transitions} transitions")

    def __len__(self):
        return len(self._index)

    @property
    def total_transitions(self):
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
        """Find the head/ego-view video in an episode directory.
        Egodex: ego_view.mp4, In-lab: *head*.mp4"""
        candidate = os.path.join(ep_dir, "ego_view.mp4")
        if os.path.isfile(candidate):
            return candidate
        matches = glob.glob(os.path.join(ep_dir, "*head*.mp4"))
        return matches[0] if matches else None

    def __getitem__(self, idx: int) -> Dict:
        ep_idx, frame_t = self._index[idx]
        ep_info = self.episodes[ep_idx]
        ep_dir = ep_info["episode_dir"]
        pretrain_h5 = os.path.join(ep_dir, "pretrain.hdf5")
        video_path = self._find_head_video(ep_dir)

        # Fallback values for bad episodes
        fallback = {
            "frame": np.zeros((288, 384, 3), dtype=np.uint8),
            "state": np.zeros(self.config.action_dim, dtype=np.float32),
            "action_chunk": np.zeros((self.config.action_chunk, self.config.action_dim), dtype=np.float32),
            "language": "",
        }

        if video_path is None or not os.path.isfile(pretrain_h5):
            return fallback

        try:
            with h5py.File(pretrain_h5, "r") as f:
                state = f["states"][frame_t]              # (62,) float32
                action_chunk = f["action_chunks"][frame_t]  # (chunk, 62) float32
                language = f.attrs.get("language", "")
        except Exception:
            return fallback

        # Read video frame via seek
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_t)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            frame_rgb = np.zeros((288, 384, 3), dtype=np.uint8)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return {
            "frame": frame_rgb,
            "state": state,
            "action_chunk": action_chunk,
            "language": language,
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

        # ── Single message: [slow_img | text | fast_img] ──────────────
        # Ego-view is used for both slow (latent) and fast (action).
        all_input_ids = []
        all_pixel_values = []
        all_grid_thw = []
        n_slow_images = 1  # 1 ego-view for latent expert

        for x in batch:
            img = PIL.Image.fromarray(x["frame"])
            if self.image_size is not None:
                img = img.resize(self.image_size, PIL.Image.LANCZOS)

            # [slow_img, text, fast_img] — same ego-view duplicated
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
        pixel_values = (torch.cat(all_pixel_values, dim=0)
                        if all_pixel_values else None)
        image_grid_thw = (torch.cat(all_grid_thw, dim=0)
                          if all_grid_thw else None)

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

        # Save aggregated stats for inference
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
            }, f, indent=2)

    accelerator.wait_for_everyone()
    logger.info(f"Checkpoint {epoch}-{global_step} saved.")

class TrainingMetrics:
    def __init__(self, device):
        self.n_step = 0
        self.action_loss = torch.tensor(0.0, device=device)
        self.total_loss = torch.tensor(0.0, device=device)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    def update(self, total_loss, action_loss):
        self.n_step += 1
        self.total_loss += total_loss.item() if torch.is_tensor(total_loss) else total_loss
        self.action_loss += action_loss.item() if torch.is_tensor(action_loss) else action_loss

    def get_metric(self, reset=True):
        if dist.is_initialized():
            dist.all_reduce(self.total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.action_loss, op=dist.ReduceOp.SUM)
        denom = self.world_size * max(self.n_step, 1)
        metrics = {
            "total_loss": self.total_loss.item() / denom,
            "action_loss": self.action_loss.item() / denom,
        }
        if reset:
            self.n_step = 0
            self.total_loss.fill_(0)
            self.action_loss.fill_(0)
        return metrics


# ───────────────────────────────────────────────────────────────────
#  Main training loop
# ───────────────────────────────────────────────────────────────────
def train(args):
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
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
    )

    model.initialize_vla_weights()
    accelerator.print("VLA weights initialized.")

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

    # Freeze vision + tactile, train latent + action
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

    # Optimizer
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

    # Dataset
    dataset = EgoDexPretrainDataset(args.data_root, args, processor, accelerator)

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

    # Prepare FIRST so accelerate handles distributed sharding,
    # then compute steps_per_epoch from the prepared dataloader.
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # After prepare(), len(dataloader) already reflects per-rank sharding.
    steps_per_epoch = len(dataloader)
    num_training_steps = steps_per_epoch * args.n_epochs // accelerator.gradient_accumulation_steps
    accelerator.print(f"Estimated {steps_per_epoch} steps/epoch "
                      f"(world={accelerator.num_processes}), "
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
        # accelerate's prepared dataloader handles set_epoch internally
        if hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch)

        from tqdm import tqdm
        it = (tqdm(dataloader, total=steps_per_epoch, desc=f"Epoch {epoch}")
              if accelerator.is_main_process else dataloader)

        _debug_printed = False

        for batch in it:
            raw_model = accelerator.unwrap_model(model)

            # ── Single VLM forward: all images in one sequence ───────────
            inputs_embeds = raw_model.prepare_inputs_embeds(
                input_ids      = batch["input_ids"],
                pixel_values   = batch.get("pixel_values"),
                image_grid_thw = batch.get("image_grid_thw"),
            )

            # ── Split into slow (latent) and fast (action) portions ──────
            n_slow_imgs = batch["n_slow_images"]
            grid_thw = batch.get("image_grid_thw")
            B = inputs_embeds.shape[0]
            if grid_thw is not None and grid_thw.shape[0] > n_slow_imgs:
                merge = getattr(raw_model.visual, "spatial_merge_size",
                                getattr(dataset.processor.image_processor, "merge_size", 2))
                n_imgs_per_sample = grid_thw.shape[0] // B
                fast_grid = grid_thw[n_slow_imgs : n_imgs_per_sample]
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

            # ── M-RoPE: compute for full VLM sequence, truncate to slow ──
            pos_ids, _ = raw_model.get_rope_index(
                input_ids      = batch["input_ids"],
                image_grid_thw = batch.get("image_grid_thw"),
                attention_mask = batch["attention_mask"],
            )
            pos_ids = pos_ids[:, :, :L_slow]

            if args.use_robot_state and batch["state_raw"] is not None:
                state_vec = batch["state_raw"].to(slow_embeds.device, dtype=slow_embeds.dtype)
                state_embeds = raw_model.state_embedder(state_vec).unsqueeze(1)
            else:
                state_embeds = torch.empty(
                    (slow_embeds.shape[0], 0, slow_embeds.shape[2]),
                    device=slow_embeds.device, dtype=slow_embeds.dtype)
            n_state = state_embeds.shape[1]

            noisy_actions = raw_model.x_embedder(
                batch["noisy_actions"].to(slow_embeds.dtype))
            timesteps = raw_model.t_embedder(
                batch["timesteps"].to(slow_embeds.dtype)).unsqueeze(1)

            chunk = args.action_chunk
            target = batch["target"].to(slow_embeds.dtype)

            # Stage 1: latent + action expert only
            # Layout: [slow_embeds | fast_embeds, state?, timestep, noisy_actions]
            full_embeds = torch.cat([
                slow_embeds,
                fast_embeds, state_embeds, timesteps, noisy_actions,
            ], dim=1)

            n_fast = fast_embeds.shape[1]
            n_action = n_fast + n_state + 1 + chunk
            L_total = full_embeds.shape[1]
            latent_indexes  = torch.arange(0, L_slow, device=full_embeds.device)
            action_indexes  = torch.arange(L_slow, L_total, device=full_embeds.device)
            tactile_indexes = torch.arange(0, 0, device=full_embeds.device)

            if global_step == 0 and accelerator.is_main_process:
                print(f"\n[Layout] slow={slow_embeds.shape[1]} fast={n_fast} "
                      f"state={n_state} t=1 act={chunk} total={L_total}")

            outputs = model.model(
                inputs_embeds   = full_embeds,
                position_ids    = pos_ids,
                attention_mask  = batch["attention_mask"],
                use_cache       = False,
                latent_indexes  = latent_indexes,
                action_indexes  = action_indexes,
                tactile_indexes = tactile_indexes,
            )
            hidden = outputs.last_hidden_state

            act_pred_start = L_slow + n_fast + n_state + 1
            v_act = raw_model.final_layer(
                hidden[:, act_pred_start: act_pred_start + chunk, :])
            loss = nn.MSELoss()(v_act, target)

            metric.update(loss, loss)
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
                                       lr=f"{lr_now:.2e}")
                    wandb.log({
                        "total_loss": m["total_loss"],
                        "action_loss": m["action_loss"],
                        "lr": lr_now,
                        "epoch": epoch,
                    }, step=global_step)

            global_step += 1

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == args.n_epochs - 1:
            accelerator.wait_for_everyone()
            save_checkpoint(model, processor, accelerator, args,
                            epoch, global_step, dataset)

    accelerator.print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="qwen3vl_egodex_pretrain")
    parser.add_argument("--run_name", type=str, default="run_1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root dir containing batch subdirs, each with pretrain_manifest.json")
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
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("W", "H"),
                        help="Resize ego view to W H before tokenization. "
                             "E.g. --image_size 384 384. Default: no resize.")

    parser.add_argument("--resume_checkpoint", type=str, default="")

    args = parser.parse_args()

    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name, args.run_name)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)
