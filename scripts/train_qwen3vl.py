import os, sys

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)  # dex_mot_qwen/
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import json
import torch
import logging
import argparse
import shutil
import math
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

from qwen_vla import Qwen3VLVLAModel

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                     min_lr_ratio=0.0, num_cycles=0.5):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine  = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
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

        ds = next(iter(self.stats_data))
        self.dataset_name = ds

        def _arr(key, sub):
            return np.array(self.stats_data[ds][key][sub])

        self.action_mask = _arr("action", "mask")
        self.action_min = _arr("action", "q01")
        self.action_max = _arr("action", "q99")
        self.state_mask = _arr("state",  "mask")
        self.state_min = _arr("state",  "q01")
        self.state_max = _arr("state",  "q99")
        self.tacf6_mask = _arr("tactile_f6", "mask")
        self.tacf6_min = _arr("tactile_f6", "q01")
        self.tacf6_max = _arr("tactile_f6", "q99")

        tracking_info = self.stats_data[ds].get("tracking_error", {})
        self.tracking_err_mean = np.array(tracking_info["mean"], dtype=np.float32) \
            if "mean" in tracking_info else None
        self.tracking_err_std  = np.array(tracking_info["std"],  dtype=np.float32) \
            if "std"  in tracking_info else None

        self.img_dir = os.path.dirname(config.data_path)

        if config.image_size:
            w, h = config.image_size
            self.image_size = (w, h)
            accelerator.print(f"Resizing RGB images to {w}x{h}")
        else:
            self.image_size = None

        accelerator.print(f"Dataset size: {len(self.hf_dataset)}")

    def __len__(self):
        return len(self.hf_dataset)

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
        img  = PIL.Image.open(full).convert("L")
        return np.array(img, dtype=np.float32) / 255.0

    def _beta_sample(self, n, device):
        d = torch.distributions.Beta(
            torch.tensor(1.5, dtype=torch.float32, device=device),
            torch.tensor(1.0, dtype=torch.float32, device=device),
        )
        t = d.sample((n,)) * 0.999 + 0.001
        return t.to(torch.bfloat16)

    def collate_fn(self, batch: List[Dict]) -> Dict:
        cfg = self.config
        B = len(batch)

        actions = np.array([x["action"] for x in batch], dtype=np.float32)
        actions = actions.reshape(B, -1, cfg.action_dim)
        norm_actions = self._normalize(actions, self.action_mask, self.action_min, self.action_max)
        norm_actions = torch.tensor(norm_actions, dtype=torch.bfloat16) # [B, chunk, dim]

        device_cpu = norm_actions.device
        time = self._beta_sample(B, device_cpu) # [B]
        t_ = time[:, None, None]
        noise = torch.randn_like(norm_actions)
        x_t = t_ * noise + (1 - t_) * norm_actions
        u_t = noise - norm_actions

        norm_tacf6 = None
        if cfg.use_tactile_vec:
            tacf6 = np.array([x["tactile_f6"] for x in batch], dtype=np.float32)
            tacf6 = tacf6.reshape(B, -1)  # [B, 30] — flat for stats broadcasting
            norm_tacf6 = self._normalize(tacf6, self.tacf6_mask, self.tacf6_min, self.tacf6_max)
            norm_tacf6 = torch.tensor(norm_tacf6.reshape(B, -1, 6), dtype=torch.bfloat16)

        deforms_tensor = None
        if cfg.use_tactile_deform:
            deforms = []
            for x in batch:
                imgs = [self._open_gray(p) for p in x.get("tactile_image_deform", [])]
                deforms.append(imgs)
            deforms_tensor = torch.tensor(np.array(deforms)).unsqueeze(2)  # [B,5,1,H,W]

        has_tactile = cfg.use_tactile_vec or cfg.use_tactile_deform

        state_raw_list = []
        if cfg.use_robot_state:
            for x in batch:
                state_raw = np.array(x["state_fast"], dtype=np.float32)
                norm_state = self._normalize(state_raw, self.state_mask,
                                             self.state_min, self.state_max)
                state_raw_list.append(torch.tensor(norm_state, dtype=torch.bfloat16))

        all_input_ids = []
        all_pixel_values = []
        all_grid_thw = []

        for x in batch:
            slow_imgs = x.get("input_image_slow", [])
            fast_imgs = x.get("input_image_fast", [])

            pil_imgs_slow = [self._open(p) for p in slow_imgs]
            pil_imgs_fast = [self._open(p) for p in fast_imgs]

            content = []
            for _ in pil_imgs_slow:
                content.append({"type": "image"})
            content.append({"type": "text", "text": x.get("input_prompt", "")})
            for _ in pil_imgs_fast:
                content.append({"type": "image"})

            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_pil = pil_imgs_slow + pil_imgs_fast

            inp = self.processor(
                text=text,
                images=all_pil if all_pil else None,
                return_tensors="pt",
                padding=False,
            )
            all_input_ids.append(inp.input_ids[0]) # [L_i]
            if "pixel_values" in inp and inp.pixel_values is not None:
                all_pixel_values.append(inp.pixel_values) # [patches_i, ...]
                all_grid_thw.append(inp.image_grid_thw) # [n_imgs_i, 3]

        max_len = max(ids.shape[0] for ids in all_input_ids)
        pad_id  = self.processor.tokenizer.pad_token_id or 0

        padded_ids   = []
        attention_ms = []
        for ids in all_input_ids:
            pad_len = max_len - ids.shape[0]
            padded_ids.append(F.pad(ids, (pad_len, 0), value=pad_id))
            attn = torch.ones(max_len, dtype=torch.long)
            if pad_len > 0:
                attn[:pad_len] = 0
            attention_ms.append(attn)

        input_ids = torch.stack(padded_ids) # [B, max_len]
        attention_mask = torch.stack(attention_ms) # [B, max_len]

        pixel_values = (torch.cat(all_pixel_values, dim=0)
                        if all_pixel_values else None)
        image_grid_thw = (torch.cat(all_grid_thw, dim=0)
                          if all_grid_thw else None)

        state_raw = (torch.stack(state_raw_list)
                     if state_raw_list else None)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "noisy_actions": x_t, # [B, chunk, dim]
            "target": u_t, # [B, chunk, dim]
            "timesteps": time, # [B]
            "tactile_f6s": norm_tacf6,
            "tactile_deforms": deforms_tensor,
            "state_raw": state_raw, # [B, action_dim] or None
        }

def save_checkpoint(model, processor, accelerator, args, epoch, global_step, stats_data):
    save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")

    if accelerator.is_main_process:
        ckpts = [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint-")]
        if args.max_ckpts > 0 and len(ckpts) >= args.max_ckpts:
            oldest = min(ckpts, key=lambda f: os.path.getctime(os.path.join(args.output_dir, f)))
            shutil.rmtree(os.path.join(args.output_dir, oldest))

        os.makedirs(save_dir, exist_ok=True)

        # Save model weights
        unwrapped = accelerator.unwrap_model(model)
        sd = accelerator.get_state_dict(model)
        torch.save(sd, os.path.join(save_dir, "model.pt"))

        # Save processor (tokenizer / image processor)
        processor.save_pretrained(os.path.join(save_dir, "processor"))

        # Save base model config for architecture reconstruction at inference
        src_config = os.path.join(args.model_path, "config.json")
        if os.path.exists(src_config):
            shutil.copy(src_config, os.path.join(save_dir, "config.json"))
        else:
            logger.warning(f"config.json not found at {src_config} — inference will "
                           f"need --base_model_path to reconstruct the architecture.")

        # Save training args so inference can auto-detect base model and settings
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
        self.total_loss   = torch.tensor(0.0, device=device)
        self.world_size   = dist.get_world_size()

    def update(self, total_loss, action_loss, tactile_loss=0.0):
        self.n_step += 1
        self.total_loss   += total_loss.item() if torch.is_tensor(total_loss) else total_loss
        self.action_loss  += action_loss.item() if torch.is_tensor(action_loss) else action_loss
        self.tactile_loss += tactile_loss.item() if torch.is_tensor(tactile_loss) else tactile_loss

    def get_metric(self, reset=True):
        dist.all_reduce(self.total_loss,   op=dist.ReduceOp.SUM)
        dist.all_reduce(self.action_loss,  op=dist.ReduceOp.SUM)
        dist.all_reduce(self.tactile_loss, op=dist.ReduceOp.SUM)
        denom = self.world_size * max(self.n_step, 1)
        metrics = {
            "total_loss":   self.total_loss.item()   / denom,
            "action_loss":  self.action_loss.item()  / denom,
            "tactile_loss": self.tactile_loss.item() / denom,
        }
        if reset:
            self.n_step = 0
            self.total_loss.fill_(0)
            self.action_loss.fill_(0)
            self.tactile_loss.fill_(0)
        return metrics


def train(args):
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        wandb.init(project=args.experiment_name, name=args.run_name,
                   config=args, dir=args.log_dir)

    accelerator.state.deepspeed_plugin.deepspeed_config[
        "train_micro_batch_size_per_gpu"] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config[
        "train_batch_size"] = (args.train_bsz_per_gpu
                               * dist.get_world_size()
                               * accelerator.gradient_accumulation_steps)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    tac_isize = args.tactile_intermediate_size if args.tactile_intermediate_size > 0 else None
    model = Qwen3VLVLAModel.from_pretrained_qwen3vl(
        pretrained_path = args.model_path,
        action_dim = args.action_dim,
        action_chunk = args.action_chunk,
        use_tactile_deform = bool(args.use_tactile_deform),
        use_robot_state = bool(args.use_robot_state),
        torch_dtype = torch.bfloat16,
        tactile_intermediate_size = tac_isize,
    )

    if args.use_tactile_deform:
        model.load_deform_encoder_weights(args.deform_encoder_ckpt)

    model.initialize_vla_weights()
    accelerator.print("VLA weights initialized (Xavier + zero-init final layers).")

    is_stage1 = (args.training_stage == 1)
    has_any_tactile = bool(args.use_tactile_vec or args.use_tactile_deform)
    freeze_tactile = is_stage1 or (not has_any_tactile)

    if not args.resume_checkpoint:
        named_params = dict(model.named_parameters())
        for name, param in model.named_parameters():
            if "_action" in name:
                base = name.replace("_action", "")
                if base in named_params:
                    param.data.copy_(named_params[base].data)
        accelerator.print("Action expert initialized from latent expert (VLM weights).")

    if args.resume_checkpoint:
        resume_sd = torch.load(args.resume_checkpoint, map_location="cpu")
        if "state_dict" in resume_sd:
            resume_sd = resume_sd["state_dict"]
        missing, unexpected = model.load_state_dict(resume_sd, strict=False)
        accelerator.print(f"Resumed from {args.resume_checkpoint}: "
                          f"missing={len(missing)}, unexpected={len(unexpected)}")

    if not freeze_tactile:
        for name, param in model.named_parameters():
            if "_tactile" in name:
                if param.ndim >= 2:
                    nn.init.xavier_uniform_(param)
                elif param.ndim == 1:
                    nn.init.zeros_(param)
        nn.init.zeros_(model.final_layer_tactile.mlp.fc2.weight)
        if model.final_layer_tactile.mlp.fc2.bias is not None:
            nn.init.zeros_(model.final_layer_tactile.mlp.fc2.bias)
        accelerator.print("Tactile expert freshly initialized (Xavier + zero final layer).")

    for name, param in model.named_parameters():
        if name.startswith("visual") or name.startswith("deform_encoder"):
            param.requires_grad = False
        elif "_tactile" in name or "final_layer_tactile" in name:
            param.requires_grad = (not freeze_tactile)
        else:
            param.requires_grad = True

    if freeze_tactile:
        accelerator.print(f"[{'Stage 1' if is_stage1 else 'No tactile'}] "
                          f"Tactile expert + final_layer_tactile FROZEN.")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Total: {total/1e9:.2f}B  Trainable: {trainable/1e9:.2f}B "
                      f"({trainable/total*100:.1f}%)")

    if accelerator.is_main_process:
        buckets = {"visual": 0, "latent_attn": 0, "latent_mlp": 0, "latent_norm": 0,
                   "action_attn": 0, "action_mlp": 0, "action_norm": 0,
                   "tactile_attn": 0, "tactile_mlp": 0, "tactile_norm": 0,
                   "vla_heads": 0, "other": 0}
        for n, p in model.named_parameters():
            cnt = p.numel()
            if n.startswith("visual") or n.startswith("deform_encoder"):
                buckets["visual"] += cnt
            elif any(k in n for k in ("x_embedder", "t_embedder", "final_layer",
                                       "tacf6_embedder", "deform_proj", "state_embedder")):
                buckets["vla_heads"] += cnt
            elif "_tactile" in n:
                if "mlp" in n:    buckets["tactile_mlp"] += cnt
                elif "norm" in n: buckets["tactile_norm"] += cnt
                else:             buckets["tactile_attn"] += cnt
            elif "_action" in n:
                if "mlp" in n:    buckets["action_mlp"] += cnt
                elif "norm" in n: buckets["action_norm"] += cnt
                else:             buckets["action_attn"] += cnt
            elif "self_attn" in n or "rotary" in n:
                buckets["latent_attn"] += cnt
            elif "mlp" in n and "_action" not in n and "_tactile" not in n:
                buckets["latent_mlp"] += cnt
            elif "norm" in n or "embed_tokens" in n:
                buckets["latent_norm"] += cnt
            else:
                buckets["other"] += cnt
        print("\n" + "=" * 60)
        print("  Per-Expert Parameter Breakdown")
        print("=" * 60)
        for k, v in buckets.items():
            if v > 0:
                print(f"  {k:20s}: {v/1e6:10.2f}M")
        print(f"  {'TOTAL':20s}: {sum(buckets.values())/1e6:10.2f}M")
        print("=" * 60 + "\n")

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
    model.train()

    for epoch in range(args.n_epochs):
        from tqdm import tqdm
        it = (tqdm(dataloader, total=len(dataloader))
              if accelerator.is_main_process else dataloader)

        for batch in it:
            raw_model = accelerator.unwrap_model(model)

            inputs_embeds = raw_model.prepare_inputs_embeds(
                input_ids = batch["input_ids"],
                pixel_values = batch.get("pixel_values"),
                image_grid_thw = batch.get("image_grid_thw"),
            ) # [B, L, H]

            pos_ids, _ = raw_model.get_rope_index(
                input_ids = batch["input_ids"],
                image_grid_thw = batch.get("image_grid_thw"),
                attention_mask = batch["attention_mask"],
            )

            if args.use_robot_state and batch["state_raw"] is not None:
                state_vec = batch["state_raw"].to(inputs_embeds.device,
                                                   dtype=inputs_embeds.dtype)  # [B, action_dim]
                state_embeds = raw_model.state_embedder(state_vec).unsqueeze(1)  # [B, 1, H]
            else:
                state_embeds = torch.empty(
                    (inputs_embeds.shape[0], 0, inputs_embeds.shape[2]),
                    device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            n_state = state_embeds.shape[1]  # 1 when use_robot_state, 0 otherwise

            noisy_actions = raw_model.x_embedder(
                batch["noisy_actions"].to(inputs_embeds.dtype)) # [B, chunk, H]
            timesteps = raw_model.t_embedder(
                batch["timesteps"].to(inputs_embeds.dtype)).unsqueeze(1) # [B, 1, H]

            chunk = args.action_chunk
            target = batch["target"].to(inputs_embeds.dtype)   # [B, chunk, dim]

            if is_stage1:
                # ── Stage 1: latent + action expert only ─────────────────────
                full_embeds = torch.cat([
                    inputs_embeds,
                    state_embeds, timesteps, noisy_actions,
                ], dim=1)

                n_action = n_state + 1 + chunk
                L_latent = inputs_embeds.shape[1]
                L_total  = full_embeds.shape[1]
                latent_indexes  = torch.arange(0, L_latent, device=full_embeds.device)
                action_indexes  = torch.arange(L_latent, L_total, device=full_embeds.device)
                tactile_indexes = torch.arange(0, 0, device=full_embeds.device)

                outputs = model.model(
                    inputs_embeds  = full_embeds,
                    position_ids   = pos_ids,
                    attention_mask = batch["attention_mask"],
                    use_cache      = False,
                    latent_indexes = latent_indexes,
                    action_indexes = action_indexes,
                    tactile_indexes = tactile_indexes,
                )
                hidden = outputs.last_hidden_state

                act_pred_start = L_latent + n_state + 1
                v_act = raw_model.final_layer(
                    hidden[:, act_pred_start : act_pred_start + chunk, :])
                loss_act = nn.MSELoss()(v_act, target)
                loss_tac = 0.0
                loss = loss_act

            else:
                # ── Stage 2: full sequence with residual tactile correction ──

                # Build tactile_embeds: [f6_tokens, deform_tokens] (concat when both on)
                tac_parts = []
                if args.use_tactile_vec and batch["tactile_f6s"] is not None:
                    tac_parts.append(raw_model.tacf6_embedder(
                        batch["tactile_f6s"].to(inputs_embeds.dtype)))
                if args.use_tactile_deform and batch["tactile_deforms"] is not None:
                    deforms = batch["tactile_deforms"].to(
                        inputs_embeds.device, dtype=inputs_embeds.dtype)
                    Bs, nf, C, H, W = deforms.shape
                    with torch.no_grad():
                        feats = raw_model.deform_encoder(deforms.view(-1, C, H, W))
                    feats = feats.view(Bs, nf, -1)
                    tac_parts.append(raw_model.deform_proj(
                        feats.to(inputs_embeds.dtype)))

                if tac_parts:
                    tactile_embeds = torch.cat(tac_parts, dim=1)
                else:
                    tactile_embeds = torch.empty(
                        (inputs_embeds.shape[0], 0, inputs_embeds.shape[2]),
                        device=inputs_embeds.device, dtype=inputs_embeds.dtype)

                has_tactile = tactile_embeds.shape[1] > 0

                if has_tactile:
                    noisy_actions_tac = raw_model.x_embedder(
                        batch["noisy_actions"].to(inputs_embeds.dtype))
                    timesteps_tac = raw_model.t_embedder(
                        batch["timesteps"].to(inputs_embeds.dtype)).unsqueeze(1)
                    full_embeds = torch.cat([
                        inputs_embeds,
                        state_embeds, timesteps, noisy_actions,
                        tactile_embeds, timesteps_tac, noisy_actions_tac,
                    ], dim=1)
                else:
                    # No tactile tokens — same layout as stage 1
                    full_embeds = torch.cat([
                        inputs_embeds,
                        state_embeds, timesteps, noisy_actions,
                    ], dim=1)

                n_action  = n_state + 1 + chunk
                L_latent = inputs_embeds.shape[1]
                L_total  = full_embeds.shape[1]
                latent_indexes  = torch.arange(0, L_latent, device=full_embeds.device)
                action_indexes  = torch.arange(L_latent, L_latent + n_action,
                                               device=full_embeds.device)
                if has_tactile:
                    tactile_indexes = torch.arange(L_latent + n_action, L_total,
                                                   device=full_embeds.device)
                else:
                    tactile_indexes = torch.arange(0, 0, device=full_embeds.device)

                outputs = model.model(
                    inputs_embeds  = full_embeds,
                    position_ids   = pos_ids,
                    attention_mask = batch["attention_mask"],
                    use_cache      = False,
                    latent_indexes = latent_indexes,
                    action_indexes = action_indexes,
                    tactile_indexes = tactile_indexes,
                )
                hidden = outputs.last_hidden_state

                # Action expert velocity (from action block's noisy_action positions)
                act_pred_start = L_latent + n_state + 1
                v_act = raw_model.final_layer(
                    hidden[:, act_pred_start : act_pred_start + chunk, :])
                loss_act = nn.MSELoss()(v_act, target)

                if has_tactile:
                    # Tactile expert residual (from tactile block's last chunk positions)
                    delta_v = raw_model.final_layer_tactile(hidden[:, -chunk:, :])
                    residual_target = target - v_act.detach()
                    loss_tac = nn.MSELoss()(delta_v, residual_target)
                else:
                    loss_tac = 0.0

                loss = loss_act + args.tactile_loss_weight * loss_tac

            metric.update(loss, loss_act, loss_tac)

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
                                       lr=f"{lr_now:.2e}")
                    wandb.log({
                        "total_loss":   m["total_loss"],
                        "action_loss":  m["action_loss"],
                        "tactile_loss": m["tactile_loss"],
                        "lr": lr_now,
                    }, step=global_step)

            global_step += 1

        if (epoch + 1) % args.save_freq == 0 or epoch == args.n_epochs - 1:
            accelerator.wait_for_everyone()
            save_checkpoint(model, processor, accelerator, args,
                            epoch, global_step, dataset.stats_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="qwen3vl_mot")
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

    parser.add_argument("--action_dim", type=int, default=29)
    parser.add_argument("--action_chunk", type=int, default=8)
    parser.add_argument("--use_robot_state", type=int, default=1)
    parser.add_argument("--use_tactile_vec", type=int, default=0)
    parser.add_argument("--use_tactile_deform", type=int, default=1)
    parser.add_argument("--deform_encoder_ckpt", type=str, default="")
    parser.add_argument("--tactile_intermediate_size", type=int, default=0,
                        help="Intermediate size for tactile expert MLP. "
                             "0 = same as latent (config.intermediate_size).")
    parser.add_argument("--training_stage", type=int, default=2, choices=[1, 2],
                        help="1 = pretrain (latent+action only, no tactile). "
                             "2 = full training with residual tactile correction.")
    parser.add_argument("--resume_checkpoint", type=str, default="",
                        help="Path to model.pt from a previous stage to resume from.")
    parser.add_argument("--tactile_loss_weight", type=float, default=1.0,
                        help="Weight for tactile residual loss in stage 2.")
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("W", "H"),
                        help="Resize RGB images to W H before tokenization. "
                             "E.g. --image_size 384 288. Default: no resize.")

    args = parser.parse_args()

    args.log_dir    = os.path.join(args.log_dir,    args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name, args.run_name)
    os.makedirs(args.log_dir,    exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)
