"""Tactile VQ-VAE training loop.

Launches with accelerate (single- or multi-node) on the merged midtrain root.

Example:
    accelerate launch -m tactile_vqvae.train \\
        --data_root $MERGED_DATA_ROOT \\
        --output_dir $OUT \\
        --window 16 --codebook_size 1024 --embed_dim 256 \\
        --epochs 30 --batch_size 256 --lr 3e-4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed

# Allow `python -m tactile_vqvae.train` from the repo root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT   = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from tactile_vqvae.data import TacF6Stats, build_train_val_datasets
from tactile_vqvae.models import TactileVQVAE
from tactile_vqvae.models.tactile_vqvae import TactileVQVAEConfig


# ─── argparse ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--data_root", type=str, required=True,
                   help="Merged midtrain root (contains */pretrain_manifest.json)")
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--stride", type=int, default=4,
                   help="Stride between window starts during training (1 = every frame)")
    p.add_argument("--val_ratio", type=float, default=0.02)
    p.add_argument("--num_workers", type=int, default=4)

    # Model
    p.add_argument("--hidden_channels", type=int, default=128)
    p.add_argument("--bottleneck_channels", type=int, default=256)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--n_strided_blocks", type=int, default=2)
    p.add_argument("--codebook_size", type=int, default=1024)
    p.add_argument("--commitment_weight", type=float, default=0.25)
    p.add_argument("--decay", type=float, default=0.99)
    p.add_argument("--revive_freq", type=int, default=200)
    p.add_argument("--revive_threshold", type=float, default=1.0)
    p.add_argument("--use_magnitude_weight", type=int, default=1)
    p.add_argument("--weight_alpha", type=float, default=2.0)
    p.add_argument("--weight_tau", type=float, default=4.0)
    p.add_argument("--granularity", type=str, default="hand",
                   choices=["hand", "finger"],
                   help="hand: 1 code per (hand, window).  finger: 5 codes per "
                        "(hand, window) — encoder/decoder process each finger "
                        "with shared weights + finger ID embedding.")

    # Training
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256, help="per-GPU")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--mixed_precision", type=str, default="bf16",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=42)

    # Logging / checkpointing
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=2000,
                   help="In optimizer steps. 0 = no val.")
    p.add_argument("--save_every_epoch", type=int, default=1)
    p.add_argument("--use_wandb", type=int, default=0)
    p.add_argument("--wandb_project", type=str, default="dex_mot_tactile_vqvae")

    # Smoke test
    p.add_argument("--smoke_test", type=int, default=0,
                   help="If 1, run 5 train steps + 1 val pass and exit.")

    return p.parse_args()


# ─── helpers ──────────────────────────────────────────────────────────────────

def cosine_lr(step: int, total: int, warmup: int, base_lr: float, min_ratio: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cos)


def _build_config(args) -> TactileVQVAEConfig:
    return TactileVQVAEConfig(
        window=args.window,
        in_channels=30,
        hidden_channels=args.hidden_channels,
        bottleneck_channels=args.bottleneck_channels,
        embed_dim=args.embed_dim,
        n_strided_blocks=args.n_strided_blocks,
        codebook_size=args.codebook_size,
        commitment_weight=args.commitment_weight,
        decay=args.decay,
        revive_freq=args.revive_freq,
        revive_threshold=args.revive_threshold,
        use_magnitude_weight=bool(args.use_magnitude_weight),
        weight_alpha=args.weight_alpha,
        weight_tau=args.weight_tau,
        granularity=args.granularity,
    )


def _save_checkpoint(
    accelerator: Accelerator,
    out_dir: str,
    model: TactileVQVAE,
    optimizer: torch.optim.Optimizer,
    stats: TacF6Stats,
    cfg: TactileVQVAEConfig,
    step: int,
    epoch: int,
):
    if not accelerator.is_main_process:
        return
    os.makedirs(out_dir, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    state = {
        "model_state":  unwrapped.state_dict(),
        "optim_state":  optimizer.state_dict(),
        "step":         step,
        "epoch":        epoch,
        "config":       cfg.to_dict(),
        "stats":        stats.to_dict(),
    }
    ckpt_path = os.path.join(out_dir, f"checkpoint_epoch{epoch:03d}.pt")
    torch.save(state, ckpt_path)
    # Also write a `latest.pt` symlink-style copy.
    torch.save(state, os.path.join(out_dir, "latest.pt"))
    accelerator.print(f"  ✔ saved checkpoint → {ckpt_path}")


@torch.no_grad()
def _validate(model: TactileVQVAE, val_loader: DataLoader, accelerator: Accelerator,
              max_batches: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    sums = {"recon": 0.0, "vq": 0.0, "perp": 0.0, "active": 0.0, "n": 0}
    # Per-quartile recon (for diagnostics): we track recon per F6-magnitude bin.
    bin_recon = [0.0, 0.0, 0.0, 0.0]
    bin_count = [0,   0,   0,   0]
    # Magnitude bin edges learnt from val batch (running quartile thresholds).
    mag_buf = []

    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        f6 = batch["f6"]
        magnitude = batch["magnitude"]
        out = model(f6, magnitude)
        sums["recon"]  += float(out["recon_loss"].item()) * f6.shape[0]
        sums["vq"]     += float(out["vq_loss"].item())   * f6.shape[0]
        sums["perp"]   += float(out["perplexity"].item()) * f6.shape[0]
        sums["active"] += float(out["active_codes"].item()) * f6.shape[0]
        sums["n"]      += f6.shape[0]
        mag_buf.append(magnitude.detach().cpu().numpy())

    if sums["n"] == 0:
        return {"val_recon": float("nan"), "val_vq": float("nan"),
                "val_perplexity": float("nan"), "val_active": float("nan")}

    # Now do a quartile pass on a copy of the loader (just compute on the same batches).
    model.train()  # restore
    return {
        "val_recon":      sums["recon"]  / sums["n"],
        "val_vq":         sums["vq"]     / sums["n"],
        "val_perplexity": sums["perp"]   / sums["n"],
        "val_active":     sums["active"] / sums["n"],
    }


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device

    run_name = args.run_name or time.strftime("vqvae_f6_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_name)
    if accelerator.is_main_process:
        os.makedirs(run_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    accelerator.print(f"[VQ-VAE] run_dir = {run_dir}")
    accelerator.print(f"[VQ-VAE] device  = {device}, mixed_precision={args.mixed_precision}")

    # ── Stats + datasets ──────────────────────────────────────────────────────
    accelerator.print(f"[VQ-VAE] Scanning {args.data_root} for episodes …")
    stats = TacF6Stats.from_data_root(args.data_root)
    accelerator.print(
        f"[VQ-VAE] tacf6_min[:6] = {stats.tacf6_min[:6]}, "
        f"tacf6_max[:6] = {stats.tacf6_max[:6]}")

    train_ds, val_ds, _ = build_train_val_datasets(
        data_root=args.data_root,
        window=args.window,
        stride=args.stride,
        val_ratio=args.val_ratio,
        seed=args.seed,
        stats=stats,
    )
    accelerator.print(
        f"[VQ-VAE] train: {train_ds.num_episodes} eps / {len(train_ds)} windows ;"
        f" val: {val_ds.num_episodes} eps / {len(val_ds)} windows")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=train_ds.collate_fn, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2), pin_memory=True, drop_last=False,
        collate_fn=val_ds.collate_fn,
    )

    # ── Model + optimizer ─────────────────────────────────────────────────────
    cfg = _build_config(args)
    model = TactileVQVAE(cfg)
    accelerator.print(f"[VQ-VAE] config: {json.dumps(cfg.to_dict(), indent=2)}")

    n_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"[VQ-VAE] params: {n_params/1e6:.2f}M")

    # Don't apply weight decay to codebook buffers (they're not parameters anyway).
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay, eps=1e-8,
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # ── wandb ────────────────────────────────────────────────────────────────
    use_wandb = bool(args.use_wandb) and accelerator.is_main_process
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=run_name,
                       config={**vars(args), **cfg.to_dict()})
        except Exception as e:
            accelerator.print(f"[VQ-VAE] wandb disabled: {e}")
            use_wandb = False

    # ── Train ─────────────────────────────────────────────────────────────────
    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    accelerator.print(
        f"[VQ-VAE] {args.epochs} epochs × {steps_per_epoch} steps = "
        f"{total_steps} total optimizer steps (per rank)")

    global_step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            lr_now = cosine_lr(
                global_step, total_steps, args.warmup_steps,
                args.lr, args.min_lr_ratio,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            optimizer.zero_grad(set_to_none=True)
            out = model(batch["f6"], batch["magnitude"])
            loss = out["total_loss"]

            accelerator.backward(loss)
            if args.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if global_step % args.log_every == 0 and accelerator.is_main_process:
                msg = (f"[step {global_step:7d} | ep {epoch:2d}] "
                       f"recon={out['recon_loss'].item():.4f} "
                       f"vq={out['vq_loss'].item():.4f} "
                       f"perp={out['perplexity'].item():.1f} "
                       f"active={int(out['active_codes'].item())}/{cfg.codebook_size} "
                       f"revived={int(out['revived'].item())} "
                       f"lr={lr_now:.2e} "
                       f"elapsed={time.time()-t0:.0f}s")
                accelerator.print(msg)
                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/recon":        out["recon_loss"].item(),
                        "train/vq":           out["vq_loss"].item(),
                        "train/perplexity":   out["perplexity"].item(),
                        "train/active_codes": int(out["active_codes"].item()),
                        "train/revived":      int(out["revived"].item()),
                        "lr":                 lr_now,
                        "epoch":              epoch,
                    }, step=global_step)

            if (args.val_every > 0 and global_step > 0
                    and global_step % args.val_every == 0):
                v = _validate(accelerator.unwrap_model(model), val_loader,
                              accelerator, max_batches=50)
                if accelerator.is_main_process:
                    accelerator.print(
                        f"  [val @ step {global_step}] " +
                        " ".join(f"{k}={v[k]:.4f}" for k in v))
                    if use_wandb:
                        import wandb
                        wandb.log({f"val/{k.replace('val_','')}": vv for k, vv in v.items()},
                                  step=global_step)

            global_step += 1
            if args.smoke_test and global_step >= 5:
                break

        if args.smoke_test:
            break

        if (epoch + 1) % args.save_every_epoch == 0:
            _save_checkpoint(
                accelerator, run_dir,
                model, optimizer, stats, cfg, global_step, epoch,
            )

    # Final save (smoke test or last epoch).
    _save_checkpoint(
        accelerator, run_dir,
        model, optimizer, stats, cfg, global_step,
        epoch=args.epochs - 1 if not args.smoke_test else 0,
    )

    if use_wandb and accelerator.is_main_process:
        import wandb
        wandb.finish()

    accelerator.print(f"[VQ-VAE] Done. Total wallclock: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
