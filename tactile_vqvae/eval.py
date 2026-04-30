"""Evaluate a trained Tactile VQ-VAE checkpoint.

Reports:
  - Reconstruction MSE overall + by F6-magnitude quartile.
  - Codebook perplexity, active-code count, occupancy histogram.
  - Top-K code exemplars (saved to a .npz for offline inspection).

Example:
    python -m tactile_vqvae.eval \\
        --checkpoint $RUN/latest.pt \\
        --data_root  $MERGED_DATA_ROOT \\
        --output     $RUN/eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT   = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from tactile_vqvae.data import F6WindowDataset, TacF6Stats
from tactile_vqvae.models import TactileVQVAE
from tactile_vqvae.models.tactile_vqvae import TactileVQVAEConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_root",  type=str, required=True)
    p.add_argument("--output",     type=str, default=None,
                   help="Path to write JSON results. Defaults to <ckpt_dir>/eval.json")
    p.add_argument("--exemplars",  type=str, default=None,
                   help="Optional .npz path to save top-K code exemplars")
    p.add_argument("--top_k",      type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--max_batches",type=int, default=0,
                   help="0 = use all val windows; >0 = subsample")
    p.add_argument("--device",     type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[Eval] Loading checkpoint {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cpu")
    cfg = TactileVQVAEConfig.from_dict(state["config"])
    stats = TacF6Stats.from_dict(state["stats"])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = TactileVQVAE(cfg).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    # Use the same train/val split convention by passing *all* episodes as one
    # dataset — we want eval over a representative sample, not just held-out.
    ds = F6WindowDataset(
        data_root=args.data_root, window=cfg.window,
        stride=max(4, cfg.window // 4), stats=stats,
    )
    print(f"[Eval] {ds.num_episodes} eps / {len(ds)} windows")

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=ds.collate_fn,
    )

    # ── Pass 1: collect per-sample recon + magnitude + indices ────────────────
    all_recon = []
    all_mag   = []
    all_idx   = []
    code_count = torch.zeros(cfg.codebook_size, dtype=torch.long)

    # Exemplar buffers: store (magnitude, f6_window, recon) per code.
    # Keep top-K by magnitude per code so we see informative examples.
    exemplars: Dict[int, list] = {}

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if args.max_batches > 0 and bi >= args.max_batches:
                break
            f6 = batch["f6"].to(device, non_blocking=True)
            mag = batch["magnitude"].to(device, non_blocking=True)
            out = model(f6, mag)
            ps  = out["per_sample_recon"]   # [B]
            idx = out["indices"]            # [B]

            all_recon.append(ps.cpu().numpy())
            all_mag.append(mag.cpu().numpy())
            all_idx.append(idx.cpu().numpy())

            for k in range(idx.numel()):
                ci = int(idx[k].item())
                code_count[ci] += 1

            # Maintain exemplars (cheap: keep up to top-3 per code by magnitude).
            if args.exemplars is not None:
                f6_cpu = f6.cpu().numpy()
                recon_cpu = out["recon"].cpu().numpy()
                mag_cpu = mag.cpu().numpy()
                idx_cpu = idx.cpu().numpy()
                for k in range(idx.numel()):
                    ci = int(idx_cpu[k])
                    bucket = exemplars.setdefault(ci, [])
                    bucket.append((float(mag_cpu[k]),
                                   f6_cpu[k].astype(np.float16),
                                   recon_cpu[k].astype(np.float16)))
                    bucket.sort(key=lambda t: -t[0])
                    if len(bucket) > 3:
                        bucket.pop()

            if bi % 50 == 0:
                print(f"  ... batch {bi}/{len(loader)}")

    recon = np.concatenate(all_recon)        # [N]
    mag   = np.concatenate(all_mag)          # [N]
    idx_arr = np.concatenate(all_idx)        # [N]
    print(f"[Eval] N samples = {recon.shape[0]}")

    # ── Quartile breakdown by F6 magnitude ────────────────────────────────────
    quartiles = np.quantile(mag, [0.25, 0.50, 0.75])
    bins = np.digitize(mag, quartiles)        # 0..3
    breakdown = {}
    for b in range(4):
        m = bins == b
        if m.sum() == 0:
            breakdown[f"q{b}"] = {"count": 0}
            continue
        breakdown[f"q{b}"] = {
            "count":     int(m.sum()),
            "recon_mse": float(recon[m].mean()),
            "mag_min":   float(mag[m].min()),
            "mag_max":   float(mag[m].max()),
        }

    # ── Codebook utilization ─────────────────────────────────────────────────
    counts = code_count.numpy()
    total = counts.sum()
    probs = counts / max(1, total)
    perplexity = float(np.exp(-(probs * np.log(probs + 1e-12)).sum()))
    active = int((counts > 0).sum())

    # Concentration: max per-code frequency.
    max_freq = float(probs.max())

    summary = {
        "checkpoint": args.checkpoint,
        "data_root":  args.data_root,
        "n_samples":  int(recon.shape[0]),
        "overall_recon_mse": float(recon.mean()),
        "perplexity":         perplexity,
        "active_codes":       active,
        "codebook_size":      cfg.codebook_size,
        "active_ratio":       active / cfg.codebook_size,
        "max_code_freq":      max_freq,
        "by_magnitude":       breakdown,
        "magnitude_quartiles": quartiles.tolist(),
    }
    print(json.dumps(summary, indent=2))

    # ── Save outputs ─────────────────────────────────────────────────────────
    out_path = args.output or os.path.join(
        os.path.dirname(args.checkpoint) or ".", "eval.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Eval] Wrote summary → {out_path}")

    if args.exemplars is not None:
        # Top-K codes by usage with their exemplar windows.
        top_codes = sorted(range(cfg.codebook_size), key=lambda i: -counts[i])[:args.top_k]
        ex_idx, ex_mag, ex_f6, ex_recon = [], [], [], []
        for ci in top_codes:
            for (mg, f6w, rec) in exemplars.get(ci, []):
                ex_idx.append(ci); ex_mag.append(mg)
                ex_f6.append(f6w); ex_recon.append(rec)
        np.savez(
            args.exemplars,
            code=np.array(ex_idx, dtype=np.int32),
            magnitude=np.array(ex_mag, dtype=np.float32),
            f6=np.stack(ex_f6) if ex_f6 else np.zeros((0, cfg.window, 5, 6), dtype=np.float16),
            recon=np.stack(ex_recon) if ex_recon else np.zeros((0, cfg.window, 5, 6), dtype=np.float16),
            top_codes=np.array(top_codes, dtype=np.int32),
            counts=counts,
        )
        print(f"[Eval] Wrote exemplars → {args.exemplars}")


if __name__ == "__main__":
    main()
