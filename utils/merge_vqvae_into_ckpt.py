"""Merge a standalone Tactile VQ-VAE checkpoint into a VLA checkpoint.

Bakes the VQ-VAE encoder/quantizer weights + F6 normalization stats into a
trained VLA checkpoint so the model carries its own on-the-fly tactile
tokenizer.  After merging, training / inference no longer need a separate
`--vqvae_ckpt` or pre-baked `tactile_codes`: the model reads raw F6 history
and produces codes internally (see `Qwen3VLVLAModel.encode_tactile_f6_history`).

The source VLA checkpoint must already have a trained `tactile_code_embedder`
(i.e. it was trained with `--use_tactile_code 1`) whose codebook size matches
the VQ-VAE — that embedding is what turns each integer code into a token, and
it is *not* something this script can synthesize.

Usage:
  python utils/merge_vqvae_into_ckpt.py \
      --vla_ckpt   /path/checkpoint-99-12345 \
      --vqvae_ckpt /path/vqvae_f6_w16_k64_finger_XXXX/latest.pt \
      --output     /path/checkpoint-99-12345-vqvae
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vla_ckpt",   required=True,
                    help="VLA checkpoint dir (contains model.pt, training_args.json, ...).")
    ap.add_argument("--vqvae_ckpt", required=True,
                    help="Standalone VQ-VAE checkpoint blob (config/model_state/stats).")
    ap.add_argument("--output",     required=True,
                    help="Output checkpoint dir to write the merged model into.")
    args = ap.parse_args()

    model_pt = os.path.join(args.vla_ckpt, "model.pt")
    if not os.path.exists(model_pt):
        raise FileNotFoundError(f"model.pt not found in {args.vla_ckpt}")

    print(f">>> loading VLA state dict: {model_pt}")
    sd = torch.load(model_pt, map_location="cpu")
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    print(f">>> loading VQ-VAE blob: {args.vqvae_ckpt}")
    blob = torch.load(args.vqvae_ckpt, map_location="cpu", weights_only=False)
    vq_cfg   = blob["config"]
    vq_state = blob["model_state"]
    vq_stats = blob["stats"]
    codebook_size = int(vq_cfg["codebook_size"])
    granularity   = vq_cfg.get("granularity", "hand")
    n_fingers     = int(vq_cfg.get("n_fingers", 5))
    print(f"    VQ-VAE: codebook={codebook_size}, granularity={granularity}, "
          f"window={vq_cfg.get('window')}")

    # ── Sanity: the code embedder must exist and match the codebook size ──────
    emb_key = "tactile_code_embedder.weight"
    if emb_key not in sd:
        print(f"!!! WARNING: {emb_key} not found in the VLA checkpoint.  The "
              f"merged model will have a randomly-initialised code embedder — "
              f"only do this if you intend to (re)train it.  Normally you merge "
              f"into a checkpoint trained with --use_tactile_code 1.")
    else:
        emb_rows = sd[emb_key].shape[0]
        if emb_rows != codebook_size:
            raise ValueError(
                f"tactile_code_embedder has {emb_rows} rows but the VQ-VAE "
                f"codebook is {codebook_size}.  They must match — re-train the "
                f"VLA with --vqvae_codebook_size {codebook_size} or use the "
                f"matching VQ-VAE checkpoint.")
        print(f"    code embedder OK: {emb_rows} rows == codebook {codebook_size}")

    # ── Inject VQ-VAE weights under the tactile_vqvae.* prefix ───────────────
    n_added = 0
    for k, v in vq_state.items():
        sd[f"tactile_vqvae.{k}"] = v
        n_added += 1
    print(f">>> injected {n_added} tactile_vqvae.* tensors")

    # ── Inject F6 normalization stat buffers ─────────────────────────────────
    sd["tacf6_vqvae_min"]  = torch.as_tensor(vq_stats["tacf6_min"],  dtype=torch.float32)
    sd["tacf6_vqvae_max"]  = torch.as_tensor(vq_stats["tacf6_max"],  dtype=torch.float32)
    sd["tacf6_vqvae_mask"] = torch.as_tensor(vq_stats["tacf6_mask"], dtype=torch.bool)
    print(">>> injected F6 stat buffers (tacf6_vqvae_min/max/mask)")

    # ── Write merged checkpoint dir ──────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    torch.save(sd, os.path.join(args.output, "model.pt"))
    print(f">>> wrote {os.path.join(args.output, 'model.pt')}")

    # Copy the static sidecars needed to reconstruct the model at load time.
    for name in ("processor", "config.json", "stats_data.json"):
        src = os.path.join(args.vla_ckpt, name)
        dst = os.path.join(args.output, name)
        if os.path.exists(src):
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy(src, dst)
            print(f">>> copied {name}")
    # Note: the optimizer state/ dir is intentionally NOT copied — the merged
    # model has extra (frozen) params, so resume must start a fresh optimizer.

    # ── Update training_args.json so loaders auto-detect the embedded VQ-VAE ──
    ta_path = os.path.join(args.vla_ckpt, "training_args.json")
    ta = {}
    if os.path.exists(ta_path):
        with open(ta_path) as f:
            ta = json.load(f)
    ta["use_tactile_vqvae"]   = 1
    ta["use_tactile_code"]    = 1
    ta["vqvae_codebook_size"] = codebook_size
    ta["vqvae_config"]        = vq_cfg
    with open(os.path.join(args.output, "training_args.json"), "w") as f:
        json.dump(ta, f, indent=2)
    print(">>> updated training_args.json (use_tactile_vqvae=1, vqvae_config baked in)")
    print(f"\nMerged checkpoint ready at: {args.output}")


if __name__ == "__main__":
    main()
