"""Pre-bake Tactile VQ-VAE codes into a post-training JSON.

For each sample at frame `idx` in episode `ep`, builds the historical 16-frame
window of `tactile_f6` from sibling samples in the same episode (frames
[idx-15, ..., idx], left-edge-padded if idx < 15), encodes it per-hand with the
trained VQ-VAE, and writes `tactile_codes: [code_left, code_right]` into the
sample.  Output is a new JSON; the original is not touched.

Usage:
  python -m utils.encode_vqvae_codes_to_json \
      --input_json  /path/wipe_plate_..._train.json \
      --output_json /path/wipe_plate_..._train_vqvae_k64.json \
      --vqvae_ckpt  /path/vqvae_f6_w16_k64_0504_1856/latest.pt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from tactile_vqvae.data.stats import TacF6Stats
from tactile_vqvae.models.tactile_vqvae import TactileVQVAE, TactileVQVAEConfig


_FRAME_RE = re.compile(r"(.+/episode_\d+)/image(\d+)_")


def _parse_episode_frame(deform_path: str) -> Tuple[str, int]:
    """tactile_image_deform[0] → (episode_dir, frame_idx)."""
    m = _FRAME_RE.search(deform_path)
    if not m:
        raise ValueError(f"Could not parse episode/frame from: {deform_path}")
    return m.group(1), int(m.group(2))


def _load_vqvae(ckpt_path: str, device: torch.device) -> Tuple[TactileVQVAE, TacF6Stats, int]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = TactileVQVAEConfig.from_dict(blob["config"])
    model = TactileVQVAE(cfg)
    model.load_state_dict(blob["model_state"])
    model.eval().to(device)

    stats = TacF6Stats.from_dict(blob["stats"])
    return model, stats, int(cfg.window)


def _build_windows(
    ordered_samples: List[Tuple[int, dict]],
    window: int,
) -> np.ndarray:
    """Stack tactile_f6 from consecutive frames in episode-order into [N, W, 10, 6]
    where N = number of samples and the W frames at row i are the historical
    window ending at sample i (left-edge-padded with sample 0 when i < W-1).
    """
    f6_per_frame = np.stack(
        [np.asarray(s["tactile_f6"], dtype=np.float32) for _, s in ordered_samples],
        axis=0,
    )                                                           # [N, 10, 6]
    n = f6_per_frame.shape[0]
    out = np.empty((n, window, 10, 6), dtype=np.float32)
    for i in range(n):
        start = i - (window - 1)
        if start >= 0:
            out[i] = f6_per_frame[start: i + 1]
        else:
            pad = -start
            head = np.repeat(f6_per_frame[:1], pad, axis=0)     # [pad, 10, 6]
            out[i] = np.concatenate([head, f6_per_frame[: i + 1]], axis=0)
    return out


def _encode_per_hand(
    windows: np.ndarray,        # [N, W, 10, 6] normalized
    model: TactileVQVAE,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Encode each hand separately. Returns [N, 2] int32 (left, right)."""
    n = windows.shape[0]
    codes = np.zeros((n, 2), dtype=np.int32)
    for hand in (0, 1):
        wh = windows[:, :, hand * 5: (hand + 1) * 5, :]         # [N, W, 5, 6]
        for i in range(0, n, batch_size):
            batch = torch.from_numpy(wh[i: i + batch_size]).to(device)
            with torch.no_grad():
                idx = model.encode(batch).cpu().numpy().astype(np.int32)
            codes[i: i + batch.shape[0], hand] = idx
    return codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json",  required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--vqvae_ckpt",  required=True)
    ap.add_argument("--batch_size",  type=int, default=512)
    ap.add_argument("--cuda",        type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    model, stats, window = _load_vqvae(args.vqvae_ckpt, device)
    print(f">>> loaded VQ-VAE: window={window}, "
          f"codebook_size={model.cfg.codebook_size}")

    with open(args.input_json, "r") as f:
        samples = json.load(f)
    print(f">>> loaded {len(samples):,} samples")

    # Group by episode, keep original sample index for write-back.
    by_episode = defaultdict(list)                              # ep_dir → [(frame, orig_idx, sample), ...]
    for orig_idx, s in enumerate(samples):
        deforms = s.get("tactile_image_deform", [])
        if not deforms:
            raise ValueError(f"sample {orig_idx} has no tactile_image_deform")
        ep_dir, frame = _parse_episode_frame(deforms[0])
        by_episode[ep_dir].append((frame, orig_idx, s))

    print(f">>> grouped into {len(by_episode)} episodes")

    for ep_dir, items in tqdm(by_episode.items(), desc="encoding episodes"):
        items.sort(key=lambda t: t[0])                          # by frame
        ordered = [(f, s) for f, _, s in items]

        raw_windows  = _build_windows(ordered, window)          # [N, W, 10, 6]
        norm_windows = stats.normalize(raw_windows).astype(np.float32, copy=False)
        codes = _encode_per_hand(norm_windows, model, device, args.batch_size)

        for (_, orig_idx, _), (cl, cr) in zip(items, codes):
            samples[orig_idx]["tactile_codes"] = [int(cl), int(cr)]

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    # indent=2 matches the original JSON's pretty-printed format.  The newlines
    # are required so HF datasets / pyarrow can chunk the file — a single-line
    # JSON >2GB triggers an int32 block_size overflow in pyarrow's JSON reader.
    with open(args.output_json, "w") as f:
        json.dump(samples, f, indent=2)
    print(f">>> wrote {args.output_json}")

    # Mirror the sibling _statistics.json so train_qwen3vl_flare.py can find it.
    src_stats = args.input_json.replace(".json", "_statistics.json")
    dst_stats = args.output_json.replace(".json", "_statistics.json")
    if os.path.exists(src_stats) and not os.path.exists(dst_stats):
        shutil.copy(src_stats, dst_stats)
        print(f">>> copied stats → {dst_stats}")


if __name__ == "__main__":
    main()
