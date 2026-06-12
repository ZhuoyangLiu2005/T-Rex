"""Extract per-episode tactile codes and write sidecar HDF5 files.

For each episode under <data_root>/*/episode_* (resolved through symlinks),
loads tactile_f6 from pretrain.hdf5, encodes per-hand windows of size W with
stride W (so codes align with action chunks), and writes:

    <episode_dir>/tactile_codes.h5
        codes_per_chunk : [M, 2] int32   (left, right) per chunk
        codes_per_frame : [N, 2] int32   (each frame inherits its chunk's code)
        attrs.window           : int
        attrs.codebook_size    : int
        attrs.checkpoint_path  : str

Episodes are processed sequentially per worker; multi-worker parallelism
distributes episodes across processes.

Example:
    python -m tactile_vqvae.extract_codes \\
        --checkpoint $RUN/latest.pt \\
        --data_root  $MERGED_DATA_ROOT \\
        --num_workers 8
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.multiprocessing as mp

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT   = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from tactile_vqvae.data import TacF6Stats
from tactile_vqvae.models import TactileVQVAE
from tactile_vqvae.models.tactile_vqvae import TactileVQVAEConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_root",  type=str, required=True,
                   help="Merged midtrain root (with */pretrain_manifest.json)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--overwrite",  type=int, default=0,
                   help="If 0, skip episodes that already have tactile_codes.h5")
    p.add_argument("--out_name",   type=str, default="tactile_codes.h5")
    p.add_argument("--alignment",  type=str, default="historical",
                   choices=["historical", "chunk"],
                   help="historical: per-frame codes encode window [t-W+1..t] "
                        "(left-edge-padded) — matches post-training and real-"
                        "world rolling-buffer inference.  chunk: legacy non-"
                        "overlapping block alignment, frame t inherits the "
                        "code of chunk floor(t/W).")
    return p.parse_args()


def _scan_episodes(data_root: str) -> List[Tuple[str, int]]:
    manifest_paths = sorted(
        glob.glob(os.path.join(data_root, "*", "pretrain_manifest.json"))
    )
    out: List[Tuple[str, int]] = []
    for mp in manifest_paths:
        with open(mp, "r") as f:
            manifest = json.load(f)
        for ep in manifest["episodes"]:
            out.append((ep["episode_dir"], int(ep["num_frames"])))
    return out


def _build_windows(f6: np.ndarray, window: int, hand: int) -> Tuple[np.ndarray, int]:
    """Chunk alignment: f6 [N, 10, 6] → [M, T, 5, 6] where M = ceil(N/W)
    non-overlapping blocks (legacy).
    """
    n = f6.shape[0]
    if n == 0:
        return np.zeros((0, window, 5, 6), dtype=np.float32), 0
    n_chunks = (n + window - 1) // window
    pad_n = n_chunks * window - n
    if pad_n > 0:
        last = f6[-1:].repeat(pad_n, axis=0)
        f6_pad = np.concatenate([f6, last], axis=0)
    else:
        f6_pad = f6
    f6_h = f6_pad[:, hand * 5: (hand + 1) * 5, :]    # [n_chunks*W, 5, 6]
    windows = f6_h.reshape(n_chunks, window, 5, 6)
    return windows, n_chunks


def _build_historical_windows(
    f6: np.ndarray, window: int, hand: int
) -> np.ndarray:
    """Historical alignment: for each frame t, window covers
    f6[max(0, t-W+1) .. t] left-edge-padded with f6[0] when t < W-1.

    Returns [N, W, 5, 6] for one hand.  Same alignment as
    utils/encode_vqvae_codes_to_json.py and the real-world server's
    rolling F6 buffer.
    """
    n = f6.shape[0]
    if n == 0:
        return np.zeros((0, window, 5, 6), dtype=np.float32)
    f6_h = f6[:, hand * 5: (hand + 1) * 5, :]                 # [N, 5, 6]
    out = np.empty((n, window, 5, 6), dtype=np.float32)
    for t in range(n):
        start = t - (window - 1)
        if start >= 0:
            out[t] = f6_h[start: t + 1]
        else:
            pad = -start
            head = np.repeat(f6_h[:1], pad, axis=0)            # [pad, 5, 6]
            out[t] = np.concatenate([head, f6_h[: t + 1]], axis=0)
    return out


def _process_episode(
    ep_dir: str,
    n_frames: int,
    model: TactileVQVAE,
    stats: TacF6Stats,
    window: int,
    device: torch.device,
    batch_size: int,
    out_name: str,
    overwrite: bool,
    alignment: str = "historical",
) -> str:
    out_path = os.path.join(ep_dir, out_name)
    if (not overwrite) and os.path.isfile(out_path):
        return f"skip   {ep_dir}"

    # Try two data layouts:
    #   (A) pretrain.hdf5 has tactile_f6 [N, 10, 6]  (legacy mecka 100h)
    #   (B) raw.h5  has left_hand_tactile_f6 [N, 5, 6] + right_hand_tactile_f6
    #       (newer inlab data; raw.h5 is a renamed symlink of episode_<id>.h5)
    f6 = None
    ph5 = os.path.join(ep_dir, "pretrain.hdf5")
    if os.path.isfile(ph5):
        try:
            with h5py.File(ph5, "r") as f:
                if "tactile_f6" in f:
                    f6 = f["tactile_f6"][:].astype(np.float32, copy=False)
        except Exception as e:
            return f"err    {ep_dir}  (pretrain.hdf5: {e})"

    if f6 is None:
        rh5 = os.path.join(ep_dir, "raw.h5")
        if not os.path.isfile(rh5):
            return f"miss   {ep_dir}  (no tactile_f6 in pretrain.hdf5, no raw.h5)"
        try:
            with h5py.File(rh5, "r") as f:
                has_l = "left_hand_tactile_f6" in f
                has_r = "right_hand_tactile_f6" in f
                if not (has_l and has_r):
                    return f"miss   {ep_dir}  (raw.h5 missing left/right tactile_f6)"
                l = f["left_hand_tactile_f6"][:].astype(np.float32, copy=False)
                r = f["right_hand_tactile_f6"][:].astype(np.float32, copy=False)
                # Each is [N, 5, 6]; concatenate fingers → [N, 10, 6] in the
                # canonical (left first, then right) order used by the trainer.
                f6 = np.concatenate([l, r], axis=1)
        except Exception as e:
            return f"err    {ep_dir}  (raw.h5: {e})"

    n = f6.shape[0]
    if n == 0:
        return f"empty  {ep_dir}"

    f6_norm_full = stats.normalize(f6).astype(np.float32, copy=False)

    is_per_finger = getattr(model.cfg, "granularity", "hand") == "finger"
    n_fingers = int(getattr(model.cfg, "n_fingers", 5)) if is_per_finger else 1

    if alignment == "historical":
        # Encode one window per frame (length N) per hand.  This matches
        # post-training (encode_vqvae_codes_to_json.py) and the real-world
        # rolling-buffer inference, so the policy sees the same conditioning
        # signal across all three pipelines.
        per_hand = []
        for hand in (0, 1):
            windows = _build_historical_windows(f6_norm_full, window, hand)
            if is_per_finger:
                all_indices = np.zeros((n, n_fingers), dtype=np.int32)
            else:
                all_indices = np.zeros((n,), dtype=np.int32)
            for i in range(0, n, batch_size):
                batch = torch.from_numpy(windows[i: i + batch_size]).to(device)
                with torch.no_grad():
                    idx = model.encode(batch).cpu().numpy().astype(np.int32)
                all_indices[i: i + batch.shape[0]] = idx
            per_hand.append(all_indices)

        codes_per_frame = np.stack(per_hand, axis=1)
        # Shape: [N, 2] (hand) or [N, 2, F] (finger).

        # Also save chunk-level codes (downsample frames every W) for
        # diagnostics / backward compat readers.
        chunk_starts = np.arange(0, n, window)
        codes_per_chunk = codes_per_frame[chunk_starts]
        n_chunks = codes_per_chunk.shape[0]

    else:  # alignment == "chunk" — legacy non-overlapping blocks
        chunk_codes = []
        for hand in (0, 1):
            windows, n_chunks = _build_windows(f6_norm_full, window, hand)
            if n_chunks == 0:
                shape = (0, n_fingers) if is_per_finger else (0,)
                chunk_codes.append(np.zeros(shape, dtype=np.int32))
                continue

            if is_per_finger:
                all_indices = np.zeros((n_chunks, n_fingers), dtype=np.int32)
            else:
                all_indices = np.zeros((n_chunks,), dtype=np.int32)
            for i in range(0, n_chunks, batch_size):
                batch = torch.from_numpy(windows[i: i + batch_size]).to(device)
                with torch.no_grad():
                    idx = model.encode(batch).cpu().numpy().astype(np.int32)
                all_indices[i: i + batch.shape[0]] = idx
            chunk_codes.append(all_indices)

        n_chunks = max(chunk_codes[0].shape[0], chunk_codes[1].shape[0])

        def _pad(c: np.ndarray) -> np.ndarray:
            if c.shape[0] == n_chunks:
                return c
            pad_shape = (n_chunks - c.shape[0],) + c.shape[1:]
            return np.concatenate(
                [c, np.zeros(pad_shape, dtype=np.int32)], axis=0)

        codes_per_chunk = np.stack(
            [_pad(chunk_codes[0]), _pad(chunk_codes[1])], axis=1)
        chunk_idx_per_frame = np.minimum(np.arange(n) // window, n_chunks - 1)
        codes_per_frame = codes_per_chunk[chunk_idx_per_frame]

    tmp = out_path + ".tmp"
    try:
        with h5py.File(tmp, "w") as fout:
            fout.create_dataset("codes_per_chunk", data=codes_per_chunk,
                                compression="gzip", compression_opts=4)
            fout.create_dataset("codes_per_frame", data=codes_per_frame,
                                compression="gzip", compression_opts=4)
            fout.attrs["window"] = int(window)
            fout.attrs["n_frames"] = int(n)
            fout.attrs["n_chunks"] = int(n_chunks)
            fout.attrs["granularity"] = getattr(model.cfg, "granularity", "hand")
            fout.attrs["alignment"] = alignment
            if is_per_finger:
                fout.attrs["n_fingers"] = n_fingers
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

    return f"ok     {ep_dir}  N={n}  M={n_chunks}"


def _worker(rank: int, world: int, args, ep_list: List[Tuple[str, int]]):
    print(f"[w{rank}] starting; episodes={len(ep_list)//world}+", flush=True)
    state = torch.load(args.checkpoint, map_location="cpu")
    cfg = TactileVQVAEConfig.from_dict(state["config"])
    stats = TacF6Stats.from_dict(state["stats"])

    if args.device == "cuda" and torch.cuda.is_available():
        n_dev = torch.cuda.device_count()
        device = torch.device(f"cuda:{rank % max(1, n_dev)}")
    else:
        device = torch.device("cpu")

    model = TactileVQVAE(cfg).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    # Stamp the checkpoint path attribute on each output file.
    ckpt_str = os.path.abspath(args.checkpoint)

    chunk = ep_list[rank::world]
    t0 = time.time()
    for j, (ep_dir, n_frames) in enumerate(chunk):
        msg = _process_episode(
            ep_dir, n_frames, model, stats,
            window=cfg.window, device=device,
            batch_size=args.batch_size,
            out_name=args.out_name,
            overwrite=bool(args.overwrite),
            alignment=args.alignment,
        )
        # Append checkpoint path attr post-hoc (avoid passing through helper).
        out_path = os.path.join(ep_dir, args.out_name)
        if msg.startswith("ok") and os.path.isfile(out_path):
            try:
                with h5py.File(out_path, "a") as f:
                    f.attrs["codebook_size"]   = int(cfg.codebook_size)
                    f.attrs["checkpoint_path"] = ckpt_str
            except Exception:
                pass
        if j % 50 == 0:
            print(f"[w{rank}] {j}/{len(chunk)}  ({time.time()-t0:.0f}s)  {msg}", flush=True)
    print(f"[w{rank}] done in {time.time()-t0:.0f}s", flush=True)


def main():
    args = parse_args()
    eps = _scan_episodes(args.data_root)
    print(f"[ExtractCodes] Found {len(eps)} episodes under {args.data_root}")
    if not eps:
        return

    if args.num_workers <= 1:
        _worker(0, 1, args, eps)
        return

    mp.set_start_method("spawn", force=True)
    procs = []
    for rank in range(args.num_workers):
        p = mp.Process(target=_worker, args=(rank, args.num_workers, args, eps))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
