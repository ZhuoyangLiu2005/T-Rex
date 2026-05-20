#!/usr/bin/env python3
"""Pre-compute midtrain normalization stats on a single node.

Walks every `pretrain.hdf5` under <data_root>/<batch>/<episode>/ (with a
raw.h5 fallback for tactile_f6), computes per-(step, dim) action quantiles
and per-dim tactile_f6 quantiles, and writes the cache JSON that
`MidtrainTacFlareDataset._compute_or_load_normalization_stats` looks for:

    <data_root>/midtrain_statistics_c<C>_d<D>.json

Schema matches the post-train `_statistics.json` so the file is also a
drop-in for `<DATA_JSON>_statistics.json` on a post-train resume.

Running this once before launching distributed training lets every rank
hit the cache on `os.path.isfile()` and skip the 3–10 min h5 walk —
no NCCL barrier wait, no idle GPUs.

Usage
-----
    python precompute_midtrain_stats.py \
        --data_root /mnt/amlfs-02/.../midtrain/merged_inlab \
        --action_chunk 16 --action_dim 62
"""

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm


def _read_episode_tactile_f6(ep_dir):
    """Mirror MidtrainTacFlareDataset._read_episode_tactile_f6 — try
    pretrain.hdf5["tactile_f6"] first, fall back to raw.h5 per-hand keys.
    Returns [N, 10, 6] float32 array or None.
    """
    ph5 = os.path.join(ep_dir, "pretrain.hdf5")
    if os.path.isfile(ph5):
        try:
            with h5py.File(ph5, "r") as f:
                if "tactile_f6" in f:
                    return f["tactile_f6"][:].astype(np.float32, copy=False)
        except Exception:
            pass
    rh5 = os.path.join(ep_dir, "raw.h5")
    if os.path.isfile(rh5):
        try:
            with h5py.File(rh5, "r") as f:
                if ("left_hand_tactile_f6" in f
                        and "right_hand_tactile_f6" in f):
                    l = f["left_hand_tactile_f6"][:]
                    r = f["right_hand_tactile_f6"][:]
                    return np.concatenate(
                        [l, r], axis=1).astype(np.float32, copy=False)
        except Exception:
            pass
    return None


def _read_episode_action_chunks(ep_dir):
    """Read action_chunks [N, C, D] from pretrain.hdf5.  None when missing."""
    ph5 = os.path.join(ep_dir, "pretrain.hdf5")
    if not os.path.isfile(ph5):
        return None
    try:
        with h5py.File(ph5, "r") as f:
            if "action_chunks" in f:
                return f["action_chunks"][:].astype(np.float32, copy=False)
    except Exception:
        pass
    return None


def collect_episode_dirs(data_root):
    """Walk <data_root>/*/pretrain_manifest.json and return all episode dirs."""
    manifest_paths = sorted(
        glob.glob(os.path.join(data_root, "*", "pretrain_manifest.json")))
    if not manifest_paths:
        raise FileNotFoundError(
            f"No pretrain_manifest.json under {data_root}/*/")
    episode_dirs = []
    for mp in manifest_paths:
        with open(mp, "r") as f:
            manifest = json.load(f)
        for ep in manifest.get("episodes", []):
            episode_dirs.append(ep["episode_dir"])
    print(f"[precompute] {len(manifest_paths)} manifests, "
          f"{len(episode_dirs)} episodes")
    return episode_dirs


def _subsample_then_quantile(A, axis, max_samples, label):
    """Compute 1%/99% quantiles along `axis` after subsampling along that
    axis when its size > max_samples.  Avoids the np.quantile path that
    allocates a non-contiguous working buffer the size of the input array
    (which makes a [10M, 16, 62] float32 input peak at ~70 GB of RAM and
    swap-thrash on smaller nodes).

    For 1%/99% quantiles, 1–2M samples gives standard error <1e-3, far
    below any sane practical precision for normalization stats.
    """
    n = A.shape[axis]
    if n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_samples, replace=False)
        idx.sort()
        A_sub = np.take(A, idx, axis=axis)
        print(f"[precompute] {label}: subsampled {max_samples:,} / {n:,} "
              f"along axis {axis}")
    else:
        A_sub = A
        print(f"[precompute] {label}: quantile over full {n:,} samples")
    # np.ascontiguousarray on the subsampled array is cheap (a few GB at
    # most) and gives np.quantile a contiguous buffer to partition.
    A_sub = np.ascontiguousarray(A_sub)
    q01 = np.quantile(A_sub, 0.01, axis=axis).astype(np.float32)
    q99 = np.quantile(A_sub, 0.99, axis=axis).astype(np.float32)
    del A_sub
    return q01, q99


def compute_stats(episode_dirs, action_chunk, action_dim,
                  max_action_samples, max_tactile_samples):
    """Concatenate action_chunks + tactile_f6 across all episodes, compute
    quantiles.  Returns (q01_action [C,D], q99_action [C,D],
    q01_tactile [60] | None, q99_tactile [60] | None).

    Subsamples down to `max_*_samples` along the time axis before
    np.quantile to avoid the (n, C, D) memory blow-up path.
    """
    chunks_list = []
    tac_list = []
    for ep_dir in tqdm(episode_dirs, desc="reading h5"):
        ac = _read_episode_action_chunks(ep_dir)
        if ac is not None:
            chunks_list.append(ac)
        tf = _read_episode_tactile_f6(ep_dir)
        if tf is not None:
            tac_list.append(tf)

    if not chunks_list:
        raise RuntimeError(
            f"No action_chunks found in any episode HDF5 under {episode_dirs[0]} ...")
    A = np.concatenate(chunks_list, axis=0)
    del chunks_list
    print(f"[precompute] action stack: {A.shape[0]:,} chunks "
          f"(shape={tuple(A.shape)}, "
          f"{A.nbytes / (1024**3):.1f} GB float32)")
    if A.shape[1] != action_chunk or A.shape[2] != action_dim:
        raise ValueError(
            f"action_chunks shape {tuple(A.shape[1:])} doesn't match "
            f"--action_chunk={action_chunk} --action_dim={action_dim}; "
            f"check args.")
    q01_a, q99_a = _subsample_then_quantile(
        A, axis=0, max_samples=max_action_samples, label="action")
    del A

    q01_t = q99_t = None
    if tac_list:
        T = np.concatenate(tac_list, axis=0).reshape(-1, 60)  # [Nt, 60]
        del tac_list
        print(f"[precompute] tactile_f6 stack: {T.shape[0]:,} frames "
              f"(shape={tuple(T.shape)}, "
              f"{T.nbytes / (1024**3):.1f} GB float32)")
        q01_t, q99_t = _subsample_then_quantile(
            T, axis=0, max_samples=max_tactile_samples, label="tactile_f6")
        del T
    else:
        print("[precompute] no tactile_f6 found in any episode "
              "(falls back to manifest pool / VQ-VAE ckpt at training time)")
    return q01_a, q99_a, q01_t, q99_t


def write_cache(cache_path, q01_a, q99_a, q01_t, q99_t, action_dim):
    blob = {
        "midtrain": {
            "action": {
                "mask": np.ones(action_dim, dtype=bool).tolist(),
                "q01":  q01_a.tolist(),
                "q99":  q99_a.tolist(),
            }
        }
    }
    if q01_t is not None:
        blob["midtrain"]["tactile_f6"] = {
            "mask": np.ones(60, dtype=bool).tolist(),
            "q01":  q01_t.tolist(),
            "q99":  q99_t.tolist(),
        }
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(blob, f)
    os.replace(tmp, cache_path)
    print(f"[precompute] wrote {cache_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True,
                    help="<data_root> containing <batch>/pretrain_manifest.json")
    ap.add_argument("--action_chunk", type=int, default=16)
    ap.add_argument("--action_dim",   type=int, default=62)
    ap.add_argument("--max_action_samples", type=int, default=2_000_000,
                    help="Subsample N chunks before np.quantile to bound "
                         "peak RAM.  For 1%%/99%% quantiles 1–2M samples "
                         "give SE < 1e-3.  Set to a very large number "
                         "(e.g. 100_000_000) to disable subsampling.")
    ap.add_argument("--max_tactile_samples", type=int, default=2_000_000,
                    help="Same idea, for tactile_f6 frames (smaller "
                         "footprint per frame, but keeps things consistent).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing cache file.")
    args = ap.parse_args()

    cache_path = os.path.join(
        args.data_root,
        f"midtrain_statistics_c{args.action_chunk}_d{args.action_dim}.json")

    if os.path.isfile(cache_path) and not args.force:
        print(f"[precompute] cache already exists: {cache_path}")
        print(f"[precompute] pass --force to recompute, or delete the file.")
        return

    episode_dirs = collect_episode_dirs(args.data_root)
    q01_a, q99_a, q01_t, q99_t = compute_stats(
        episode_dirs, args.action_chunk, args.action_dim,
        max_action_samples=args.max_action_samples,
        max_tactile_samples=args.max_tactile_samples)
    write_cache(cache_path, q01_a, q99_a, q01_t, q99_t, args.action_dim)

    # Quick diagnostic so user can sanity-check
    a_range = (q99_a - q01_a).mean()
    print(f"[precompute] action q99-q01 mean (across step+dim): {a_range:.4f}")
    if q01_t is not None:
        t_range = (q99_t - q01_t).mean()
        print(f"[precompute] tactile_f6 q99-q01 mean (across 60 dims): "
              f"{t_range:.4f}")


if __name__ == "__main__":
    main()
