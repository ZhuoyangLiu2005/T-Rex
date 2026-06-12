"""F6WindowDataset — yields normalized per-hand F6 windows from midtrain episodes.

Each item is a [T, 5, 6] window from one hand of one episode, where:
  T = window size in frames
  5 = fingers per hand
  6 = force/torque dims

Left and right hands are treated as independent samples (the F6 representation
has no left/right structural difference at the per-finger level), which doubles
the effective dataset size.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .stats import TacF6Stats


def _scan_episodes(data_root: str) -> Tuple[List[str], List[int]]:
    """Walk merged_data_root → list of (episode_dir, num_frames)."""
    manifest_paths = sorted(
        glob.glob(os.path.join(data_root, "*", "pretrain_manifest.json"))
    )
    if not manifest_paths:
        raise FileNotFoundError(
            f"No pretrain_manifest.json under {data_root}/*/")

    ep_dirs: List[str] = []
    n_frames: List[int] = []
    for mp in manifest_paths:
        with open(mp, "r") as f:
            manifest = json.load(f)
        for ep in manifest["episodes"]:
            ep_dirs.append(ep["episode_dir"])
            n_frames.append(int(ep["num_frames"]))
    return ep_dirs, n_frames


class F6WindowDataset(Dataset):
    """Per-hand F6 window dataset.

    Parameters
    ----------
    data_root : str
        Merged midtrain root (the symlink dir from the existing midtrain
        launcher). Must contain `*/pretrain_manifest.json`.
    window : int
        Number of consecutive frames per window (default 16, aligned with
        action_chunk).
    stride : int
        Spacing between window starts within an episode (default 1 = max
        diversity).
    stats : TacF6Stats
        Pre-built normalizer. Pass `TacF6Stats.from_data_root(data_root)`
        on the same root.
    episodes : Optional[Tuple[List[str], List[int]]]
        Pre-scanned (episode_dirs, num_frames) — used by `build_train_val_datasets`
        to share the scan and split episodes between train/val.
    drop_short : bool
        Skip episodes with fewer than `window` frames (default True).
    """

    def __init__(
        self,
        data_root: str,
        window: int = 16,
        stride: int = 1,
        stats: Optional[TacF6Stats] = None,
        episodes: Optional[Tuple[List[str], List[int]]] = None,
        drop_short: bool = True,
    ):
        self.data_root = data_root
        self.window = int(window)
        self.stride = max(1, int(stride))
        self.stats = stats if stats is not None else TacF6Stats.from_data_root(data_root)

        if episodes is None:
            ep_dirs, n_frames = _scan_episodes(data_root)
        else:
            ep_dirs, n_frames = episodes

        if drop_short:
            kept = [(d, n) for d, n in zip(ep_dirs, n_frames) if n >= self.window]
            if not kept:
                raise RuntimeError(
                    f"All episodes are shorter than window={self.window}.")
            ep_dirs = [d for d, _ in kept]
            n_frames = [n for _, n in kept]

        self._episode_dirs: List[str] = ep_dirs
        self._n_frames:     List[int] = n_frames

        # Build the (ep_idx, frame_start, hand) flat index.
        # For each episode: starts at [0, stride, 2*stride, ...] up to n - window.
        # × 2 hands. Encode as (ep_idx * MAX_HAND + hand, frame_start).
        windows_per_ep = [
            max(0, (n - self.window) // self.stride + 1) for n in n_frames
        ]
        self._windows_per_ep = np.array(windows_per_ep, dtype=np.int64)
        self._cum_windows = np.cumsum(self._windows_per_ep)
        self._total_windows_per_hand = int(self._cum_windows[-1]) if len(self._cum_windows) else 0
        self._total = self._total_windows_per_hand * 2  # × 2 hands

        # Per-worker single-episode F6 cache
        self._cache_ep_idx: int = -1
        self._cache_f6: Optional[np.ndarray] = None  # [N, 10, 6] raw

    def __len__(self) -> int:
        return self._total

    @property
    def num_episodes(self) -> int:
        return len(self._episode_dirs)

    def _load_ep_f6(self, ep_idx: int) -> Optional[np.ndarray]:
        """Load tactile_f6 for one episode (cached per worker)."""
        if ep_idx == self._cache_ep_idx:
            return self._cache_f6

        self._cache_ep_idx = ep_idx
        self._cache_f6 = None

        ph5 = os.path.join(self._episode_dirs[ep_idx], "pretrain.hdf5")
        if not os.path.isfile(ph5):
            return None
        try:
            with h5py.File(ph5, "r") as f:
                if "tactile_f6" not in f:
                    return None
                self._cache_f6 = f["tactile_f6"][:].astype(np.float32, copy=False)
        except Exception:
            self._cache_f6 = None
        return self._cache_f6

    def _decode_idx(self, idx: int) -> Tuple[int, int, int]:
        """Map flat idx → (ep_idx, frame_start, hand)."""
        # Hand index = idx // total_windows_per_hand (0=left, 1=right).
        hand = idx // self._total_windows_per_hand
        within = idx % self._total_windows_per_hand
        ep_idx = int(np.searchsorted(self._cum_windows, within, side="right"))
        prev = int(self._cum_windows[ep_idx - 1]) if ep_idx > 0 else 0
        frame_start = (within - prev) * self.stride
        return ep_idx, frame_start, hand

    def __getitem__(self, idx: int) -> Dict:
        ep_idx, frame_start, hand = self._decode_idx(idx)
        f6 = self._load_ep_f6(ep_idx)

        if f6 is None or frame_start + self.window > f6.shape[0]:
            # Fallback (rare) — return zero window so DataLoader doesn't crash.
            f6_window = np.zeros((self.window, 5, 6), dtype=np.float32)
        else:
            # [window, 10, 6] → slice the 5 fingers for this hand.
            slc = f6[frame_start: frame_start + self.window]    # [T, 10, 6]
            f6_window = slc[:, hand * 5: (hand + 1) * 5, :]      # [T, 5, 6]

        # Normalize using the pre-pooled q01/q99. Stats expect last dim = 6.
        f6_normed = self.stats.normalize(f6_window).astype(np.float32, copy=False)

        # Per-window magnitude on RAW (un-normalized) F6 — drives loss weighting.
        magnitude = float(np.linalg.norm(f6_window))   # scalar L2 over (T,5,6)

        return {
            "f6":         torch.from_numpy(f6_normed),               # [T, 5, 6]
            "magnitude":  torch.tensor(magnitude, dtype=torch.float32),
            "ep_idx":     ep_idx,
            "frame":      frame_start,
            "hand":       hand,
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        f6 = torch.stack([b["f6"] for b in batch], dim=0)            # [B, T, 5, 6]
        magnitude = torch.stack([b["magnitude"] for b in batch], dim=0)  # [B]
        return {
            "f6":        f6,
            "magnitude": magnitude,
            "ep_idx":    torch.tensor([b["ep_idx"] for b in batch], dtype=torch.long),
            "frame":     torch.tensor([b["frame"]  for b in batch], dtype=torch.long),
            "hand":      torch.tensor([b["hand"]   for b in batch], dtype=torch.long),
        }


def build_train_val_datasets(
    data_root: str,
    window: int = 16,
    stride: int = 1,
    val_ratio: float = 0.02,
    seed: int = 42,
    stats: Optional[TacF6Stats] = None,
) -> Tuple[F6WindowDataset, F6WindowDataset, TacF6Stats]:
    """Episode-level train/val split.

    Splits at the episode level (not the frame level) so that no frames from a
    val episode leak into training — important for VQ-VAE generalization
    estimates.
    """
    if stats is None:
        stats = TacF6Stats.from_data_root(data_root)

    ep_dirs, n_frames = _scan_episodes(data_root)
    n_eps = len(ep_dirs)
    if n_eps < 2:
        raise RuntimeError(f"Need ≥2 episodes for train/val split; have {n_eps}.")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_eps)
    n_val = max(1, int(round(n_eps * val_ratio)))
    val_idx = sorted(perm[:n_val].tolist())
    tr_idx  = sorted(perm[n_val:].tolist())

    val_eps = ([ep_dirs[i] for i in val_idx], [n_frames[i] for i in val_idx])
    tr_eps  = ([ep_dirs[i] for i in tr_idx],  [n_frames[i] for i in tr_idx])

    train_ds = F6WindowDataset(
        data_root=data_root, window=window, stride=stride,
        stats=stats, episodes=tr_eps,
    )
    val_ds = F6WindowDataset(
        data_root=data_root, window=window, stride=stride,
        stats=stats, episodes=val_eps,
    )
    return train_ds, val_ds, stats
