"""TacF6Stats — per-finger-per-dim normalization stats pooled across batch manifests.

Mirrors the pooling scheme used in
`dex_mot_qwen/scripts/train_qwen3vl_midtrain_flare.py` so the VQ-VAE sees the
same normalized F6 distribution the downstream MoT will see.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


_F6_DIM = 60   # 10 fingers × 6 dims


@dataclass
class TacF6Stats:
    tacf6_min: np.ndarray   # [60]
    tacf6_max: np.ndarray   # [60]
    tacf6_mask: np.ndarray  # [60] bool

    @classmethod
    def from_data_root(cls, data_root: str) -> "TacF6Stats":
        manifest_paths = sorted(
            glob.glob(os.path.join(data_root, "*", "pretrain_manifest.json"))
        )
        if not manifest_paths:
            raise FileNotFoundError(
                f"No pretrain_manifest.json under {data_root}/*/")

        all_q01, all_q99 = [], []
        for mp in manifest_paths:
            with open(mp, "r") as f:
                manifest = json.load(f)
            stats = manifest.get("statistics", {})
            block = stats.get("tactile_f6")
            if block:
                all_q01.append(np.array(block["q01"], dtype=np.float32))
                all_q99.append(np.array(block["q99"], dtype=np.float32))

        if all_q01:
            tacf6_min = np.min(np.stack(all_q01), axis=0)
            tacf6_max = np.max(np.stack(all_q99), axis=0)
        else:
            tacf6_min = np.full(_F6_DIM, -1.0, dtype=np.float32)
            tacf6_max = np.full(_F6_DIM, +1.0, dtype=np.float32)

        if tacf6_min.shape[0] != _F6_DIM or tacf6_max.shape[0] != _F6_DIM:
            raise ValueError(
                f"Expected F6 stats of dim {_F6_DIM}, got "
                f"{tacf6_min.shape}/{tacf6_max.shape}")

        return cls(
            tacf6_min=tacf6_min,
            tacf6_max=tacf6_max,
            tacf6_mask=np.ones(_F6_DIM, dtype=bool),
        )

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Min-max normalize F6 to [-1, 1].

        Accepts shape [..., 10, 6] or [..., 60]. Returns the same shape.
        """
        orig_shape = x.shape
        flat = x.reshape(-1, _F6_DIM).astype(np.float32, copy=False)
        denom = (self.tacf6_max - self.tacf6_min) + 1e-8
        normed = np.clip(2.0 * (flat - self.tacf6_min) / denom - 1.0, -1.0, 1.0)
        # Apply mask (mask is currently all-True; kept for parity with trainer)
        out = np.where(self.tacf6_mask, normed, flat)
        return out.reshape(orig_shape)

    def denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        orig_shape = x_norm.shape
        flat = x_norm.reshape(-1, _F6_DIM).astype(np.float32, copy=False)
        denom = (self.tacf6_max - self.tacf6_min)
        out = (flat + 1.0) * 0.5 * denom + self.tacf6_min
        out = np.where(self.tacf6_mask, out, flat)
        return out.reshape(orig_shape)

    def to_dict(self) -> dict:
        return {
            "tacf6_min":  self.tacf6_min.tolist(),
            "tacf6_max":  self.tacf6_max.tolist(),
            "tacf6_mask": self.tacf6_mask.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TacF6Stats":
        return cls(
            tacf6_min=np.array(d["tacf6_min"], dtype=np.float32),
            tacf6_max=np.array(d["tacf6_max"], dtype=np.float32),
            tacf6_mask=np.array(d["tacf6_mask"], dtype=bool),
        )

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TacF6Stats":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))
