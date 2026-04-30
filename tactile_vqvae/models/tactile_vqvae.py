"""TactileVQVAE — full encoder + EMA quantizer + decoder over F6 windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .encoder import F6Encoder
from .decoder import F6Decoder
from .quantizer import VQEMAQuantizer


@dataclass
class TactileVQVAEConfig:
    window: int = 16
    in_channels: int = 30                 # 5 fingers × 6 dims
    hidden_channels: int = 128
    bottleneck_channels: int = 256
    embed_dim: int = 256
    n_strided_blocks: int = 2
    codebook_size: int = 1024
    commitment_weight: float = 0.25
    decay: float = 0.99
    revive_freq: int = 200
    revive_threshold: float = 1.0
    # Recon weighting
    use_magnitude_weight: bool = True
    weight_alpha: float = 2.0             # max extra weight on top of 1.0
    weight_tau: float = 4.0               # F6 magnitude scale (≈sqrt(T*5*6) for unit-norm)
    # Misc
    init_mode: str = "uniform"

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "TactileVQVAEConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class TactileVQVAE(nn.Module):
    def __init__(self, cfg: TactileVQVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = F6Encoder(
            window=cfg.window,
            in_channels=cfg.in_channels,
            hidden_channels=cfg.hidden_channels,
            bottleneck_channels=cfg.bottleneck_channels,
            embed_dim=cfg.embed_dim,
            n_strided_blocks=cfg.n_strided_blocks,
        )
        self.quantizer = VQEMAQuantizer(
            codebook_size=cfg.codebook_size,
            embed_dim=cfg.embed_dim,
            commitment_weight=cfg.commitment_weight,
            decay=cfg.decay,
            revive_freq=cfg.revive_freq,
            revive_threshold=cfg.revive_threshold,
            init_mode=cfg.init_mode,
        )
        self.decoder = F6Decoder(
            window=cfg.window,
            out_channels=cfg.in_channels,
            hidden_channels=cfg.hidden_channels,
            bottleneck_channels=cfg.bottleneck_channels,
            embed_dim=cfg.embed_dim,
            n_strided_blocks=cfg.n_strided_blocks,
        )

    def _recon_weight(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Per-sample weight = 1 + α·sigmoid(magnitude/τ - 1).

        magnitude is the L2 norm of the *raw* (un-normalized) F6 window.
        Free-air windows (magnitude ≈ 0) get weight ≈ 1; high-contact windows
        approach 1 + α. Cheap, monotonic, and avoids hard-thresholding which
        can cause loss-discontinuity artifacts at the boundary.
        """
        cfg = self.cfg
        return 1.0 + cfg.weight_alpha * torch.sigmoid(magnitude / cfg.weight_tau - 1.0)

    def forward(
        self,
        f6: torch.Tensor,
        magnitude: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        f6        : [B, T, 5, 6]  normalized F6 window
        magnitude : [B]            raw F6 magnitude (for loss weighting). Optional.

        Returns dict:
            recon, indices, recon_loss, vq_loss, total_loss, plus quantizer info.
        """
        z_e = self.encoder(f6)                                        # [B, D]
        z_q, indices, vq_loss, qinfo = self.quantizer(z_e)            # [B, D], [B], scalar
        recon = self.decoder(z_q)                                     # [B, T, 5, 6]

        per_sample = (recon - f6).pow(2).mean(dim=[1, 2, 3])          # [B]
        if self.cfg.use_magnitude_weight and magnitude is not None:
            w = self._recon_weight(magnitude.to(per_sample.device))
            recon_loss = (per_sample * w).sum() / (w.sum() + 1e-8)
        else:
            recon_loss = per_sample.mean()

        total_loss = recon_loss + vq_loss

        return {
            "recon":          recon,
            "indices":        indices,
            "recon_loss":     recon_loss,
            "vq_loss":        vq_loss,
            "total_loss":     total_loss,
            "perplexity":     torch.tensor(qinfo["perplexity"], device=f6.device),
            "active_codes":   torch.tensor(qinfo["active_codes"], device=f6.device),
            "revived":        torch.tensor(qinfo["revived"], device=f6.device),
            "per_sample_recon": per_sample.detach(),
        }

    @torch.no_grad()
    def encode(self, f6: torch.Tensor) -> torch.Tensor:
        """Inference: f6 [B, T, 5, 6] → indices [B]."""
        z_e = self.encoder(f6)
        return self.quantizer.encode_only(z_e)

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """indices [B] → recon [B, T, 5, 6]."""
        z_q = self.quantizer.embed[indices]
        return self.decoder(z_q)
