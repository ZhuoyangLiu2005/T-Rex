"""VQ-EMA quantizer with dead-code revival.

Standard exponential-moving-average codebook (van den Oord 2017 + Razavi 2019)
plus a periodic check that replaces under-used codes with random samples from
the current batch's encoder outputs — important when training on F6, where
the free-air "near-zero" cluster otherwise swamps the codebook.

Distributed-aware: EMA stats are all-reduced across ranks before the codebook
update, so the codebook stays consistent.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if _is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def _all_gather_2d(x: torch.Tensor) -> torch.Tensor:
    """All-gather a [B, D] tensor along dim 0; returns [W*B, D] (W=world_size)."""
    if not _is_dist():
        return x
    world = dist.get_world_size()
    out = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(out, x.contiguous())
    return torch.cat(out, dim=0)


class VQEMAQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int = 1024,
        embed_dim: int = 256,
        commitment_weight: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        revive_freq: int = 200,                 # in update steps
        revive_threshold: float = 1.0,          # codes with cluster_size < this are dead
        init_mode: str = "uniform",             # "uniform" or "kaiming"
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.eps = eps
        self.revive_freq = revive_freq
        self.revive_threshold = revive_threshold

        # Codebook init.
        if init_mode == "kaiming":
            embed = torch.empty(codebook_size, embed_dim)
            nn.init.kaiming_uniform_(embed, a=5 ** 0.5)
        else:
            embed = torch.randn(codebook_size, embed_dim) * 0.02

        self.register_buffer("embed", embed)                          # [K, D]
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embed.clone())               # [K, D]
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _ema_update(self, z_e: torch.Tensor, indices: torch.Tensor):
        """EMA-update codebook stats. z_e: [N, D]; indices: [N]."""
        K = self.codebook_size
        D = self.embed_dim

        # Local one-hot sums.
        flat_idx = indices.view(-1)
        onehot = F.one_hot(flat_idx, K).type_as(z_e)                  # [N, K]
        local_cluster = onehot.sum(dim=0)                             # [K]
        local_embed   = onehot.t() @ z_e                              # [K, D]

        # Aggregate across DDP ranks.
        local_cluster = _all_reduce_sum(local_cluster.contiguous())
        local_embed   = _all_reduce_sum(local_embed.contiguous())

        # EMA update.
        self.cluster_size.mul_(self.decay).add_(local_cluster, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(local_embed, alpha=1 - self.decay)

        # Laplace-smoothed normalization.
        n = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (n + K * self.eps) * n
        self.embed.copy_(self.embed_avg / smoothed.unsqueeze(1))

    @torch.no_grad()
    def _revive_dead_codes(self, z_e_pool: torch.Tensor):
        """Replace under-used codes with random samples from z_e_pool.

        z_e_pool: [N, D] — gathered across ranks; use the global pool so all
        ranks pick consistent replacements.
        """
        K = self.codebook_size
        D = self.embed_dim

        dead = self.cluster_size < self.revive_threshold              # [K] bool
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0

        # Gather across ranks so every rank sees the same pool.
        pool = _all_gather_2d(z_e_pool)
        N = pool.shape[0]
        if N == 0:
            return 0

        # Rank 0 picks indices, broadcast for consistency.
        if not _is_dist() or dist.get_rank() == 0:
            sel = torch.randint(0, N, (n_dead,), device=pool.device)
        else:
            sel = torch.zeros(n_dead, dtype=torch.long, device=pool.device)
        if _is_dist():
            dist.broadcast(sel, src=0)

        replacements = pool[sel].to(self.embed.dtype)                 # [n_dead, D]
        dead_idx = dead.nonzero(as_tuple=False).flatten()
        self.embed[dead_idx]      = replacements
        self.embed_avg[dead_idx]  = replacements
        self.cluster_size[dead_idx] = self.revive_threshold * 2.0     # give them a head start
        return n_dead

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        z_e: [B, D]
        Returns
        -------
        z_q_st : [B, D]   straight-through quantized (gradient flows to encoder)
        indices: [B]      long
        vq_loss: scalar   commitment loss (encoder side)
        info   : dict     {perplexity, active_codes, revived}
        """
        if z_e.dim() != 2 or z_e.shape[1] != self.embed_dim:
            raise ValueError(f"Expected z_e [B, {self.embed_dim}], got {tuple(z_e.shape)}")

        # Distances ‖e - z‖² = ‖e‖² + ‖z‖² - 2 e·z
        z_sq = z_e.pow(2).sum(dim=1, keepdim=True)                    # [B, 1]
        e_sq = self.embed.pow(2).sum(dim=1)                           # [K]
        ez   = z_e @ self.embed.t()                                   # [B, K]
        dists = z_sq + e_sq - 2.0 * ez                                # [B, K]
        indices = dists.argmin(dim=1)                                 # [B]
        z_q = self.embed[indices]                                     # [B, D]

        # Straight-through estimator.
        z_q_st = z_e + (z_q - z_e).detach()
        # Commitment loss only (codebook updated via EMA, not gradient).
        vq_loss = self.commitment_weight * F.mse_loss(z_e, z_q.detach())

        revived = 0
        if self.training:
            self._ema_update(z_e.detach(), indices)
            self.step += 1
            if int(self.step.item()) % self.revive_freq == 0:
                revived = self._revive_dead_codes(z_e.detach())

        # Diagnostics (always on; cheap).
        with torch.no_grad():
            onehot = F.one_hot(indices, self.codebook_size).type_as(z_e)   # [B, K]
            local_count = onehot.sum(dim=0)                                # [K]
            global_count = _all_reduce_sum(local_count.clone())
            probs = global_count / (global_count.sum() + 1e-12)
            active = int((global_count > 0).sum().item())
            perplexity = torch.exp(-(probs * (probs.add(1e-12)).log()).sum()).item()

        info = {
            "perplexity":   float(perplexity),
            "active_codes": active,
            "revived":      int(revived),
        }
        return z_q_st, indices, vq_loss, info

    @torch.no_grad()
    def encode_only(self, z_e: torch.Tensor) -> torch.Tensor:
        """Inference helper: nearest-code lookup, no EMA / loss."""
        z_sq = z_e.pow(2).sum(dim=1, keepdim=True)
        e_sq = self.embed.pow(2).sum(dim=1)
        ez   = z_e @ self.embed.t()
        return (z_sq + e_sq - 2.0 * ez).argmin(dim=1)
