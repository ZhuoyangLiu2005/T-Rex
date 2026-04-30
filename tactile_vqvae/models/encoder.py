"""F6Encoder — 1-D conv encoder over a per-hand F6 window.

Input  : [B, T, 5, 6]  (T frames, 5 fingers, 6 F/T dims)
Output : [B, embed_dim]  (one continuous code per window)

Strided convs downsample along the time axis; finger and dim are folded into
the channel dim. Strides are skipped when T is too small to halve.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int, kernel: int, stride: int) -> nn.Sequential:
    pad = kernel // 2
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.GELU(),
    )


class F6Encoder(nn.Module):
    def __init__(
        self,
        window: int = 16,
        in_channels: int = 30,         # 5 fingers × 6 dims
        hidden_channels: int = 128,
        bottleneck_channels: int = 256,
        embed_dim: int = 256,
        n_strided_blocks: int = 2,     # each halves time
    ):
        super().__init__()
        self.window = window
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Stem: keep time, project to hidden_channels
        self.stem = _conv_block(in_channels, hidden_channels, kernel=5, stride=1)

        # Strided blocks (skip stride if T is too small after current block)
        blocks: List[nn.Module] = []
        cur_T = window
        cur_ch = hidden_channels
        for i in range(n_strided_blocks):
            stride = 2 if cur_T >= 4 else 1
            out_ch = bottleneck_channels if i == n_strided_blocks - 1 else hidden_channels
            blocks.append(_conv_block(cur_ch, out_ch, kernel=5, stride=stride))
            cur_ch = out_ch
            cur_T = cur_T // stride if stride > 1 else cur_T
        self.strided = nn.Sequential(*blocks)
        self._bottleneck_T = cur_T   # frames left at the bottleneck

        # Final 1×1 projection to embed_dim (kept channel-wise so we can
        # mean-pool over the time axis afterwards).
        self.proj = nn.Conv1d(cur_ch, embed_dim, kernel_size=3, padding=1)

    @property
    def bottleneck_T(self) -> int:
        return self._bottleneck_T

    def forward(self, f6: torch.Tensor) -> torch.Tensor:
        """f6: [B, T, 5, 6]  →  z_e: [B, embed_dim]"""
        B, T, F, D = f6.shape
        if T != self.window:
            raise ValueError(f"Encoder built for window={self.window}, got T={T}")
        x = f6.reshape(B, T, F * D).transpose(1, 2).contiguous()   # [B, 30, T]
        x = self.stem(x)
        x = self.strided(x)
        x = self.proj(x)                                           # [B, E, T_bn]
        z_e = x.mean(dim=2)                                        # [B, E]
        return z_e
