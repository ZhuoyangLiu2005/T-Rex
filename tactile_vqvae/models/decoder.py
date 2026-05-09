"""F6 decoders — mirror of the encoders.

F6Decoder
    Input  : [B, embed_dim]
    Output : [B, T, 5, 6]

F6PerFingerDecoder
    Input  : [B, 5, embed_dim]
    Output : [B, T, 5, 6]
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def _upconv_block(in_ch: int, out_ch: int, kernel: int, stride: int) -> nn.Sequential:
    pad = kernel // 2
    if stride > 1:
        layer = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size=kernel, stride=stride,
            padding=pad, output_padding=stride - 1,
        )
    else:
        layer = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=1, padding=pad)
    return nn.Sequential(
        layer,
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.GELU(),
    )


class F6Decoder(nn.Module):
    def __init__(
        self,
        window: int = 16,
        out_channels: int = 30,
        hidden_channels: int = 128,
        bottleneck_channels: int = 256,
        embed_dim: int = 256,
        n_strided_blocks: int = 2,
    ):
        super().__init__()
        self.window = window
        self.embed_dim = embed_dim

        # Reproduce the encoder's strides to recover bottleneck length.
        cur_T = window
        strides: List[int] = []
        cur_ch_chain = [hidden_channels]
        for i in range(n_strided_blocks):
            stride = 2 if cur_T >= 4 else 1
            strides.append(stride)
            cur_ch_chain.append(
                bottleneck_channels if i == n_strided_blocks - 1 else hidden_channels)
            if stride > 1:
                cur_T //= stride
        self._bottleneck_T = cur_T

        # Project embed → bottleneck channels.
        self.from_embed = nn.Conv1d(embed_dim, bottleneck_channels, kernel_size=3, padding=1)

        # Reverse the strided stack.
        blocks: List[nn.Module] = []
        # cur_ch_chain[-1] = bottleneck_channels; reverse direction.
        in_ch = bottleneck_channels
        # Apply upsamples in reverse: stride[i] block was hidden→{hidden or bn},
        # so reverse goes {hidden or bn}→{prev_hidden}.
        rev_strides = list(reversed(strides))
        rev_ch_chain = list(reversed(cur_ch_chain))   # [bottleneck, ..., hidden]
        for i, st in enumerate(rev_strides):
            out_ch = rev_ch_chain[i + 1]
            blocks.append(_upconv_block(in_ch, out_ch, kernel=5, stride=st))
            in_ch = out_ch
        self.up_strided = nn.Sequential(*blocks)

        # Stem inverse: hidden_channels → out_channels
        self.head = nn.Conv1d(hidden_channels, out_channels, kernel_size=5, padding=2)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """z_q: [B, embed_dim]  →  recon: [B, T, 5, 6]"""
        B = z_q.shape[0]
        x = z_q.unsqueeze(2).expand(-1, -1, self._bottleneck_T)   # [B, E, T_bn]
        x = self.from_embed(x)                                    # [B, bn, T_bn]
        x = self.up_strided(x)                                    # [B, hidden, T]
        x = self.head(x)                                          # [B, 30, T]

        # Some stride/output_padding combos may yield T+1 or T-1; trim/pad.
        if x.shape[-1] != self.window:
            if x.shape[-1] > self.window:
                x = x[..., : self.window]
            else:
                pad = self.window - x.shape[-1]
                x = torch.nn.functional.pad(x, (0, pad))

        x = x.transpose(1, 2).contiguous()                        # [B, T, 30]
        x = x.reshape(B, self.window, 5, 6)
        return x


class F6PerFingerDecoder(nn.Module):
    """Per-finger decoder.  Mirror of F6PerFingerEncoder.

    Input  : [B, 5, embed_dim]
    Output : [B, T, 5, 6]
    """

    def __init__(
        self,
        window: int = 16,
        per_finger_dim: int = 6,
        n_fingers: int = 5,
        hidden_channels: int = 128,
        bottleneck_channels: int = 256,
        embed_dim: int = 256,
        n_strided_blocks: int = 2,
    ):
        super().__init__()
        self.window = window
        self.n_fingers = n_fingers
        self.embed_dim = embed_dim

        cur_T = window
        strides: List[int] = []
        cur_ch_chain = [hidden_channels]
        for i in range(n_strided_blocks):
            stride = 2 if cur_T >= 4 else 1
            strides.append(stride)
            cur_ch_chain.append(
                bottleneck_channels if i == n_strided_blocks - 1 else hidden_channels)
            if stride > 1:
                cur_T //= stride
        self._bottleneck_T = cur_T

        self.from_embed = nn.Conv1d(embed_dim, bottleneck_channels, kernel_size=3, padding=1)

        # Finger ID injected at the bottleneck side (added to from_embed output).
        self.finger_embed = nn.Embedding(n_fingers, bottleneck_channels)

        blocks: List[nn.Module] = []
        in_ch = bottleneck_channels
        rev_strides = list(reversed(strides))
        rev_ch_chain = list(reversed(cur_ch_chain))
        for i, st in enumerate(rev_strides):
            out_ch = rev_ch_chain[i + 1]
            blocks.append(_upconv_block(in_ch, out_ch, kernel=5, stride=st))
            in_ch = out_ch
        self.up_strided = nn.Sequential(*blocks)

        self.head = nn.Conv1d(hidden_channels, per_finger_dim, kernel_size=5, padding=2)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """z_q: [B, 5, embed_dim]  →  recon: [B, T, 5, 6]"""
        B, F, E = z_q.shape
        x = z_q.reshape(B * F, E).unsqueeze(2).expand(-1, -1, self._bottleneck_T)
        x = self.from_embed(x)                                    # [B*5, bn, T_bn]

        ids = torch.arange(F, device=z_q.device).repeat(B)        # [B*5]
        fid = self.finger_embed(ids).unsqueeze(-1)                # [B*5, bn, 1]
        x = x + fid

        x = self.up_strided(x)                                    # [B*5, hidden, T]
        x = self.head(x)                                          # [B*5, 6, T]

        if x.shape[-1] != self.window:
            if x.shape[-1] > self.window:
                x = x[..., : self.window]
            else:
                pad = self.window - x.shape[-1]
                x = torch.nn.functional.pad(x, (0, pad))

        x = x.transpose(1, 2).contiguous()                        # [B*5, T, 6]
        x = x.reshape(B, F, self.window, -1).permute(0, 2, 1, 3).contiguous()
        return x                                                  # [B, T, 5, 6]
