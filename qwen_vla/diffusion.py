"""
Diffusion / flow-matching helper modules.
Copied and trimmed from janus/diffusion/models.py – only the VLA-relevant
pieces (ActionEmbedder, TimestepEmbedder, FinalLayer) are kept here.
"""

import math
import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp


class TimestepEmbedder(nn.Module):
    """Embeds scalar diffusion timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            next(self.mlp.parameters()).dtype
        )
        return self.mlp(t_freq)


class ActionEmbedder(nn.Module):
    """Projects continuous action / tactile vectors into the LLM hidden space."""

    def __init__(self, action_size: int, hidden_size: int):
        super().__init__()
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=action_size,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FinalLayer(nn.Module):
    """Projects the LLM hidden states back to the action space."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
