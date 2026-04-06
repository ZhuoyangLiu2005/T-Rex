"""
Future visual prediction module for the latent expert.

Adds K learnable future query tokens to the latent expert sequence.
After passing through the MoT backbone, these tokens are supervised
to predict frozen ViT features of future frames (cosine similarity).

At inference, the queries act as "planning tokens" — the latent expert
fills them with its prediction of the future, and the action expert
reads them via joint attention.  No decoding needed.

Usage:
    from qwen_vla.future_prediction import FuturePredictionHead, encode_future_frames

    # Attach to an existing Qwen3VLVLAModel
    future_head = FuturePredictionHead(hidden_size=2048, n_future_tokens=8)

    # Training: append queries to slow_embeds, extract predictions, compute loss
    # Inference: append queries to slow_embeds, pass to forward_flow as-is
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class FuturePredictionHead(nn.Module):
    """
    Learnable future query tokens + projection head.

    Parameters
    ----------
    hidden_size : int
        Transformer hidden dimension (must match the MoT backbone).
    n_future_tokens : int
        Number of future prediction tokens (typically = action_chunk).
    proj_hidden : int or None
        Hidden dim for the 2-layer projection MLP.  None = same as hidden_size.
    """

    def __init__(self, hidden_size: int, n_future_tokens: int = 8,
                 proj_hidden: int = None):
        super().__init__()
        self.n_future_tokens = n_future_tokens
        self.hidden_size = hidden_size

        # Learnable query tokens [1, K, H] — small random init
        self.future_queries = nn.Parameter(
            torch.randn(1, n_future_tokens, hidden_size) * 0.02
        )

        # Projection: transformer hidden → ViT feature space (same dim)
        ph = proj_hidden or hidden_size
        self.future_proj = nn.Sequential(
            nn.Linear(hidden_size, ph),
            nn.GELU(),
            nn.Linear(ph, hidden_size),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.future_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_query_tokens(self, batch_size: int) -> torch.Tensor:
        """Return [B, K, H] query tokens expanded for the batch."""
        return self.future_queries.expand(batch_size, -1, -1)

    def project(self, future_hidden: torch.Tensor) -> torch.Tensor:
        """Project future query hidden states → prediction space. [B, K, H] → [B, K, H]."""
        return self.future_proj(future_hidden)

    def compute_loss(self, future_pred: torch.Tensor,
                     future_targets: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity loss between predictions and targets.

        Parameters
        ----------
        future_pred : [B, K, H]  predicted future features
        future_targets : [B, K, H]  frozen ViT features of future frames

        Returns
        -------
        Scalar loss (1 - mean cosine similarity).
        """
        # Normalize for cosine similarity
        pred_norm = F.normalize(future_pred, dim=-1)
        tgt_norm = F.normalize(future_targets.detach(), dim=-1)
        cos_sim = (pred_norm * tgt_norm).sum(dim=-1)  # [B, K]
        return (1.0 - cos_sim).mean()




@torch.no_grad()
def encode_future_frames(
    visual_encoder: nn.Module,
    processor,
    future_pil_images: list,
    image_size: tuple = None,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Encode future frames through the frozen ViT and global-average-pool per frame.

    Parameters
    ----------
    visual_encoder : the model's .visual module (frozen)
    processor : Qwen3-VL processor (for image processing)
    future_pil_images : list of B*K PIL images (B samples, K future frames each)
    image_size : (W, H) resize target, or None
    device : target device
    dtype : target dtype

    Returns
    -------
    [B*K, H]  one feature vector per future frame (global avg pooled)
    """
    if not future_pil_images:
        return torch.empty(0, device=device, dtype=dtype)

    if image_size is not None:
        import PIL.Image
        future_pil_images = [img.resize(image_size, PIL.Image.LANCZOS)
                             for img in future_pil_images]

    # Process through image processor
    img_inp = processor.image_processor(future_pil_images, return_tensors="pt")
    pixel_values = img_inp.pixel_values.to(device=device, dtype=dtype)
    grid_thw = img_inp.image_grid_thw.to(device=device)

    # Run frozen ViT
    vit_out = visual_encoder(pixel_values, grid_thw=grid_thw)
    features = vit_out[0] if isinstance(vit_out, (tuple, list)) else vit_out
    # features: [total_merged_tokens, H]

    # Global average pool per frame
    merge = getattr(visual_encoder, "spatial_merge_size", 2)
    frame_features = []
    offset = 0
    for g in grid_thw:
        n_tokens = int(g[0] * (g[1] // merge) * (g[2] // merge))
        frame_feat = features[offset: offset + n_tokens].mean(dim=0)  # [H]
        frame_features.append(frame_feat)
        offset += n_tokens

    return torch.stack(frame_features)  # [B*K, H]
