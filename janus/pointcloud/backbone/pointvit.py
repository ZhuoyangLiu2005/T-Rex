import logging
from typing import List
import torch
import torch.nn as nn
import math
from ..layers import create_norm
from ..layers.attention import ResidualAttentionBlock
import numpy as np
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr
import torch.nn.functional as F
from ..peft_module.mv_utils import PCViews
# from pointnet2_ops import pointnet2_utils
from .Point_PN import Point_PN_scan

    
class PointTokenizer(nn.Module):
    def __init__(self,
                 in_channels=3,
                 embed_dim=768, depth=12,
                 num_heads=6, mlp_ratio=4.,
                 target_token_count=256,
                 norm_args={'norm': 'ln', 'eps': 1.0e-6},
                 **kwargs
            ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        
        self.patch_embed = Point_PN_scan()
        if self.patch_embed.out_channels != self.embed_dim: 
            self.proj = nn.Linear(self.patch_embed.out_channels, self.embed_dim)
        else:
            self.proj = nn.Identity() 
        self.proj = nn.Linear(384, 768) 
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        
        self.pos_embed = nn.Parameter(torch.zeros(1, target_token_count + 1, self.embed_dim))

        self.norm = create_norm(norm_args, self.embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, p, x=None, **kwargs):
        device = next(self.parameters()).device
        if hasattr(p, 'keys'): 
            p, x = p['pos'], p['x'] if 'x' in p.keys() else None
        if x is None:
            x = p.clone().transpose(1, 2).contiguous()
        if isinstance(p, np.ndarray):
            p = torch.from_numpy(p).float().to(device)
        else:
            p = p.float().to(device)
        x = x.float().to(device) 
            
        if isinstance(p, np.ndarray):
            p = torch.from_numpy(p).float().to(self.device) 
        elif p.dtype != torch.float32:
            p = p.float()  
            
        center, group_input_tokens, center_idx, neighbor_idx, post_center = self.patch_embed(x,p)
        # print(f"embed: {self.embed_dim}")  # Debugging line
        # print(f"Center shape: {center.shape}")  # Debugging line
        # print(f"Group input tokens shape: {group_input_tokens.shape}")  # Debugging line
        center_p, patch_tokens = center, self.proj(group_input_tokens.transpose(1, 2))
        
        return patch_tokens, center
