'''
File Description: attention layer for transformer. borrowed from TIMM
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from torch import Tensor
import logging
from . import Mlp, DropPath, trunc_normal_, lecun_normal_
from . import create_norm, create_act
from ..peft_module.adapter import AdapterSuper_noout
from ..backbone.Point_PN import square_distance, index_points
from torch_geometric.nn.pool import voxel_grid
from collections import OrderedDict
from torch_scatter import scatter_max, scatter_mean
from torch_scatter import segment_csr
import math
import random

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        # this is different from what I understand before. 
        # the num_heads here actually works as groups for shared attentions. it partition the channels to different groups.
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        ########3
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple), shape [B, #Heads, N, C]
        ########
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_ratio=4., init_values=1e-4,adapter_dim=16, drop_rate_adapter=0.0,num_view=6):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_hidden_dim)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(mlp_hidden_dim, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)


        self.adapter = AdapterSuper_noout(embed_dims=d_model, reduction_dims=adapter_dim, drop_rate_adapter=drop_rate_adapter)
        self.out_transform_3d = nn.Sequential(nn.BatchNorm1d(d_model), nn.GELU())
        self.out_transform_1d = nn.ModuleList([nn.Sequential(nn.BatchNorm1d(d_model), nn.GELU()) for i in range(num_view)])

    def attention(self, x: torch.Tensor):
        return self.attn(x, x, x, need_weights=True, attn_mask=None)

    def forward(self, x: torch.Tensor,center1=None, idx_ptr=None, 
                sorted_cluster_indices=None, cluster=None,grid_shape=None, 
                mask=None, flat_grid_index=None,attn1=None, norm3=None,
                scale_factor=0.3,maxmean_1d=None,coef_pro=0.3):

        x_attn,attn_weight = self.attention(self.ln_1(x))
        x = x + x_attn 
        x_ffn = self.mlp(self.ln_2(x))
        x = x + x_ffn + scale_factor*self.adapter(x_ffn) 
        

        pospara_list = []
        B,G,_ = x.shape
        cls_x = x[:,0]
        x = x[:,1:]
        G = G-1

        feat = x.reshape(-1,x.shape[-1])
        feat_max = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        feat_mean = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="mean")
        x_3d = self.out_transform_3d((feat_max+feat_mean))[cluster].reshape(x.shape)

        for i in range(center1.shape[1]):
            main_center = center1[:,i,:]
            flat_x = x.reshape(-1, x.shape[-1])
            if attn1 is not None:  # 新增判空
                flat_x = attn1[i](norm3[i](flat_x.clone()), mask=mask[i]) + flat_x.clone()
            else:
                flat_x = flat_x.clone()  # 直接跳过注意力增强
            if maxmean_1d:
                max_feature, max_indices = scatter_max(flat_x, flat_grid_index[i], dim=0, dim_size=main_center.shape[0] * grid_shape)
                mean_feature = scatter_mean(flat_x, flat_grid_index[i], dim=0, dim_size=main_center.shape[0] * grid_shape)
                d2_view_post_feat = self.out_transform_1d[i]((mean_feature+max_feature))[flat_grid_index[i]].view(x.shape)
            else:
                mean_feature = scatter_mean(flat_x, flat_grid_index[i], dim=0, dim_size=main_center.shape[0] * grid_shape)
                d2_view_post_feat = self.out_transform_1d[i]((mean_feature))[flat_grid_index[i]].view(x.shape)

            pospara_list.append(d2_view_post_feat)
        x_sup = torch.stack(pospara_list,0)
        x_sup = x_sup.transpose(0,1)

        cosine_similarities = [(F.cosine_similarity(tensor_1d, x_3d, dim=-1)+1)/2 for tensor_1d in pospara_list]
        stacked_similarities = torch.stack(cosine_similarities, dim=0).to(x_sup.device)
        stacked_similarities = stacked_similarities.transpose(0,1)
        stacked_similarities /= stacked_similarities.sum(dim=1, keepdim=True)
        x_sup_weight = torch.sum(x_sup * stacked_similarities.unsqueeze(-1), dim=1)

        x = x+coef_pro*x_sup_weight
        x = torch.cat((cls_x.unsqueeze(1), x), 1)

        return x,attn_weight

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_args={'act': 'gelu'}, norm_args={'norm': 'ln'}):
        super().__init__()
        self.norm1 = create_norm(norm_args, dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = create_norm(norm_args, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # invertable bottleneck layer
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_args=act_args, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 act_args={'act': 'gelu'}, norm_args={'norm': 'ln'}
                 ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                norm_args=norm_args, act_args=act_args
            )
            for i in range(depth)])
        self.depth = depth

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            # x = block(x)

            # Injecting the positional information at each block.
            #  (Dehghani et al., 2018) and (Lan et al.,2019) observe better performance
            #  by further injecting the position information at each block
            # Reference: Learning to Encode Position for Transformer with Continuous Dynamical Model.
            # http://proceedings.mlr.press/v119/liu20n/liu20n.pdf
            x = block(x + pos)
        return x

    def forward_features(self, x, pos, num_outs=None):
        dilation = self.depth // num_outs
        out_depth = list(range(self.depth))[(self.depth - (num_outs-1)*dilation -1) :: dilation]
        
        out = []
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in out_depth:
                out.append(x)
        return out
