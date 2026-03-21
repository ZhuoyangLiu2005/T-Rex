from typing import List, Tuple
from torch.autograd import Function

import torch
import torch.nn as nn
import torch.nn.functional as F


def three_nearest_neighbors(unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the 3 nearest neighbors of `unknown` points in `known` points.
    Args:
        unknown: (B, N, 3) tensor
        known: (B, M, 3) tensor
    Returns:
        dist: (B, N, 3) l2 distances to neighbors
        idx: (B, N, 3) indices of neighbors
    """
    B, N, _ = unknown.shape
    M = known.shape[1]
    
    # Compute pairwise distances (B, N, M)
    dist = torch.cdist(unknown, known)
    
    # Find top-3 smallest distances
    dist, idx = torch.topk(dist, k=3, dim=2, largest=False)
    
    return dist, idx

class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist, idx = three_nearest_neighbors(unknown, known)  # 使用上述任一实现
        ctx.mark_non_differentiable(idx)
        return dist, idx

    @staticmethod
    def backward(ctx, grad_dist, grad_idx):
        return None, None

three_nn = ThreeNN.apply

def three_interpolate_forward(features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Weighted linear interpolation of 3 nearest neighbors.
    Args:
        features: (B, C, M)  # 原始特征
        idx: (B, N, 3)       # 最近邻索引
        weight: (B, N, 3)    # 权重（和为1）
    Returns:
        output: (B, C, N)    # 插值后的特征
    """
    B, C, M = features.shape
    N = idx.shape[1]
    
    # 扩展索引以匹配特征维度 (B, C, N, 3)
    idx_expanded = idx.unsqueeze(1).expand(-1, C, -1, -1)
    
    # 收集邻居特征 (B, C, N, 3)
    neighbor_features = torch.gather(
        features.unsqueeze(-1).expand(-1, -1, -1, 3),  # (B, C, M, 3)
        dim=2,
        index=idx_expanded
    )
    
    # 加权求和 (B, C, N)
    output = (neighbor_features * weight.unsqueeze(1)).sum(dim=-1)
    return output

def three_interpolate_backward(
    grad_out: torch.Tensor, 
    idx: torch.Tensor, 
    weight: torch.Tensor, 
    m: int
) -> torch.Tensor:
    """
    Compute gradients for features.
    Args:
        grad_out: (B, C, N)  # 输出梯度
        idx: (B, N, 3)       # 前向传播的索引
        weight: (B, N, 3)    # 前向传播的权重
        m: int               # 原始特征数量 M
    Returns:
        grad_features: (B, C, M)  # 特征梯度
    """
    B, C, N = grad_out.shape
    
    # 初始化梯度张量
    grad_features = torch.zeros((B, C, m), device=grad_out.device)
    
    # 扩展维度以匹配计算 (B, C, N, 3)
    grad_out_expanded = grad_out.unsqueeze(-1).expand(-1, -1, -1, 3)
    weight_expanded = weight.unsqueeze(1).expand(-1, C, -1, -1)
    
    # 使用 scatter_add_ 累加梯度
    grad_features.scatter_add_(
        dim=2,
        index=idx.unsqueeze(1).expand(-1, C, -1, -1),
        src=grad_out_expanded * weight_expanded
    )
    
    return grad_features


class ThreeInterpolate(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, features, idx, weight):
        output = three_interpolate_forward(features, idx, weight)  # 使用上述实现
        ctx.save_for_backward(idx, weight)
        ctx.m = features.size(2)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_out):
        idx, weight = ctx.saved_tensors
        grad_features = three_interpolate_backward(grad_out, idx, weight, ctx.m)  # 选择纯PyTorch或torch_scatter版本
        return grad_features, None, None

three_interpolate = ThreeInterpolate.apply


def three_interpolation(unknown_xyz, known_xyz, know_feat):
    """
    input: known_xyz: (m, 3), unknown_xyz: (n, 3), feat: (m, c), offset: (b), new_offset: (b)
    output: (n, c)
    """
    dist, idx = three_nn(unknown_xyz, known_xyz)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    interpolated_feats = three_interpolate(know_feat, idx, weight)
    return interpolated_feats


if __name__ == "__main__":
    pass
