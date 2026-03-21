import torch
import timm
import numpy as np
from torch import nn

from .point_encoder import PointcloudEncoder, Encoder, Group

class GroupNoRGB(Group):
    def forward(self, xyz, color=None): 
        batch_size, num_points, _ = xyz.shape
        center = self.fps(xyz, self.num_group) # B G 3
        idx = self.knn_point(self.group_size, xyz, center) # B G M
        
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        neighborhood = neighborhood - center.unsqueeze(2)

        features = neighborhood # B G M 3
        
        return neighborhood, center, features

class EncoderNoRGB(Encoder):
    def __init__(self, encoder_channel):
        super().__init__(encoder_channel)
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1), # <--- 6 变 3
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

    def forward(self, point_groups):
        bs, g, n, c = point_groups.shape
        
        point_groups = point_groups.reshape(bs * g, n, c)
        
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 128 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 128 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1) # BG 256 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        
        return feature_global.reshape(bs, g, self.encoder_channel)


class Uni3D(PointcloudEncoder):
    def __init__(self, point_transformer, args):
        super().__init__(point_transformer, args)
        
        self.group_divider = GroupNoRGB(num_group=args.num_group, group_size=args.group_size)
        self.encoder = EncoderNoRGB(encoder_channel=args.pc_encoder_dim)
        
        from .point_encoder import fps, knn_point
        self.group_divider.fps = fps
        self.group_divider.knn_point = knn_point

    def forward(self, pts, colors=None):
        """
        Args:
            pts: [B, N, 3] 只有坐标
            colors: 忽略
        """
        # 1. Grouping (No RGB)
        # features 维度现在是 [B, G, M, 3]
        _, center, features = self.group_divider(pts, None)

        # 2. Encoding
        # EncoderNoRGB 接受 3 通道输入
        group_input_tokens = self.encoder(features) 
        group_input_tokens = self.encoder2trans(group_input_tokens)

        # 3. Transformer (后续逻辑完全不变)
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        
        x = x + pos
        x = self.patch_dropout(x)
        x = self.visual.pos_drop(x)

        for i, blk in enumerate(self.visual.blocks):
            x = blk(x)

        x = self.visual.norm(x)
        if hasattr(self.visual, 'fc_norm'):
             x = self.visual.fc_norm(x)
        
        x = self.trans2embed(x) 

        return x[:, 1:, :], center
