import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class AdapterSuper(nn.Module):
    def __init__(self,
                 embed_dims,
                 reduction_dims,
                 drop_rate_adapter=0
                        ):
        super(AdapterSuper, self).__init__()
    
        self.embed_dims = embed_dims

        # Follow towards unified
        self.super_reductuion_dim = reduction_dims
        self.dropout = nn.Dropout(p=drop_rate_adapter)

        if self.super_reductuion_dim > 0:
            self.ln1 = nn.Linear(self.embed_dims, self.super_reductuion_dim)
            self.activate = QuickGELU()
            self.ln2 = nn.Linear(self.super_reductuion_dim, self.embed_dims)
            self.init_weights()
        
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)


        self.apply(_init_weights)

    def forward(self, x, identity=None):
        out = self.ln1(x)
        out = self.activate(out)
        out = self.dropout(out)
        out = self.ln2(out)
        if identity is None:
            identity = x

        return identity + out
    
class AdapterSuper_noout(nn.Module):
    def __init__(self,
                 embed_dims,
                 reduction_dims,
                 drop_rate_adapter=0
                        ):
        super(AdapterSuper_noout, self).__init__()
    
        self.embed_dims = embed_dims

        # Follow towards unified
        self.super_reductuion_dim = reduction_dims
        self.dropout = nn.Dropout(p=drop_rate_adapter)

        if self.super_reductuion_dim > 0:
            self.ln1 = nn.Linear(self.embed_dims, self.super_reductuion_dim)
            self.activate = QuickGELU()
            self.ln2 = nn.Linear(self.super_reductuion_dim, self.embed_dims)
            self.init_weights()
        
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)


        self.apply(_init_weights)

    def forward(self, x, identity=None):
        out = self.ln1(x)
        out = self.activate(out)
        out = self.dropout(out)
        out = self.ln2(out)

        if identity is None:
            identity = x

        return out
