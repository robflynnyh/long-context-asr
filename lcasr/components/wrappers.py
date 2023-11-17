import torch.nn as nn, torch.nn.functional as F
from apex.normalization import FusedRMSNorm as DEFAULT_NORM

class PreNorm(nn.Module): # applies normalization before fn
    def __init__(self, d_model, fn, norm = DEFAULT_NORM, sandwich_norm = False):
        super().__init__()
        self.norm = norm(d_model)
        self.fn = fn
        self.sandwich_norm = sandwich_norm
        if self.sandwich_norm:
            self.norm_out = norm(d_model)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        if self.sandwich_norm:
            x = self.norm_out(x)
        return x

class Scale(nn.Module): # scales output of fn by scale
    def __init__(self, scale, fn):
        super().__init__()
        self.scale = scale
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale
