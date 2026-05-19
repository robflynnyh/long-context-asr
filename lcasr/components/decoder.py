import torch.nn as nn, torch.nn.functional as F, torch
try: from apex.normalization import FusedRMSNorm as DEFAULT_NORM
except: from lcasr.components.normalisation import RMSNorm as DEFAULT_NORM
from einops import rearrange

class ASRLinearSCDecoder(nn.Module):
    def __init__(
            self, 
            d_model, 
            vocab_size, 
            norm=False, 
            norm_fn=DEFAULT_NORM,
            **kwargs
        ):
        super().__init__()
        # Add 1 for blank char
        self.num_classes = vocab_size + 1
        self.ff = nn.Linear(d_model, self.num_classes)
        self.reprojection = nn.Linear(self.num_classes, d_model)
        self.norm = norm_fn(d_model) if norm else nn.Identity()

    def forward(self, x, logits=False):
        x_norm = self.norm(x)
        x = self.ff(x_norm)
        x = F.log_softmax(x, dim=-1) if not logits else x
        return x        

    def project_back(self, x):
        return self.reprojection(x)

    def integrate_projections(self, x, proj1):
        return x + proj1