import torch.nn as nn, torch.nn.functional as F

class swiglu(nn.Module):
    def __init__(self, dim, exp_f=2, dim_out=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.exp_f = exp_f
        self.ff_in = nn.Linear(dim, dim*exp_f*2)
        self.ff_out = nn.Linear(dim*exp_f, self.dim_out)

    def forward(self, x):
        a, b = self.ff_in(x).chunk(2, dim=-1)
        return self.ff_out(F.silu(a) * b)