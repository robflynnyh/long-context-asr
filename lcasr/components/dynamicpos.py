'''
Dynamic Positional encodings taken from: https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py @lucidrains 
'''
import torch.nn as nn, torch
from einops import rearrange


class DynamicPositionBias(nn.Module):
    def __init__(self, dim, *, heads, depth, log_distance = False, norm = False, activation=nn.ReLU):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.log_distance = log_distance

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(1, dim),
            nn.LayerNorm(dim) if norm else nn.Identity(),
            activation()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim) if norm else nn.Identity(),
                activation()
            ))

        self.mlp.append(nn.Linear(dim, heads))

    def forward(self, n, device, dtype):

        # get the (n x n) matrix of distances
        seq_arange = torch.arange(n, device = device)
        context_arange = torch.arange(n, device = device)
        indices = rearrange(seq_arange, 'i -> i 1') - rearrange(context_arange, 'j -> 1 j')
        indices += (n - 1)
        
        # input to continuous positions MLP
        pos = torch.arange(-n + 1, n, device = device, dtype = dtype)
        pos = rearrange(pos, '... -> ... 1')

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # get position biases        
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias


class DynamicPositionBiasXL(nn.Module):
    '''Adapted From Phil Wang's x-transformers library
       Altered to work with attention matrix that is not square
    '''
    def __init__(self, dim, *, heads, depth, log_distance = False, norm = False, init_history_decay = 1.0, activation=nn.SiLU):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.log_distance = log_distance

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(1, dim),
            nn.LayerNorm(dim) if norm else nn.Identity(),
            activation()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim) if norm else nn.Identity(),
                activation()
            ))
        
        self.mlp.append(nn.Linear(dim, heads))
        self.history_decay = nn.Parameter(torch.ones(heads, 1, 1) * init_history_decay)
      

    @staticmethod
    def fetch_module_kwargs(kwargs):
        return {
            'dim': kwargs.get('dpos_dim', 64),
            'depth': kwargs.get('dpos_depth', 2),
            'log_distance': kwargs.get('dpos_log_distance', False),
            'norm': kwargs.get('dpos_norm', False),
            'init_history_decay': kwargs.get('dpos_init_history_decay', 1.0)
        }

    def forward(self, i, j, device, dtype):
        # get the (i x j) matrix of distances
        assert i >= 1 and j >= 1 and i <= j, 'I should be in the range [1, j] and j >= 1'
      
        seq_arange = torch.arange(i, device = device)
        context_arange = torch.arange(j, device = device)
        indices = rearrange(seq_arange, 'i -> i 1') - rearrange(context_arange, 'j -> 1 j')
        indices += (j-1)
        
        # input to continuous positions MLP
        pos = torch.arange(-i + 1, (j+i), device = device, dtype = dtype)
        pos = rearrange(pos, '... -> ... 1')
     

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)
        
        for layer in self.mlp:
            pos = layer(pos)

        # get position biases        
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')

        if j > i:
            history_len = j - i
            bias[:, :, :history_len] *= self.history_decay

        return bias