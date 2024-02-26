import torch, torch.nn as nn
from lcasr.components.rotary_emb import RotaryPositionalEmbedding, apply_rotary
import math
from einops import rearrange
from torch import einsum

class LearnableFourierPosEnc(torch.nn.Module): # code taken from espnet: https://espnet.github.io/espnet/_modules/espnet/nets/pytorch_backend/transformer/embedding.html#LearnableFourierPosEnc
    """Learnable Fourier Features for Positional Encoding.

    See https://arxiv.org/pdf/2106.02795.pdf

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        gamma (float): init parameter for the positional kernel variance
            see https://arxiv.org/pdf/2106.02795.pdf.
        apply_scaling (bool): Whether to scale the input before adding the pos encoding.
        hidden_dim (int): if not None, we modulate the pos encodings with
            an MLP whose hidden layer has hidden_dim neurons.
    """

    def __init__(
        self,
        d_model,
        dropout_rate=0.0,
        gamma=1.0,
        apply_scaling=False,
        hidden_dim=None,
    ):
        """Initialize class."""
        super(LearnableFourierPosEnc, self).__init__()

        self.d_model = d_model

        if apply_scaling:
            self.xscale = math.sqrt(self.d_model)
        else:
            self.xscale = 1.0

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.gamma = gamma
        if self.gamma is None:
            self.gamma = self.d_model // 2

        assert (
            d_model % 2 == 0
        ), "d_model should be divisible by two in order to use this layer."
        self.w_r = torch.nn.Parameter(torch.empty(1, d_model // 2))
        self._reset()  # init the weights

        self.hidden_dim = hidden_dim
        if self.hidden_dim is not None:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(d_model, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, d_model),
            )

    def _reset(self):
        self.w_r.data = torch.normal(
            0, (1 / math.sqrt(self.gamma)), (1, self.d_model // 2)
        )

    def extend_pe(self, x, lengths, position_offsets):
        """Reset the positional encodings."""
        start, end = position_offsets.min().item(), (position_offsets + lengths).max().item()
        position_v = torch.arange(start, end, dtype=torch.float32, device=x.device).unsqueeze(1)

        cosine = torch.cos(torch.matmul(position_v, self.w_r))
        sine = torch.sin(torch.matmul(position_v, self.w_r))
        pos_enc = torch.cat((cosine, sine), -1)
        pos_enc /= math.sqrt(self.d_model)

        if self.hidden_dim is None:
            pos_enc =  pos_enc.unsqueeze(0)
        else:
            pos_enc = self.mlp(pos_enc.unsqueeze(0))
        if (position_offsets == position_offsets[0]).all():
            return pos_enc
        else:
            zero_index = torch.zeros(1, 1, self.d_model, device=x.device)
            pos_enc = torch.cat((pos_enc, zero_index), dim=1)
            indexes = torch.arange(0, lengths.max(), device=x.device)[None].expand(x.size(0), -1) + position_offsets[:, None]
            indexes = indexes.clamp(start, end)
            pos_enc = pos_enc.expand(x.size(0), -1, -1).gather(1, indexes[:,:,None].expand(-1, -1, pos_enc.size(-1)))
            return pos_enc


    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None, position_offsets: torch.Tensor = None):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        lengths = torch.LongTensor([x.size(1)] * x.size(0)).to(x) if lengths is None else lengths
        position_offsets = torch.LongTensor([0] * x.size(0)).to(x) if position_offsets is None else position_offsets
        pe = self.extend_pe(x, lengths, position_offsets)
        x = x * self.xscale + pe
        return self.dropout(x)
    
class ScaledSinuEmbedding(nn.Module):
    '''taken From Phil Wang's x-transformers library'''

    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1,))
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n, device=device).type_as(self.inv_freq)
        sinu = einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim=-1)
        return emb * self.scale + x

class DynamicPositionBias(nn.Module):
    '''Adapted from Phil Wang's x-transformers library'''

    def __init__(self, dim, *, heads, depth, log_distance=False, activation=nn.SiLU):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.log_distance = log_distance

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(1, dim),
            activation()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                activation()
            ))

        self.mlp.append(nn.Linear(dim, heads))

    def forward(self, pos, indices, device, dtype):
        pos = pos.to(device=device, dtype=dtype)

        if self.log_distance:
            # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)

        for layer in self.mlp:
            pos = layer(pos)

        bias = pos[indices]
        # print(bias.shape)
        bias = rearrange(bias, 'b i j h -> b h i j')
        return bias