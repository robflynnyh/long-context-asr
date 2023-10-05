import torch


class RotaryPositionalEmbedding(torch.nn.Module): # TODO: incl fused kernel versions of rotary pos emb
    def __init__(
            self, 
            dim, 
            base=1000,
            rotary_heads=1, 
            learned_freq=False,
            rotary_interpolation_factor=1.0,
            precision=torch.bfloat16, 
        ):
        """Rotary positional embedding
        Reference : https://blog.eleuther.ai/rotary-embeddings/
        Paper: https://arxiv.org/pdf/2104.09864.pdf
        Adapted from: https://fairseq.readthedocs.io/en/latest/_modules/fairseq/modules/rotary_positional_embedding.html
        Args:
            dim: Dimension of embedding
            base: Base value for exponential
            precision: precision to use for numerical values
        """
        super().__init__()
        bases = torch.tensor([base+((i*2*base)**2)/100 for i  in range(rotary_heads)], dtype=precision)[:, None]
        inv_freq = 1.0 / (bases ** (torch.arange(0, dim, 2).float() / dim)[None,:].repeat(rotary_heads, 1))
        inv_freq = inv_freq if rotary_heads > 1 else inv_freq.squeeze(0) # back compatability with older checkpoints
        
        self.learned_freq = learned_freq

        if self.learned_freq:
            self.inv_freq = torch.nn.Parameter(inv_freq, requires_grad=True)
        else:
            self.register_buffer("inv_freq", inv_freq)

        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision
        # register rotary interpolation factor as buffer so it can be saved
        self.register_buffer("rotary_interpolation_factor", torch.tensor(rotary_interpolation_factor))

    def reset_if_needed(self):
        if self.learned_freq: # bcos we cant keep them after backward pass
            self.cos_cached = None
            self.sin_cached = None
            self.seq_len_cached = None
    
    def forward(self, seq_len, device=torch.device("cpu")):
        """
        Args:
            x: Input x with T X B X C
            seq_len: Sequence length of input x
        """
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq) / self.rotary_interpolation_factor
            freqs = torch.einsum("i,hj->ihj", t, self.inv_freq if self.inv_freq.ndim == 2 else self.inv_freq[None, :])
            emb = torch.cat((freqs, freqs), dim=-1).to(device)
            self.cos_cached = emb.cos()[None, :, :]
            self.sin_cached = emb.sin()[None, :, :]
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers: (this should all just be moved into rotary class tbh)
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in earlier torch versions


def apply_rotary_pos_emb(q, k, cos, sin, q_offset: int = 0):
    q_cos, q_sin = (
        cos[:, q_offset : q.shape[1] + q_offset],
        sin[:, q_offset : q.shape[1] + q_offset],
    )
    return (q * q_cos) + (rotate_half(q) * q_sin), (k * cos) + (rotate_half(k) * sin)

class apply_rotary(): 
    def __init__(self, cos, sin, q_offset: int = 0, learned: bool = False):
        self.learned = learned
        self.cos = cos
        self.sin = sin
        self.q_offset = q_offset
    
    def apply(self, q, k):
        return apply_rotary_pos_emb(q, k, self.cos, self.sin, self.q_offset)

# class RotaryPositionalEmbedding(torch.nn.Module): # TODO: incl fused kernel versions of rotary pos emb
#     def __init__(
#             self, 
#             dim, 
#             base=10000, 
#             learned_freq=False,
#             rotary_interpolation_factor=1.0,
#             precision=torch.bfloat16, 
#         ):
#         """Rotary positional embedding
#         Reference : https://blog.eleuther.ai/rotary-embeddings/
#         Paper: https://arxiv.org/pdf/2104.09864.pdf
#         Adapted from: https://fairseq.readthedocs.io/en/latest/_modules/fairseq/modules/rotary_positional_embedding.html
#         Args:
#             dim: Dimension of embedding
#             base: Base value for exponential
#             precision: precision to use for numerical values
#         """
#         super().__init__()
#         inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
#         self.learned_freq = learned_freq

#         if self.learned_freq:
#             self.inv_freq = torch.nn.Parameter(inv_freq, requires_grad=True)
#         else:
#             self.register_buffer("inv_freq", inv_freq)

#         self.seq_len_cached = None
#         self.cos_cached = None
#         self.sin_cached = None
#         self.precision = precision
#         # register rotary interpolation factor as buffer so it can be saved
#         self.register_buffer("rotary_interpolation_factor", torch.tensor(rotary_interpolation_factor))

#     def reset_if_needed(self):
#         if self.learned_freq: # bcos we cant keep them after backward pass
#             self.cos_cached = None
#             self.sin_cached = None
#             self.seq_len_cached = None
    
#     def forward(self, seq_len, device=torch.device("cpu")):
#         """
#         Args:
#             x: Input x with T X B X C
#             seq_len: Sequence length of input x
#         """
#         if seq_len != self.seq_len_cached:
#             self.seq_len_cached = seq_len
#             t = torch.arange(seq_len, device=device).type_as(self.inv_freq) / self.rotary_interpolation_factor
#             freqs = torch.einsum("i,j->ij", t, self.inv_freq)
#             emb = torch.cat((freqs, freqs), dim=-1).to(device)
#             self.cos_cached = emb.cos()[None, :, None, :]
#             self.sin_cached = emb.sin()[None, :, None, :]
#         return self.cos_cached, self.sin_cached


# rotary pos emb helpers: (this should all just be moved into rotary class tbh)
# def rotate_half(x):
#     x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
#     return torch.cat(
#         (-x2, x1), dim=x1.ndim - 1
#     )  # dim=-1 triggers a bug in earlier torch versions


# def apply_rotary_pos_emb(q, k, cos, sin, q_offset: int = 0):
#     q_cos, q_sin = (
#         cos[:, q_offset : q.shape[1] + q_offset],
#         sin[:, q_offset : q.shape[1] + q_offset],
#     )
#     return (q * q_cos) + (rotate_half(q) * q_sin), (k * cos) + (rotate_half(k) * sin)

# class apply_rotary(): 
#     def __init__(self, cos, sin, q_offset: int = 0, learned: bool = False):
#         self.learned = learned
#         self.cos = cos
#         self.sin = sin
#         self.q_offset = q_offset
    
#     def apply(self, q, k):
#         return apply_rotary_pos_emb(q, k, self.cos, self.sin, self.q_offset)