import torch, torch.nn as nn
from einops import rearrange, repeat
from typing import List
from torch.nn import functional as F
import math
from os.path import join

try:
    from flash_attn import (flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func)
    from flash_attn.bert_padding import unpad_input, pad_input
    from flash_attn.flash_attn_interface import _get_block_size_n
except ImportError:
    flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func = None, None
    flash_attn_qkvpacked_func, flash_attn_kvpacked_func = None, None
    flash_attn_with_kvcache = None
    unpad_input, pad_input = None, None
    _get_block_size_n = None


## misc flash attention stuff from: https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py
def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )

def convert_flash_attn_S_to_softmax( # https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py
    S,
    seqlen_q,
    seqlen_k,
    query_padding_mask,
    key_padding_mask,
    head_dim,
    is_dropout,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """FlashAttention stores the S matrix in a different way.
    Arguments:
        S: (batch_size, nheads, seqlen_q_rounded, seqlen_k_rounded)
        query_padding_mask: (batch_size, seqlen_q_rounded)
        key_padding_mask: (batch_size, seqlen_k_rounded)
    """
    if causal:
        window_size = (window_size[0], 0)
    seqlen_q_rounded, seqlen_k_rounded = S.shape[-2:]
    S_converted = S
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            S.device,
        )
        local_mask = F.pad(
            local_mask,
            (0, seqlen_k_rounded - seqlen_k, 0, seqlen_q_rounded - seqlen_q),
            value=True,
        )
        S_converted = S_converted.masked_fill(local_mask, 0.0)

    # Need to zero out things not in attention_mask in case S was initialized with random values
    # and some of those values aren't overwritten.
    seqlen_q_og = (
        query_padding_mask.shape[-1] if query_padding_mask is not None else seqlen_q_rounded
    )
    if query_padding_mask is not None:
        query_padding_mask = F.pad(query_padding_mask, (0, seqlen_q_rounded - seqlen_q_og))
        S_converted = S_converted.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    seqlen_k_og = key_padding_mask.shape[-1] if key_padding_mask is not None else seqlen_k
    if key_padding_mask is not None:
        key_padding_mask = F.pad(key_padding_mask, (0, seqlen_k_rounded - seqlen_k_og))
        S_converted = S_converted.masked_fill(rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0)
    S_converted = F.pad(S_converted, (0, 0, 0, seqlen_q_og - seqlen_q_rounded))
    S_converted = F.pad(S_converted, (0, seqlen_k_og - seqlen_k_rounded))
    return S_converted[:, :, :seqlen_q, :seqlen_k]

def normalize_flash_attn_S(
    attn_unnorm,
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    is_dropout=False,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k, v: (batch_size, seqlen_k, nheads, head_dim)
        key_padding_mask: (batch_size, seqlen_q)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
    Output:
        softmax_lse: (batch_size, nheads, seqlen_q)
        softmax_max: (batch_size, nheads, seqlen_q)
    """
    if causal:
        window_size = (window_size[0], 0)
    q, k, v = q.float(), k.float(), v.float()
    _, seqlen_q, _, head_dim = q.shape
    seqlen_k = k.shape[1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(head_dim), k)
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias.to(dtype=scores.dtype)
    block_size_n = _get_block_size_n(scores.device, head_dim, is_dropout, causal)
    scores_block = scores.split(block_size_n, dim=-1)
    lse_block = torch.stack([torch.logsumexp(s, dim=-1) for s in scores_block], dim=-1)
    lse = torch.logsumexp(lse_block, dim=-1)
    # lse could be -inf (i.e. all values in scores are -inf), and we want to set those to inf
    # so that when we do torch.exp(m - lse), we get 0.0 instead of NaN.
    lse[lse == float("-inf")] = float("inf")
    scores_max_block = torch.stack([torch.amax(s, dim=-1) for s in scores_block], dim=-1)
    cummax_block = torch.cummax(scores_max_block.flip(-1), dim=-1).values.flip(-1).unbind(dim=-1)
    attn_unnorm_block = attn_unnorm.split(block_size_n, dim=-1)
    attn_norm = torch.cat(
        [
            a * rearrange(torch.exp(m - lse), "b h s -> b h s 1")
            for a, m in zip(attn_unnorm_block, cummax_block)
        ],
        dim=-1,
    )
    if query_padding_mask is not None:
        attn_norm.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    return attn_norm.to(dtype=attn_unnorm.dtype)
## misc flash attention stuff

class FlashSelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(
        self,
        causal=False,
        softmax_scale=None,
        attention_dropout=0.0,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attention=False,
    ):
        super().__init__()
        assert flash_attn_varlen_qkvpacked_func is not None, "FlashAttention is not installed"
        assert flash_attn_qkvpacked_func is not None, "FlashAttention is not installed"
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)
        self.window_size = window_size
        self.deterministic = deterministic
        self.return_attention = return_attention


    def forward(self, qkv, *args, causal=None, cu_seqlens=None, max_seqlen=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value.
                If cu_seqlens is None and max_seqlen is None, then qkv has shape (B, S, 3, H, D).
                If cu_seqlens is not None and max_seqlen is not None, then qkv has shape
                (total, 3, H, D), where total is the sum of the sequence lengths in the batch.
            if you want to get attention probs (via: CollectFlashAttentionProbs) and have variable length sequences, you need to pass the following in args:
                - Key_padding_mask: (B, S) tensor, where 1 indicates that the key is a padding token in args if variable length sequence and 
            causal: if passed, will override self.causal
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into qkv.
            max_seqlen: int. Maximum sequence length in the batch.
            
        Returns:
        --------
            out: (total, H, D) if cu_seqlens is not None and max_seqlen is not None,
                else (B, S, H, D).
        """
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        causal = self.causal if causal is None else causal
        unpadded = cu_seqlens is not None
        if self.alibi_slopes is not None:
            self.alibi_slopes = self.alibi_slopes.to(torch.float32)

        dropout = self.drop.p if self.training else 0.0
        if dropout == 0.0 and self.return_attention == True:
            dropout = 1e-7 # workaround to get the attention probs from flash_attn when dropout is 0.0
        
        if unpadded:
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            return flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_seqlen,
                dropout,
                softmax_scale=self.softmax_scale,
                causal=causal,
                alibi_slopes=self.alibi_slopes,
                window_size=self.window_size,
                deterministic=self.deterministic,
                return_attn_probs=self.return_attention,
            )
        else:
            return flash_attn_qkvpacked_func(
                qkv,
                dropout,
                softmax_scale=self.softmax_scale,
                causal=causal,
                alibi_slopes=self.alibi_slopes,
                window_size=self.window_size,
                deterministic=self.deterministic,
                return_attn_probs=self.return_attention,
            )
        
class CollectFlashAttentionProbs: # old
    '''
    Pass a list of Flashattention modules as specified above and this will collect the attention probabilities
    '''
    def __init__(self, attn_modules: List[FlashSelfAttention]):
        self.attn_modules = attn_modules
        self.attn_probs = []

        for module in self.attn_modules:
            module.return_attention = True
            module.register_forward_hook(self.attn_hook)

    def collect(self):
        return self.attn_probs

    def clear(self):
        self.attn_probs = []

    def __call__(self):
        probs = self.collect()
        if len(probs) == 0: print("No attention probabilities found! Make sure to call the model on some input first!")
        self.clear()
        return probs

    def convert_S_matrix(self, S_dmask, qkv, causal, window_size, padding_mask=None):
        # see flash_attn tests funcs
        seqlen, head_dim = qkv.shape[1], qkv.shape[-1]
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask,
            seqlen,
            seqlen,
            None,
            None,
            head_dim,
            False,
            causal,
            window_size
        )
        attn_unnorm = S_dmask_converted.abs()
        attn = normalize_flash_attn_S(
            attn_unnorm,
            qkv[:,:,0],
            qkv[:,:,1],
            qkv[:,:,2],
            padding_mask,
            padding_mask,
            None, # attn_bias (not using alibi currently so not needed)
            False, # is_dropout 
            causal,
            window_size
        )
        return attn

    def attn_hook(self, module, input, output):
        qkv, padding_mask = input[0], None
        if len(input) > 1: padding_mask = input[1]
        out, lse, S_dmask = output
        causal, window_size = module.causal, module.window_size
        self.attn_probs.append(self.convert_S_matrix(S_dmask, qkv, causal, window_size, padding_mask=padding_mask).detach().cpu())
        return out


def get_window_size(kwargs, direction=None):
    if direction is None:
        return kwargs.get('attention_window_size', -1)
    else:
        if kwargs.get(f'attention_window_size_{direction}', None) is not None:
            return kwargs.get(f'attention_window_size_{direction}')
        else:
            return kwargs.get('attention_window_size', -1)
        
def attention_ref( # https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)

class CPUAttention(nn.Module):
    def __init__(self, window_size, upcast):
        super().__init__()
        self.window_size = window_size
        self.upcast = upcast

    def forward(self, q, k, v, query_padding_mask=None, key_padding_mask=None):
        return attention_ref(
            q, k, v, query_padding_mask, key_padding_mask, None, 0.0, None, False, self.window_size, self.upcast, False
        )


class ReturnAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask, causal=False):
        assert causal == False, "Causal attention not implemented when returning attention (o:)"
        
        a_weight = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        a_weight = (a_weight / q.shape[-1] ** 0.5)
        if mask is not None:
            a_weight = a_weight.masked_fill(mask, -torch.finfo(a_weight.dtype).max)

        #a_weight = a_weight.to('cpu')
        # top_values, _ = a_weight.topk(48, dim = -1)
        # sparse_topk_mask = a_weight < top_values[..., -1:]
        # mask = sparse_topk_mask
        # a_weight = a_weight.masked_fill(mask, -torch.finfo(a_weight.dtype).max)
        #a_weight = a_weight.to(q.device)

        a_probs = a_weight.softmax(dim=-1)
        return torch.einsum("b h i j, b h j d -> b h i d", a_probs, v), a_weight

    

class Attention(nn.Module):
    def __init__(
        self,
        n_feats,
        head_dim,
        n_heads,
        dropout=0.0,
        **kwargs
    ):
        super().__init__()
        self.layer_idx = kwargs.get('layer_idx', None)

        self.n_feats, self.head_dim, self.n_heads = n_feats, head_dim, n_heads
   
        self.activation = nn.Softmax(dim=-1)
        self.dropout_p = dropout
        #print(get_window_size(kwargs, direction=None))
        # softmax_scale is set to None but will default to 1/sqrt(d_k) in FlashAttention
        self.left_window, self.right_window = get_window_size(kwargs, direction='left'), get_window_size(kwargs, direction='right')
        
        try:
            self.flash_attn_fn = FlashSelfAttention(
                softmax_scale = None, 
                attention_dropout = dropout, 
                causal = kwargs.get('causal', False),
                window_size=(self.left_window, self.right_window),
                alibi_slopes=None
            )
        except: self.flash_attn_fn = None

        self.causal = kwargs.get('causal', False)
     
        self.return_attention_weights = kwargs.get('return_attention_weights', False)
        self.return_attention_module = ReturnAttention()

        self.qkv_proj = nn.Linear(n_feats, 3 * n_heads * head_dim, bias=kwargs.get('qkv_bias', False))

        self.qkv = lambda x: rearrange(self.qkv_proj(x), "b n (h d qkv) -> qkv b n h d", qkv=3, h=n_heads, d=head_dim)

        self.out_proj = nn.Linear(n_heads * head_dim, n_feats, bias=kwargs.get('bias', False))

    
    def sdpa(self, q, k, v, mask): # use to get the attention weights for visualization/debugging
        a_weight = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        a_weight = (a_weight / self.head_dim ** 0.5)
        if mask is not None:
            a_weight = a_weight.masked_fill(mask, -torch.finfo(a_weight.dtype).max)
        a_weight = a_weight.softmax(dim=-1)
        return torch.einsum("b h i j, b h j d -> b h i d", a_weight, v), a_weight

    @staticmethod
    def apply_rotary(q, kv, rotary_emb_fn):
        if rotary_emb_fn is not None:
            if rotary_emb_fn.learned == False:
                q, kv[:, :, 0] = rotary_emb_fn.apply(q, kv[:, :, 0])
            else:
                k, v = kv[:, :, 0], kv[:, :, 1]
                q, k = rotary_emb_fn.apply(q, k)
                kv = torch.stack([k, v], dim=2)
        return q, kv
        
    def forward(self, x, attn_mask=None, length=None, pad_mask=None, flash_attn = True, rotary_emb_fn = None):
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim
        if pad_mask is not None: x = x.masked_fill(pad_mask.unsqueeze(-1), 0)

        q, k, v = self.qkv(x)
        kv = torch.stack([k, v], dim=2)
        
        q, kv = self.apply_rotary(q, kv, rotary_emb_fn)
     
        ### Flash attention stuff 
        if x.device.type == 'cuda' and flash_attn and not self.return_attention_weights and self.flash_attn_fn is not None:
            q, kv = q.contiguous(), kv.contiguous()
            if q.dtype == torch.float32:
                q, kv = q.half(), kv.half()

            qkv = torch.cat([q[:,:,None], kv], dim=2)
    

            if length is not None and length.max() != length.min(): # variable length
                qkv_unpad, qkv_indices, cu_seqlens, max_seqlen = unpad_input(qkv, attn_mask)
                out = self.flash_attn_fn(qkv_unpad, attn_mask, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
                out = pad_input(out, indices=qkv_indices, batch=B, seqlen=max_seqlen)
            else:
                out = self.flash_attn_fn(qkv)

            out = out.to(x.dtype) 
            out = rearrange(out, "b n h d -> b n (h d)")
        else:
            assert self.left_window == -1 and self.right_window == -1, "windowed attention not supported in CPU mode (yet)"
            k, v = rearrange(kv, "b n kv h d -> kv b h n d", kv=2).contiguous()
            q = q.transpose(1, 2).contiguous()
            if not self.return_attention_weights:
                out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=self.causal)
            else:
                out, _ = self.return_attention_module(q, k, v, attn_mask, causal=self.causal)
            out = rearrange(out, "b h n d -> b n (h d)")
      
        if pad_mask != None:
            out = out.masked_fill(pad_mask.unsqueeze(-1), 0)

        out = self.out_proj(out)
        
        return out
    



class CollectAttentionProbs: 
    '''
    Pass a list of Attention modules as specified above and this will collect the attention probabilities
    '''
    def __init__(self, attn_modules: List[Attention], discard=False, save_path=None, save_prefix=None):
        self.attn_modules = attn_modules
        self.attn_probs = []

        self.save_path = save_path
        self.save_prefix = save_prefix
        self.discard = discard

        for idx, module in enumerate(self.attn_modules):
            module.return_attention_weights = True
            module.return_attention_module.register_forward_hook(self.get_attn_hook(idx))

    def collect(self):
        return self.attn_probs

    def clear(self):
        self.attn_probs = []

    def __call__(self):
        probs = self.collect()
        probs = torch.stack(probs, dim=0)
        if len(probs) == 0: print("No attention probabilities found! Make sure to call the model on some input first!")
        self.clear()
        return probs

    def get_attn_hook(self, layer_idx):
        def attn_hook(module, input, output):
            out, a_weight = output
            a_weight = a_weight.detach().cpu().to(torch.bfloat16)
            if not self.discard: self.attn_probs.append(a_weight)
            if self.save_path is not None:
                name = f'{self.save_prefix}_{layer_idx}.pt' if self.save_prefix is not None else f'layer_{layer_idx}.pt'
                full_path = join(self.save_path, name)
                torch.save(a_weight, full_path)
            return out, a_weight
        return attn_hook