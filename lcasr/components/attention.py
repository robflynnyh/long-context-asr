import torch, torch.nn as nn
from einops import rearrange
from flash_attn.modules.mha import FlashSelfAttention
from flash_attn.bert_padding import unpad_input, pad_input

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

        # softmax_scale is set to None but will default to 1/sqrt(d_k) in FlashAttention
        self.flash_attn_fn = FlashSelfAttention(
            softmax_scale = None, 
            attention_dropout = dropout, 
            causal = False, 
            window_size=(-1, -1), 
            alibi_slopes=None
        )
     

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

        
    def forward(self, x, attn_mask=None, length=None, pad_mask=None, flash_attn = True, rotary_emb_fn = None):
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim

        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0)

        q, k, v = self.qkv(x)
        kv = torch.stack([k, v], dim=2)
        

        if rotary_emb_fn is not None:
            if rotary_emb_fn.learned == False:
                q, kv[:, :, 0] = rotary_emb_fn.apply(q, kv[:, :, 0])
            else:
                k, v = kv[:, :, 0], kv[:, :, 1]
                q, k = rotary_emb_fn.apply(q, k)
                kv = torch.stack([k, v], dim=2)

     
        ### Flash attention stuff 
        if x.device.type == 'cuda' and flash_attn:
            q, kv = q.contiguous(), kv.contiguous()
            if q.dtype == torch.float32:
                q, kv = q.half(), kv.half()

            qkv = torch.cat([q[:,:,None], kv], dim=2)
    

            if length is not None and length.max() != length.min(): # variable length
                qkv_unpad, qkv_indices, cu_seqlens, max_seqlen = unpad_input(qkv, attn_mask)
                out = self.flash_attn_fn(qkv=qkv_unpad, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
                out = pad_input(out, indices=qkv_indices, batch=B, seqlen=max_seqlen)
            else:
                out = self.flash_attn_fn(qkv)

            out = out.to(x.dtype) 
            out = rearrange(out, "b n h d -> b n (h d)")
        else:
            k, v = rearrange(kv, "b n kv h d -> kv b h n d", kv=2).contiguous()
            q = q.transpose(1, 2).contiguous()
            out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=False)
            out = rearrange(out, "b h n d -> b n (h d)")
      
        if pad_mask != None:
            out = out.masked_fill(pad_mask.unsqueeze(-1), 0)

        out = self.out_proj(out)
        
        return out