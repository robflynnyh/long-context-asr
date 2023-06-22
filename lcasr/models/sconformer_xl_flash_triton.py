import torch, torch.nn as nn, torch.nn.functional as F

import apex
from torch.utils.checkpoint import checkpoint # # gradient/activation checkpointing
from einops import rearrange, repeat
from torch import einsum

from lcasr.components import fused_dense, subsampling, convolution
from torch.cuda.amp import autocast
from contextlib import nullcontext

ConformerConvolution = convolution.ConformerConvolution
ConformerFeedForward = fused_dense.FusedMLP
#ConformerFeedForward = lambda x: nn.Linear(x,x)
ConvSubsampling = subsampling.ConvSubsampling
DEFAULT_NORM = apex.normalization.FusedRMSNorm

from flash_attn.flash_attention import FlashAttention
from flash_attn.flash_attn_triton import flash_attn_func
'''
chunk 127/129
torch.Size([2, 64, 2048]) torch.Size([2, 22])
ten sor([2048,  438])
nan
'''

class SCConformerXL(nn.Module):
    def __init__(
        self,
        vocab_size = 128,
        feat_in = 64,
        n_layers = 12,
        d_model = 256,
        n_heads = 8,
        head_dim = 32,
        expansion_factor = 4,
        dropout_ff = 0.0,
        dropout_conv = 0.0,
        dropout_attn = 0.0,
        checkpoint_every_n_layers = 1,
        subsampling_factor = 4,
        conv_kernel_size = 31,
        **kwargs
    ):
        super().__init__()
        
        self.feat_in = feat_in
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.expansion_factor = expansion_factor
        self.conv_kernel_size = conv_kernel_size

        self.checkpoint_every_n_layers = checkpoint_every_n_layers
        
        self.subsampling_factor = subsampling_factor

        self.overlap_interp_factor = nn.Parameter(torch.tensor(0.5))

        self.dropout_ff = dropout_ff
        self.dropout_conv = dropout_conv
        self.dropout_attn = dropout_attn

        self.decoder = ASRLinearSCDecoder(
            d_model = d_model,
            vocab_size = vocab_size
        )

        self.subsampling = ConvSubsampling(
            subsampling = 'striding',
            subsampling_factor = subsampling_factor,
            feat_in = feat_in,
            feat_out = d_model,
            conv_channels = d_model,
            activation = nn.SiLU(),
        )

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            l = ConformerLayer(
                d_model = d_model,
                conv_kernel_size = conv_kernel_size,
                expansion_factor = expansion_factor,
                dropout_ff = dropout_ff,
                dropout_conv = dropout_conv,
                dropout_attn = dropout_attn,
                layer_idx = i,
                total_layers = n_layers,
                head_dim = head_dim,
                n_heads = n_heads,
                **kwargs
            )
            self.layers.append(l)

    @staticmethod
    def create_custom_forward(module): # for activation checkpointing allow passing dictionary as the argument to the module
        def custom_forward(*args, **kwargs):
            return module(*args, **kwargs)
        return custom_forward

    def forward(
            self, 
            audio_signal, 
            length = None,
            cached_kvs = None,
            cached_kv_lengths = None
        ):
        '''
        audio_signal: (batch_size, time, feat)
        length: (batch_size,)
        cached_kvs: (kv i.e 2, batch_size, layers, heads, time, head_dim)
        '''
        return self.forward_for_export(audio_signal=audio_signal, decoder=self.decoder, length=length, cached_kvs=cached_kvs, cached_kv_lengths=cached_kv_lengths)



    def forward_for_export(self, audio_signal, decoder, length = None, cached_kvs = None, cached_kv_lengths = None):
        max_audio_length: int = audio_signal.size(-1)

        if length is None:
            length = torch.tensor([max_audio_length] * audio_signal.size(0), device=audio_signal.device)
            
        audio_signal = torch.transpose(audio_signal, 1, 2)
        audio_signal, length = self.subsampling(audio_signal, lengths = length)
        max_audio_length = audio_signal.size(1)
        ## create masks
        
        mask = torch.arange(max_audio_length, device=audio_signal.device).expand(audio_signal.size(0), max_audio_length) >= length.unsqueeze(1)
        full_kv_lengths = length + cached_kv_lengths if cached_kv_lengths is not None else length

        if cached_kv_lengths is None and length.max() == length.min():
            att_mask = torch.zeros(audio_signal.size(0), 1, audio_signal.size(1), full_kv_lengths.max(), device=audio_signal.device)
        else:
            full_kv_mask = torch.arange(full_kv_lengths.max(), device=audio_signal.device).expand(audio_signal.size(0), full_kv_lengths.max()) >= full_kv_lengths.unsqueeze(1)
            qmask, kmask = ~mask, ~full_kv_mask
            att_mask = ~(rearrange(qmask, 'b n -> b () n ()') * rearrange(kmask, 'b n -> b () () n'))
            att_mask = att_mask.to(audio_signal.dtype) * -torch.finfo(audio_signal.dtype).max
            # print(att_mask.to(audio_signal.dtype))
            # exit()

        pad_mask = mask if length.max() != length.min() else None
        

        iterim_posteriors = []

        kvs_to_cache = []
        for lth, layer in enumerate(self.layers):
            current_layer_kvs = cached_kvs[:,:,lth] if cached_kvs is not None else None

            if self.checkpoint_every_n_layers > 0 and lth % self.checkpoint_every_n_layers == 0:
                audio_signal, kv_to_cache = checkpoint(
                    self.create_custom_forward(layer), 
                    audio_signal, # x
                    att_mask, # att_mask
                    pad_mask, # pad_mask
                    current_layer_kvs
                )
    
            else:
                audio_signal, kv_to_cache = layer(
                    x = audio_signal, 
                    attn_mask = att_mask, 
                    pad_mask = pad_mask,
                    cached_kv = current_layer_kvs
                )
                

            kvs_to_cache.append(kv_to_cache) # possibly detach and move to cpu ?    
            
            if lth != len(self.layers) - 1:
                iterim_logits = decoder(x=audio_signal, logits=True)
                iterim_post = torch.nn.functional.softmax(iterim_logits, dim=-1)
                iterim_logposteriors = torch.log(iterim_post)
                iterim_posteriors.append(iterim_logposteriors)
                audio_signal = decoder.integrate_projections(audio_signal, decoder.project_back(iterim_post))        

        # stack the posteriors along the first dimension (height, batch, seq_len, dim)
        #print(len(iterim_posteriors),111111111111111111)
        iterim_posteriors = torch.stack(iterim_posteriors, dim=0) if len(iterim_posteriors) > 0 else None
        kvs_to_cache = torch.stack(kvs_to_cache, dim=0)
        kvs_to_cache = rearrange(kvs_to_cache, 'l kv b h n d -> kv b l h n d')
        
        final_posts = decoder(x=audio_signal, logits=False)


        return {
            'final_posteriors': final_posts,
            'iterim_posteriors': iterim_posteriors,
            'kvs_to_cache': kvs_to_cache,
            'length': length,
            'full_kv_lengths': full_kv_lengths,
        }

    def print_total_params(self, only_trainable = False):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad) if only_trainable else sum(p.numel() for p in self.parameters())
        pstr = 'Total trainable params: ' if only_trainable else 'Total params: '
        print(f'{pstr}: ', total/1e6, 'M')
        return total

class PreNorm(nn.Module): # applies normalization before fn
    def __init__(self, d_model, fn, norm = DEFAULT_NORM, disable_autocast = False):
        super().__init__()
        self.norm = norm(d_model)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return x

class Scale(nn.Module): # scales output of fn by scale
    def __init__(self, scale, fn):
        super().__init__()
        self.scale = scale
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class ConformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        conv_kernel_size,
        dropout_ff,
        dropout_conv,
        dropout_attn,
        layer_idx,
        total_layers,
        head_dim,
        n_heads,
        **kwargs
    ):
        super().__init__()

        self.d_model = d_model
        self.conv_kernel_size = conv_kernel_size
        self.layer_idx = layer_idx
        self.total_layers = total_layers

        self.conv = PreNorm(
            d_model = d_model, 
            fn = ConformerConvolution(
                d_model = d_model,
                kernel_size = conv_kernel_size,
                norm_type = 'batch_renorm'
            ),
        )

        self.do_conv = nn.Dropout(dropout_conv)

        self.ff1 = Scale(0.5, PreNorm(d_model = d_model, fn = ConformerFeedForward(d_model)))
        self.ff2 = Scale(0.5, PreNorm(d_model = d_model, fn = ConformerFeedForward(d_model)))
        self.do_ff = nn.Dropout(dropout_ff)

        self.attend = PreNorm(
            d_model = d_model, 
            fn = Attention(
                n_feats = d_model,
                head_dim = head_dim,
                n_heads = n_heads,
                dropout = dropout_attn,
                bias = False,
                layer_idx = layer_idx,
            )
        )

        self.do_attn_out = nn.Dropout(min(dropout_ff, 0.1)) # don't wan't this too large
        self.norm_out = DEFAULT_NORM(d_model)

            

    def forward(self, x, attn_mask, pad_mask, cached_kv = None):
        '''
        pad_mask: mask for padding used in conv layers
        attn_mask: attn_mask this should include the cached keys and values
        cached_kv: kvs from previous block-reccurrent time step
        '''

        x = self.do_ff(self.ff1(x)) + x

        attn_out, kv_to_cache = self.attend(
            x = x,
            attn_mask = attn_mask,
            pad_mask = pad_mask,
            cached_kv = cached_kv,
        )
        x = self.do_attn_out(attn_out) + x
        
        x = self.do_conv(self.conv(x, pad_mask = pad_mask)) + x
    
        x = self.do_ff(self.ff2(x)) + x

        x = self.norm_out(x)

        return x, kv_to_cache


def l2norm(t, groups = 1, dim = -1):
    if groups == 1:
        return F.normalize(t, p = 2, dim = dim)
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = dim)
    return rearrange(t, '... g d -> ... (g d)')


class Attention(nn.Module):
    def __init__(
        self,
        n_feats,
        head_dim,
        n_heads,
        bias=False,
        dropout=0.0,
        **kwargs
    ):
        super().__init__()
        self.layer_idx = kwargs.get('layer_idx', None)
        #self.history_vector = torch.nn.Parameter(torch.zeros(2, 1, 1, 1, head_dim), requires_grad=True)

        self.n_feats, self.head_dim, self.n_heads = n_feats, head_dim, n_heads
   
        self.activation = nn.Softmax(dim=-1)

        self.dropout_p = dropout

        self.softmax_scale = head_dim ** -0.5

        self.qkv_proj = nn.Linear(n_feats, 3 * n_heads * head_dim, bias=bias)
        self.qkv = lambda x: rearrange(self.qkv_proj(x), "b n (h d qkv) -> qkv b n h d", qkv=3, h=n_heads, d=head_dim)

        self.out_proj = nn.Linear(n_heads * head_dim, n_feats, bias=bias)


    
    def attatch_cache(self, kv, cached_kv):
        kv = torch.stack(kv, dim=0)
        if cached_kv is None:
            return kv
        new_kv = torch.cat([cached_kv, kv], dim=1)
        return new_kv


    def forward(self, x, attn_mask=None, pad_mask=None, cached_kv=None):
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim
        #print(x.shape, mask.shape)

        if pad_mask != None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0)
            attn_mask = attn_mask.half()

        
        q, k, v = self.qkv(x)

        kv = self.attatch_cache([k, v], cached_kv)
        k, v = kv
       
        #torch.backends.cuda.enable_flash_sdp(enabled = False) # enable flash attention if cuda for faster training

        if x.device.type == 'cuda':
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            if q.dtype == torch.float32:
                q, k, v = q.half(), k.half(), v.half()
            attn_mask = attn_mask.to(q.dtype)
            out = flash_attn_func(q, k, v, attn_mask, self.softmax_scale)
            out = out.to(x.dtype)
            out = rearrange(out, "b n h d -> b n (h d)")
        else:
            out = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=False)

        if pad_mask != None:
            out = out.masked_fill(pad_mask.unsqueeze(-1), 0)

        out = self.out_proj(out)

        return out, kv


class ASRLinearSCDecoder(nn.Module):
    def __init__(self, d_model, vocab_size, norm=True):
        super().__init__()
        # Add 1 for blank char
        self.num_classes = vocab_size + 1
        self.ff = nn.Linear(d_model, self.num_classes)
        self.reprojection = nn.Linear(self.num_classes, d_model)
        self.norm = DEFAULT_NORM(d_model) if norm else nn.Identity()

    def forward(self, x, logits=False):
        x = self.norm(x)
        x = self.ff(x)
        x = F.log_softmax(x, dim=-1) if not logits else x
        return x

    def project_back(self, x):
        return self.reprojection(x)

    def integrate_projections(self, x, proj1):
        return x + proj1


