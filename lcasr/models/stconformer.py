import torch, torch.nn as nn, torch.nn.functional as F
import apex
from torch.utils.checkpoint import checkpoint # # gradient/activation checkpointing
from einops import rearrange
from functools import partial
from lcasr import components
from lcasr.components import fused_dense, subsampling, convolution, decoder, wrappers, feedforward
from lcasr.components.rotary_emb import RotaryPositionalEmbedding, apply_rotary
from lcasr.utils.helpers import exists

ConformerConvolution = convolution.ConformerConvolution
ConformerFeedForward = fused_dense.FusedMLP
ConvSubsampling, StackingSubsampling = subsampling.ConvSubsampling, subsampling.StackingSubsampling
PreNorm, Scale = wrappers.PreNorm, wrappers.Scale
from einops.layers.torch import Rearrange
from torch.func import vmap, grad, functional_call


def rms_norm(x: torch.Tensor, weight=None, eps: float=1e-05) -> torch.Tensor:
    output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        return output * weight
    return output

class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps: float=1e-05, weight: bool=True, dtype=None, device=None):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape, dtype=dtype, device=device))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x.float(), self.weight, self.eps).to(dtype=x.dtype)

DEFAULT_NORM = RMSNorm

class STConformer(nn.Module): 
    def __init__(
        self,
        vocab_size = 128,
        feat_in = 80,
        subsampling = 'dw_striding',
        subsampling_factor = 8,
        subsampling_conv_channels = 256,
        subsampling_act = 'silu',
        subsampling_norm_out = False,
        n_layers = 6,
        teacher_layers = 2,
        d_model = 768,
        n_heads = 6,
        head_dim = 128,
        expansion_factor = 4,
        dropout_ff = 0.0,
        dropout_conv = 0.0,
        dropout_attn = 0.0,
        conv_kernel_size = 9,
        conv_expansion_factor = 1,
        decoder_norm = False,
        rotary_interpolation_factor = 1.0, # https://arxiv.org/abs//2306.15595 Extending Context Window of Large Language Models via Positional Interpolation
        self_conditioning = True,
        default_norm = 'layer_norm',
        sandwich_norm = False,
        bias_in_ff = False,
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
        self.conv_expansion_factor = conv_expansion_factor
        self.rotary_interpolation_factor = rotary_interpolation_factor
        self.self_conditioning = self_conditioning
        self.sandwich_norm = sandwich_norm
        self.bias_in_ff = bias_in_ff

        accepted_norms = ['rms_norm', 'layer_norm']
        assert default_norm in accepted_norms, f'default_norm must be one of {accepted_norms} (got {default_norm})'
        default_norm = RMSNorm if default_norm == 'rms_norm' else nn.LayerNorm

        subsampling_act = components.helpers.get_act(subsampling_act)

        self.dropout_ff = dropout_ff
        self.dropout_conv = dropout_conv
        self.dropout_attn = dropout_attn

        self.subsampling_mode = subsampling
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels if subsampling_conv_channels != -1 else d_model

        self.decoder_norm = decoder_norm


        self.rotary_pos_emb = RotaryPositionalEmbedding(
            dim = head_dim,
            base = kwargs.get('rotary_base_freq', 10000),
            learned_freq = False,
            rotary_interpolation_factor = rotary_interpolation_factor
        )

        self.decoder = decoder.ASRLinearSCDecoder(
            d_model = d_model,
            vocab_size = vocab_size,
            norm = decoder_norm,
            norm_fn = default_norm,
        )

        self.st_predictor = PreNorm(d_model, nn.Linear(d_model, d_model), norm = default_norm)
        
        subsampling_args = {'subsampling_factor': self.subsampling_factor, 'feat_in': self.feat_in, 'feat_out': self.d_model, 'norm_out': subsampling_norm_out,}
        self.subsampling = ConvSubsampling(subsampling = self.subsampling_mode, conv_channels = self.subsampling_conv_channels, activation = subsampling_act, **subsampling_args)

        
        self.layers = nn.ModuleList()
        self.teacher_layers = nn.ModuleList()

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
                default_norm = default_norm,
                sandwich_norm = sandwich_norm,
                bias_in_ff = bias_in_ff,
                transformer = False,
                conv_expansion_factor = conv_expansion_factor,
                **kwargs
            )
            self.layers.append(l)

        for i in range(teacher_layers):
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
                default_norm = default_norm,
                sandwich_norm = sandwich_norm,
                bias_in_ff = bias_in_ff,
                transformer = False,
                conv_expansion_factor = conv_expansion_factor,
                **kwargs
            )
            self.teacher_layers.append(l)

    @staticmethod
    def create_custom_forward(module): # for activation checkpointing allow passing dictionary as the argument to the module
        def custom_forward(*args, **kwargs):
            return module(*args, **kwargs)
        return custom_forward


    def forward(
            self, 
            audio_signal, 
            attn_mask = None,
            pad_mask = None,
            rotary_emb_fn=None,
            mode = 'pre-update',
            return_logits = False,
        ):
        decoder = self.decoder
        assert mode in ['pre-update', 'post-update'], f'mode must be one of [pre-update, post-update] (got {mode})'

        audio_signal = audio_signal.unsqueeze(0)
        pad_mask = pad_mask.unsqueeze(0) if pad_mask is not None else None
        
        for lth, layer in enumerate(self.layers):
            audio_signal = layer(
                x = audio_signal, 
                attn_mask = attn_mask, 
                pad_mask = pad_mask,
                rotary_emb_fn = rotary_emb_fn
            )
        
            if self.self_conditioning:
                iterim_post = torch.nn.functional.softmax(decoder(x=audio_signal, logits=True), dim=-1)
                audio_signal = decoder.integrate_projections(audio_signal, decoder.project_back(iterim_post))        

        student_signal = self.st_predictor(audio_signal)

   
        for lth, layer in enumerate(self.teacher_layers):
            audio_signal = layer(
                x = audio_signal, 
                attn_mask = attn_mask, 
                pad_mask = pad_mask,
                rotary_emb_fn = rotary_emb_fn
            )
        
            if self.self_conditioning and ((lth != len(self.teacher_layers) - 1) or (mode == 'pre-update')):
                iterim_post = torch.nn.functional.softmax(decoder(x=audio_signal, logits=True), dim=-1)
                audio_signal = decoder.integrate_projections(audio_signal, decoder.project_back(iterim_post))    

        teacher_signal = self.st_predictor(audio_signal)

        if mode == 'pre-update':
            sim = torch.nn.functional.cosine_similarity(student_signal, teacher_signal, dim=-1)
            sim = (2 - 2 * sim)
            n = sim.shape[-1]
            if pad_mask is not None:
                sim = sim.masked_fill(pad_mask, 0)
                n = (~pad_mask).sum()
            sim = sim.sum() / n
            return sim, audio_signal.squeeze(0)

        elif mode == 'post-update':
            audio_signal = decoder.norm(audio_signal)
            final_posts = decoder(x = audio_signal, logits = return_logits) # having decoder.norm should have been removed is sortof a bug but probably doesn't matter

            return {
                'final_posteriors': final_posts.squeeze(0),
            }
            
        else:
            raise NotImplementedError

    def print_total_params(self, only_trainable = False):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad) if only_trainable else sum(p.numel() for p in self.parameters())
        pstr = 'Total trainable params: ' if only_trainable else 'Total params: '
        print(f'{pstr}: ', total/1e6, 'M')
        return total



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
        default_norm = DEFAULT_NORM,
        sandwich_norm = False,
        bias_in_ff = True,
        conv_expansion_factor = 1,
        **kwargs
    ):
        super().__init__()

        self.d_model = d_model
        self.conv_kernel_size = conv_kernel_size
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.sandwich_norm = sandwich_norm
        self.bias_in_ff = bias_in_ff
        
        
        self.conv = PreNorm(
            d_model = d_model, 
            fn = ConformerConvolution(
                d_model = d_model,
                kernel_size = conv_kernel_size,
                norm_type = kwargs.get('conv_norm', 'group_norm'),
                exp_factor = conv_expansion_factor,
            ),
            norm = default_norm
        )
        self.do_conv = nn.Dropout(dropout_conv)

        self.ff1 = Scale(0.5, PreNorm(d_model = d_model, 
            fn = feedforward.swiglu(dim = d_model, exp_f=2, bias=bias_in_ff),
            norm = default_norm, sandwich_norm = sandwich_norm))

        self.ff2 = Scale(0.5, PreNorm(d_model = d_model, 
            fn = feedforward.swiglu(dim = d_model, exp_f=2, bias=bias_in_ff),
            norm = default_norm, sandwich_norm = sandwich_norm))

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
                **kwargs
            ),
            norm = default_norm,
            sandwich_norm = sandwich_norm
        )

        self.do_attn_out = nn.Dropout(min(dropout_ff, 0.1)) # don't wan't this too large
        self.norm_out = default_norm(d_model)

            

    def forward(self, x, attn_mask, pad_mask, rotary_emb_fn = None):
        '''
        pad_mask: mask for padding used in conv layers
        attn_mask: attn_mask this should include the cached keys and values
        length: list of lengths of the input sequence
        cached_kv: kvs from previous block-reccurrent time step
        '''
        x = self.do_ff(self.ff1(x)) + x

        x = self.do_attn_out(self.attend(
            x = x,
            attn_mask = attn_mask,
            rotary_emb_fn = rotary_emb_fn
        )) + x
        
        x = self.do_conv(self.conv(x, pad_mask = pad_mask)) + x
    
        x = self.do_ff(self.ff2(x)) + x

        x = self.norm_out(x)

        return x


class Attention(nn.Module):
    def __init__(self, n_feats, head_dim, n_heads, bias=False, dropout=0.0, **kwargs):
        super().__init__()
        super().__init__()
        self.layer_idx = kwargs.get('layer_idx', None)

        self.n_feats, self.head_dim, self.n_heads = n_feats, head_dim, n_heads
   
        self.activation = nn.Softmax(dim=-1)

        self.dropout_p = dropout

        self.qkv_proj = nn.Linear(n_feats, 3 * n_heads * head_dim, bias=bias)

        self.qkv = lambda x: rearrange(self.qkv_proj(x), "b n (h d qkv) -> qkv b n h d", qkv=3, h=n_heads, d=head_dim)

        self.out_proj = nn.Linear(n_heads * head_dim, n_feats, bias=bias)

    def sdpa(self, q, k, v, mask): # use to get the attention weights for visualization/debugging
        a_weight = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        a_weight = (a_weight / self.head_dim ** 0.5)
        if mask is not None:
            a_weight = a_weight.masked_fill(mask, -torch.finfo(a_weight.dtype).max)
        a_weight = a_weight.softmax(dim=-1)
        return torch.einsum("b h i j, b h j d -> b h i d", a_weight, v), a_weight
        
    def forward(self, x, attn_mask=None, rotary_emb_fn = None):

        q, k, v = self.qkv(x)
        q, k = rotary_emb_fn.apply(q, k) if rotary_emb_fn is not None else (q, k)

        q, k, v = q.transpose(1, 2).contiguous().squeeze(0), k.transpose(1, 2).contiguous().squeeze(0), v.transpose(1, 2).contiguous().squeeze(0)
        #q,k,v = q.squeeze(0), k.squeeze(0), v.squeeze(0)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=False)
        out = rearrange(out[None], "b h n d -> b n (h d)")
        out = self.out_proj(out)
      
        return out

# rotary pos emb helpers: (this should all just be moved into rotary class tbh)
def rotate_half(x, rdim):
    x1, x2 = x[..., : rdim], x[..., rdim :]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions

def apply_rotary_pos_emb(q, k, cos, sin, rdim):
    return (q * cos) + (rotate_half(q, rdim=rdim) * sin), (k * cos) + (rotate_half(k, rdim=rdim) * sin)

class apply_rotary(): 
    def __init__(self, cos, sin, dim):
        self.cos = cos
        self.sin = sin
        self.rdim = dim // 2
    
    def apply(self, q, k):
        return apply_rotary_pos_emb(q, k, self.cos, self.sin, self.rdim)


def test_fn():
    lengths = torch.LongTensor([1000,1024])
    return stconformer_fwd(STConformer(), torch.randn(2, 80, 1024), lengths)


def stconformer_fwd(instance:STConformer, x, lengths=None):
    if lengths is None:
        lengths = torch.LongTensor([x.shape[-1]]*x.shape[0]).to(x.device)

    audio_signal = x
    audio_signal = torch.transpose(audio_signal, 1, 2)
    audio_signal, lengths = instance.subsampling(audio_signal, lengths = lengths) 

    cos, sin = instance.rotary_pos_emb(audio_signal.size(1))
    cos, sin = cos.to(audio_signal.device), sin.to(audio_signal.device)
    rotary_emb_fn = apply_rotary(cos, sin, dim = instance.head_dim)

    args = (audio_signal,) 
    if lengths.max() != lengths.min(): # create pad and attn mask
        pad_mask = torch.arange(lengths.max()).to(lengths.device).expand(len(lengths), lengths.max()) >= lengths.unsqueeze(1)
        attn_mask = ~(rearrange(pad_mask, 'b n -> b () n ()') * rearrange(pad_mask, 'b n -> b () () n'))
        args = (audio_signal, attn_mask, pad_mask)
    
    def innerloop(x, attn_mask=None, pad_mask=None):
        network_params = dict(instance.named_parameters())
        student_params = {k:v for k,v in network_params.items() if k.startswith('layers')}
        other_params = {k:v for k,v in network_params.items() if not k.startswith('layers')}

        compute_loss = lambda params, x, attn_mask, pad_mask: functional_call(instance, {**other_params, **params}, args=(x, attn_mask, pad_mask), kwargs={'rotary_emb_fn':rotary_emb_fn, 'mode':'pre-update'})
        w_update, x = grad(compute_loss, has_aux=True)(student_params, x, attn_mask, pad_mask)
   
        fwd_out = functional_call(
            instance, 
            {k: v - w_update[k] if k.startswith('layers') else v for k, v in network_params.items()}, 
            args=(x, attn_mask, pad_mask), 
            kwargs={'rotary_emb_fn':rotary_emb_fn, 'mode':'post-update'}
        )
        return fwd_out

    x = vmap(lambda a,b=None,c=None: innerloop(a, b, c), chunk_size=None, in_dims=0)(*args)

    return {
        **x,
        'length': lengths       
    }
    
