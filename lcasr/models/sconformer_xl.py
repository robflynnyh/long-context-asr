import torch, torch.nn as nn, torch.nn.functional as F

import apex
from torch.utils.checkpoint import checkpoint # # gradient/activation checkpointing
from einops import rearrange, repeat
from torch import einsum

from lcasr.components import fused_dense, subsampling, convolution
from lcasr.utils.helpers import exists

from torch.cuda.amp import autocast
import math
from contextlib import nullcontext

ConformerConvolution = convolution.ConformerConvolution
ConformerFeedForward = fused_dense.FusedMLP

ConvSubsampling = subsampling.ConvSubsampling
StackingSubsampling = subsampling.StackingSubsampling
DEFAULT_NORM = apex.normalization.FusedRMSNorm
RMSNorm = apex.normalization.FusedRMSNorm
LayerNorm = apex.normalization.FusedLayerNorm

#from flash_attn.flash_attention import FlashAttention
from flash_attn.flash_attention import FlashAttention
from flash_attn.modules.mha import FlashCrossAttention

from flash_attn.bert_padding import unpad_input, pad_input

from functools import partial

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
            self, 
            dim, 
            base=10000, 
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
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
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
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(device)
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]
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

class SCConformerXL(nn.Module):
    def __init__(
        self,
        vocab_size = 128,
        feat_in = 80,
        subsampling = 'striding',
        subsampling_factor = 4,
        subsampling_conv_channels = -1,
        subsampling_act = 'silu',
        subsampling_norm_out = False,
        self_condition_subsampling = False,
        n_layers = 12,
        d_model = 256,
        n_heads = 8,
        head_dim = 32,
        expansion_factor = 4,
        dropout_ff = 0.0,
        dropout_conv = 0.0,
        dropout_attn = 0.0,
        checkpoint_every_n_layers = 1,
        conv_kernel_size = 31,
        qk_rms_norm = False,
        shift_kvs = False,
        gated_sc = False,
        decoder_norm = False,
        use_rotary = False,
        encoder_mode = 'conformer',
        rotary_interpolation_factor = 1.0, # https://arxiv.org/abs//2306.15595 Extending Context Window of Large Language Models via Positional Interpolation
        learned_rotary = False,
        self_conditioning = True,
        default_norm = 'rms_norm',
        sandwich_norm = False,
        bias_in_ff = True,
        transformer=False, # disable convolutions
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
        self.encoder_mode = encoder_mode
        self.rotary_interpolation_factor = rotary_interpolation_factor
        self.learned_rotary = learned_rotary
        self.self_conditioning = self_conditioning
        self.sandwich_norm = sandwich_norm
        self.bias_in_ff = bias_in_ff
        self.transformer = transformer
        self.self_condition_subsampling = self_condition_subsampling

        accepted_norms = ['rms_norm', 'layer_norm']
        accepted_subsampling_acts = ['silu', 'relu', 'gelu', 'none']
        assert subsampling_act in accepted_subsampling_acts, f'subsampling_act must be one of {accepted_subsampling_acts} (got {subsampling_act})'
        assert default_norm in accepted_norms, f'default_norm must be one of {accepted_norms} (got {default_norm})'
        default_norm = RMSNorm if default_norm == 'rms_norm' else LayerNorm

        if subsampling_act == 'silu':
            subsampling_act = nn.SiLU()
        elif subsampling_act == 'relu':
            subsampling_act = nn.ReLU()
        elif subsampling_act == 'gelu':
            subsampling_act = nn.GELU()
        elif subsampling_act == 'none':
            subsampling_act = nn.Identity()

        supported_encoders = ['conformer', 'ebranchformer', 'ebranchformer-macaron']
        assert encoder_mode in supported_encoders, f'encoder_mode must be one of {supported_encoders} (got {encoder_mode})'
        if transformer:
            assert encoder_mode == 'conformer', 'transformer mode only supports conformer encoder'

        self.flash_attn = kwargs.get('flash_attn', True)

        self.checkpoint_every_n_layers = checkpoint_every_n_layers

        #self.overlap_interp_factor_logits = nn.Parameter(torch.tensor(0.5))
        #self.overlap_interp_factor_kvs = nn.Parameter(torch.tensor(0.5))

        self.dropout_ff = dropout_ff
        self.dropout_conv = dropout_conv
        self.dropout_attn = dropout_attn
        self.qk_rms_norm = qk_rms_norm

        self.subsampling_mode = subsampling
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels if subsampling_conv_channels != -1 else d_model

        self.shift_kvs = shift_kvs
        self.gated_sc = gated_sc
        self.decoder_norm = decoder_norm

        self.use_rotary = use_rotary

        self.rotary_pos_emb = None
        if self.use_rotary:
            self.rotary_pos_emb = RotaryPositionalEmbedding(
                dim = head_dim,
                base = 10000,
                learned_freq = learned_rotary,
                rotary_interpolation_factor = rotary_interpolation_factor
            )

        self.decoder = ASRLinearSCDecoder(
            d_model = d_model,
            vocab_size = vocab_size,
            norm = decoder_norm,
            gated_sc = gated_sc,
            norm_fn = default_norm,
        )

        self.subsampling = ConvSubsampling(
            subsampling = self.subsampling_mode,
            subsampling_factor = self.subsampling_factor,
            feat_in = feat_in,
            feat_out = d_model,
            conv_channels = self.subsampling_conv_channels,
            activation = subsampling_act,
            norm_out = subsampling_norm_out,
        ) if subsampling != 'stacking' else StackingSubsampling(
            subsampling_factor = self.subsampling_factor,
            feat_in = feat_in,
            feat_out = d_model,
            norm = True if not subsampling_norm_out else False,
            default_norm = default_norm,
            norm_out = subsampling_norm_out,
        )

        
        self.layers = nn.ModuleList()

        additonal_kwargs = {}
        if encoder_mode == 'conformer':
            encoder = ConformerLayer
        elif encoder_mode == 'ebranchformer':
            encoder = EBranchConformerLayer
            additonal_kwargs['has_macaron'] = False
        elif encoder_mode == 'ebranchformer-macaron':
            encoder = EBranchConformerLayer
            additonal_kwargs['has_macaron'] = True
        else:
            raise ValueError(f'Unknown encoder mode {encoder_mode}')

        for i in range(n_layers):
            l = encoder(
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
                qk_rms_norm = qk_rms_norm,
                default_norm = default_norm,
                sandwich_norm = sandwich_norm,
                bias_in_ff = bias_in_ff,
                transformer = transformer,
                **additonal_kwargs,
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

        if cached_kvs is not None:
            assert cached_kv_lengths.max() == cached_kvs.shape[1], 'cached kvs must all be the same length'

        if length is None:
            length = torch.tensor([max_audio_length] * audio_signal.size(0), device=audio_signal.device)
            
    
        audio_signal = torch.transpose(audio_signal, 1, 2)# if self.subsampling_mode != 'stacking' else audio_signal
        audio_signal, length = self.subsampling(audio_signal, lengths = length)
        max_audio_length = audio_signal.size(1)
        ## create masks
        
        mask = torch.arange(max_audio_length, device=audio_signal.device).expand(audio_signal.size(0), max_audio_length) >= length.unsqueeze(1)
    
        rotary_emb_fn = None
   
        full_kv_lengths = length + cached_kv_lengths if cached_kv_lengths is not None else length
        #print(full_kv_lengths, '333333333')
        if self.use_rotary:
            max_seq_len = full_kv_lengths.max()
            q_offset = 0 if cached_kvs is None else cached_kvs.shape[1]
      
            cos, sin = self.rotary_pos_emb(max_seq_len, audio_signal.device)
            rotary_emb_fn = apply_rotary(cos = cos, sin = sin, q_offset = q_offset, learned = self.rotary_pos_emb.learned_freq)
        

        if length.max() == length.min():
            att_mask, mask = None, None
        else:
            full_kv_mask = torch.arange(full_kv_lengths.max(), device=audio_signal.device).expand(audio_signal.size(0), full_kv_lengths.max()) >= full_kv_lengths.unsqueeze(1)
            if audio_signal.device.type == 'cuda' and self.flash_attn:
                att_mask = ~full_kv_mask
            else:
                qmask, kmask = ~mask, ~full_kv_mask
                att_mask = ~(rearrange(qmask, 'b n -> b () n ()') * rearrange(kmask, 'b n -> b () () n'))
                att_mask = att_mask.to(audio_signal.dtype) * -torch.finfo(audio_signal.dtype).max

        pad_mask = mask 
        
        iterim_posteriors = []
        kvs_to_cache = []

        if self.self_condition_subsampling:
            iterim_logits, x_norm = decoder(x=audio_signal, logits=True, return_norm=True)
            iterim_post = torch.nn.functional.softmax(iterim_logits, dim=-1)
            iterim_logposteriors = torch.log(iterim_post)
            iterim_posteriors.append(iterim_logposteriors)
            audio_signal = decoder.integrate_projections(audio_signal, x_norm, decoder.project_back(iterim_post)) 

        for lth, layer in enumerate(self.layers):
            lth_to_grap = lth + 1 if lth + 1 < len(self.layers) and self.shift_kvs else lth
            current_layer_kvs = cached_kvs[:,:,lth_to_grap] if cached_kvs is not None else None

            if self.checkpoint_every_n_layers > 0 and lth % self.checkpoint_every_n_layers == 0:
                audio_signal, kv_to_cache = checkpoint(
                    self.create_custom_forward(layer), 
                    audio_signal, # x
                    att_mask, # att_mask
                    pad_mask, # pad_mask
                    length,
                    current_layer_kvs,
                    self.flash_attn,
                    rotary_emb_fn
                )
    
            else:
                audio_signal, kv_to_cache = layer(
                    x = audio_signal, 
                    attn_mask = att_mask, 
                    pad_mask = pad_mask,
                    length = length,
                    cached_kv = current_layer_kvs,
                    flash_attn = self.flash_attn,
                    rotary_emb_fn = rotary_emb_fn
                )

            
            kvs_to_cache.append(kv_to_cache) # possibly detach and move to cpu ?    
            
            if lth != len(self.layers) - 1 and self.self_conditioning:
                iterim_logits, x_norm = decoder(x=audio_signal, logits=True, return_norm=True)
                iterim_post = torch.nn.functional.softmax(iterim_logits, dim=-1)
                iterim_logposteriors = torch.log(iterim_post)
                iterim_posteriors.append(iterim_logposteriors)
                audio_signal = decoder.integrate_projections(audio_signal, x_norm, decoder.project_back(iterim_post))        

        # stack the posteriors along the first dimension (height, batch, seq_len, dim)
        #print(len(iterim_posteriors),111111111111111111)
        iterim_posteriors = torch.stack(iterim_posteriors, dim=0) if len(iterim_posteriors) > 0 else None
        kvs_to_cache = torch.stack(kvs_to_cache, dim=0)
        kvs_to_cache = rearrange(kvs_to_cache, 'l kv b h n d -> kv b l h n d')
        
        final_posts = decoder(x=decoder.norm(audio_signal), logits=False)

        if self.training and self.rotary_pos_emb is not None:
            self.rotary_pos_emb.reset_if_needed()

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
        qk_rms_norm = False,
        default_norm = DEFAULT_NORM,
        sandwich_norm = False,
        bias_in_ff = True,
        transformer = False,
        **kwargs
    ):
        super().__init__()

        self.d_model = d_model
        self.conv_kernel_size = conv_kernel_size
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.sandwich_norm = sandwich_norm
        self.bias_in_ff = bias_in_ff
        self.trasformer = transformer

        if not self.trasformer:
            self.conv = PreNorm(
                d_model = d_model, 
                fn = ConformerConvolution(
                    d_model = d_model,
                    kernel_size = conv_kernel_size,
                    norm_type = 'batch_renorm'
                ),
                norm = default_norm
            )
            self.do_conv = nn.Dropout(dropout_conv)

        self.ff1 = Scale(0.5, PreNorm(d_model = d_model, fn = ConformerFeedForward(d_model, bias1 = bias_in_ff, bias2 = bias_in_ff), norm = default_norm, sandwich_norm = sandwich_norm))

        self.ff2 = Scale(0.5, PreNorm(d_model = d_model, fn = ConformerFeedForward(d_model, bias1 = bias_in_ff, bias2 = bias_in_ff), norm = default_norm, sandwich_norm = sandwich_norm))
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
                qk_rms_norm = qk_rms_norm,
                **kwargs
            ),
            norm = default_norm,
        )
        self.attn_norm_out = default_norm(d_model) if sandwich_norm else lambda x: x

        self.do_attn_out = nn.Dropout(min(dropout_ff, 0.1)) # don't wan't this too large
        self.norm_out = default_norm(d_model)

            

    def forward(self, x, attn_mask, pad_mask, length, cached_kv = None, flash_attn = True, rotary_emb_fn = None):
        '''
        pad_mask: mask for padding used in conv layers
        attn_mask: attn_mask this should include the cached keys and values
        length: list of lengths of the input sequence
        cached_kv: kvs from previous block-reccurrent time step
        '''

        x = self.do_ff(self.ff1(x)) + x

        attn_out, kv_to_cache = self.attend(
            x = x,
            length = length,
            attn_mask = attn_mask,
            pad_mask = pad_mask,
            cached_kv = cached_kv,
            flash_attn = flash_attn,
            rotary_emb_fn = rotary_emb_fn
        )
        x = self.attn_norm_out(self.do_attn_out(attn_out)) + x
        
        if not self.trasformer:
            x = self.do_conv(self.conv(x, pad_mask = pad_mask)) + x
    
        x = self.do_ff(self.ff2(x)) + x

        x = self.norm_out(x)

        return x, kv_to_cache


class EBranchConformerLayer(nn.Module):
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
        qk_rms_norm = False,
        has_macaron = True,
        default_norm = DEFAULT_NORM,
        sandwich_norm = False,
        bias_in_ff = True,
        **kwargs
    ):
        super().__init__()

        self.d_model = d_model
        self.conv_kernel_size = conv_kernel_size
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.has_macaron = has_macaron
        self.bias_in_ff = bias_in_ff

        assert sandwich_norm == False, 'sandwich norm not supported for EBranchConformerLayer'
        
        if self.has_macaron:
            self.ff1 = Scale(0.5, PreNorm(d_model = d_model, fn = ConformerFeedForward(d_model, bias1 = bias_in_ff, bias2 = bias_in_ff), norm = default_norm))
            self.ff2 = Scale(0.5, PreNorm(d_model = d_model, fn = ConformerFeedForward(d_model, bias1 = bias_in_ff, bias2 = bias_in_ff), norm = default_norm))

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
                qk_rms_norm = qk_rms_norm
            ),
            norm = default_norm
        )

        self.cgmlp = PreNorm(d_model = d_model,
            fn = ConvolutionalGatingMLP(
                size = d_model,
                linear_units = d_model * 6,
                kernel_size = conv_kernel_size,
                dropout_rate = dropout_conv,
                default_norm = default_norm
            ),
            norm = default_norm
        )
        self.do_conv = nn.Dropout(dropout_conv)

        self.depthwise_conv_fusion = torch.nn.Conv1d(
            d_model * 2,
            d_model * 2,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            groups=d_model * 2,
            bias=True,
        )
        self.merge_proj = torch.nn.Linear(d_model * 2, d_model)

        self.do_attn_out = nn.Dropout(min(dropout_ff, 0.1)) # don't wan't this too large
        self.norm_out = default_norm(d_model)

            

    def forward(self, x, attn_mask, pad_mask, length, cached_kv = None, flash_attn = True, rotary_emb_fn = None):
        '''
        pad_mask: mask for padding used in conv layers
        attn_mask: attn_mask this should include the cached keys and values
        length: list of lengths of the input sequence
        cached_kv: kvs from previous block-reccurrent time step
        '''
        if self.has_macaron:
            x = self.do_ff(self.ff1(x)) + x

        # Two branches
        x1, x2 = x, x

        attn_out, kv_to_cache = self.attend(
            x = x1,
            length = length,
            attn_mask = attn_mask,
            pad_mask = pad_mask,
            cached_kv = cached_kv,
            flash_attn = flash_attn,
            rotary_emb_fn = rotary_emb_fn
        )
        x1 = self.do_attn_out(attn_out) 
        
        x2 = self.do_conv(self.cgmlp(x = x2, pad_mask = pad_mask))
        # merge branches
        x_concat = torch.cat([x1, x2], dim=-1)
        if pad_mask is not None:
            x_concat = x_concat.masked_fill(pad_mask.unsqueeze(-1), 0)
        x_tmp = self.depthwise_conv_fusion(x_concat.transpose(1, 2)).transpose(1, 2)
        x = self.do_ff(self.merge_proj(x_tmp)) + x

        if self.has_macaron:
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
        qk_rms_norm=False,
        **kwargs
    ):
        super().__init__()
        self.layer_idx = kwargs.get('layer_idx', None)
        #self.history_vector = torch.nn.Parameter(torch.zeros(2, 1, 1, 1, head_dim), requires_grad=True)


        self.num_position_embeddings = kwargs.get('num_position_embeddings', -1)
        if self.num_position_embeddings > 0: # # B, N, KV, H, D
            self.position_embeddings = nn.Parameter(torch.empty(1, self.num_position_embeddings, 2, n_heads, head_dim))
            nn.init.uniform_(self.position_embeddings, -0.02, 0.02)


        self.n_feats, self.head_dim, self.n_heads = n_feats, head_dim, n_heads
   
        self.activation = nn.Softmax(dim=-1)

        self.dropout_p = dropout

        # softmax_scale is set to None but will default to 1/sqrt(d_k) in FlashAttention
        softmax_scale = None if not qk_rms_norm else 10.0 
        self.flash_attn_fn = FlashAttention(softmax_scale = softmax_scale, attention_dropout = dropout)
        self.flash_attn_c_fn = FlashCrossAttention(softmax_scale = softmax_scale, attention_dropout = dropout)
        ##

        self.qkv_proj = nn.Linear(n_feats, 3 * n_heads * head_dim, bias=bias)


        self.qkv = lambda x: rearrange(self.qkv_proj(x), "b n (h d qkv) -> qkv b n h d", qkv=3, h=n_heads, d=head_dim)

        self.out_proj = nn.Linear(n_heads * head_dim, n_feats, bias=bias)

        self.qk_rms_norm = qk_rms_norm

    
    def attatch_cache(self, kv, cached_kv):
        kv = torch.stack(kv, dim=2)
   
        if cached_kv is None:
            return kv, kv
            
        cached_kv = cached_kv.contiguous()
        new_kv = torch.cat([cached_kv, kv], dim=1) # B, N, KV, H, D

        return new_kv, new_kv.clone()
    
    def sdpa(self, q, k, v, mask): # use to get the attention weights for visualization/debugging
        a_weight = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        a_weight = (a_weight / self.head_dim ** 0.5)
        if mask is not None:
            a_weight = a_weight.masked_fill(mask, -torch.finfo(a_weight.dtype).max)
        a_weight = a_weight.softmax(dim=-1)
        return torch.einsum("b h i j, b h j d -> b h i d", a_weight, v), a_weight

   
        
    def forward(self, x, length, attn_mask=None, pad_mask=None, cached_kv=None, flash_attn = True, rotary_emb_fn = None):
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim
        #print(x.shape, mask.shape)

        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0)


        q, k, v = self.qkv(x)
        if self.qk_rms_norm:
            q, k = map(partial(l2norm, groups = 8), (q, k))

        kv, kv_to_cache = self.attatch_cache([k, v], cached_kv)

       

        if rotary_emb_fn is not None:
            if rotary_emb_fn.learned == False:
                q, kv[:, :, 0] = rotary_emb_fn.apply(q, kv[:, :, 0])
            else:
                k, v = kv[:, :, 0], kv[:, :, 1]
                q, k = rotary_emb_fn.apply(q, k)
                kv = torch.stack([k, v], dim=2)

        if self.num_position_embeddings > 0:
            assert kv.shape[1] <= self.num_position_embeddings, "kv_seq_len should be less than or equal to num_position_embeddings"
            kv[:,:,0] = kv[:,:,0] + self.position_embeddings[:, :kv.shape[1], 0]
            offset = kv.shape[1] - q.shape[1]
            q = q + self.position_embeddings[:, offset:offset+q.shape[1], 1]

     
        ### Flash attention stuff 
        if x.device.type == 'cuda' and flash_attn:
            q, kv = q.contiguous(), kv.contiguous()
            if q.dtype == torch.float32:
                q, kv = q.half(), kv.half()

            if kv.shape[1] == q.shape[1]: # if kv_seq_len == q_seq_len use self attention else use cross attention
                qkv = torch.cat([q[:,:,None], kv], dim=2)
                out = self.flash_attn_fn(qkv, attn_mask)[0] #!!! !!!!!!
            else:
                out = self.flash_attn_c_fn(q, kv)
                if attn_mask is None:
                    out = self.flash_attn_c_fn(q, kv)
                else:
                    q_attn_mask = (attn_mask)[:, -length.max().item():]
                    kv_attn_mask = attn_mask

                    b, qs, qh, qd = q.shape
                    b, kvs, kvn, kh, kd = kv.shape
                    q_up, q_indices, cu_seq_lens, max_seqlen = unpad_input(q, q_attn_mask)
             
                    kv_up, kv_indices, k_cu_seq_lens, max_k_seq_len = unpad_input(kv, kv_attn_mask)
                    
                    out = self.flash_attn_c_fn(
                        q_up, 
                        kv_up, 
                        cu_seqlens = cu_seq_lens.to(torch.int32),
                        max_seqlen = max_seqlen,
                        cu_seqlens_k = k_cu_seq_lens.to(torch.int32),
                        max_seqlen_k = max_k_seq_len,
                    )
                    out = pad_input(out, indices = q_indices, batch = b, seqlen = qs)
            out = out.to(x.dtype) 
           
            out = rearrange(out, "b n h d -> b n (h d)")
        else:
            k, v = rearrange(kv, "b n kv h d -> kv b h n d", kv=2).contiguous()
            q = q.transpose(1, 2).contiguous()
            # print(self.layer_idx, 'h')
            # out, weight = self.sdpa(q, k, v, attn_mask)
            # if self.layer_idx == 3:
            #     torch.save(weight.detach().cpu().half(), 'weight.pt')
            #     exit()
            
            #print('here')
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=False)

            out = rearrange(out, "b h n d -> b n (h d)")
      

        if pad_mask != None:
            out = out.masked_fill(pad_mask.unsqueeze(-1), 0)

        out = self.out_proj(out)
        
        return out, kv_to_cache


class ASRLinearSCDecoder(nn.Module):
    def __init__(self, d_model, vocab_size, norm=False, gated_sc=False, norm_fn=DEFAULT_NORM):
        super().__init__()
        # Add 1 for blank char
        self.num_classes = vocab_size + 1
        self.ff = nn.Linear(d_model, self.num_classes)
        self.reprojection = nn.Linear(self.num_classes, d_model)
        self.norm = norm_fn(d_model) if norm else nn.Identity()
        self.gated_sc = gated_sc
        if self.gated_sc:
            self.gate = nn.Linear(d_model, 1)

    def forward(self, x, logits=False, return_norm=False):
        x_norm = self.norm(x)
        x = self.ff(x_norm)
        x = F.log_softmax(x, dim=-1) if not logits else x
        return x if not return_norm else (x, x_norm)

    def project_back(self, x):
        return self.reprojection(x)

    def integrate_projections(self, x, x_norm, proj1):
        if self.gated_sc:
            gate = torch.sigmoid(self.gate(x_norm))
            
            return x + gate * proj1
        else:
            return x + proj1


class ConvolutionalSpatialGatingUnit(torch.nn.Module):
    """
    Convolutional Spatial Gating Unit (CSGU).
    Adapted from: https://github.com/espnet/espnet/blob/ed6ee209c93fe72b95a687272a5faf69639b7b49/espnet2/asr/layers/cgmlp.py#L84
    """
    def __init__(
        self,
        size: int,
        kernel_size: int,
        dropout_rate: float = 0,
        use_linear_after_conv: bool = False,
        activation = torch.nn.Identity,
        default_norm = DEFAULT_NORM,
    ):
        super().__init__()

        n_channels = size // 2  # split input channels
        self.norm = default_norm(n_channels)
        self.conv = torch.nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size,
            1,
            (kernel_size - 1) // 2,
            groups=n_channels,
        )
        if use_linear_after_conv:
            self.linear = torch.nn.Linear(n_channels, n_channels)
        else:
            self.linear = None

        self.act = activation()

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.espnet_initialization_fn()

    def espnet_initialization_fn(self):
        torch.nn.init.normal_(self.conv.weight, std=1e-6)
        torch.nn.init.ones_(self.conv.bias)
        if self.linear is not None:
            torch.nn.init.normal_(self.linear.weight, std=1e-6)
            torch.nn.init.ones_(self.linear.bias)

    def forward(self, x):
        """Forward method

        Args:
            x (torch.Tensor): (N, T, D)
            gate_add (torch.Tensor): (N, T, D/2)

        Returns:
            out (torch.Tensor): (N, T, D/2)
        """

        x_r, x_g = x.chunk(2, dim=-1)

        x_g = self.norm(x_g)  # (N, T, D/2)
        x_g = self.conv(x_g.transpose(1, 2)).transpose(1, 2)  # (N, T, D/2)

        if self.linear is not None:
            x_g = self.linear(x_g)

        x_g = self.act(x_g)
        out = x_r * x_g  # (N, T, D/2)
        out = self.dropout(out)
        return out


class ConvolutionalGatingMLP(torch.nn.Module):
    """
    Convolutional Gating MLP (cgMLP).
    Adapted from https://github.com/espnet/espnet/blob/ed6ee209c93fe72b95a687272a5faf69639b7b49/espnet2/asr/layers/cgmlp.py#L84
    """

    def __init__(
        self,
        size: int,
        linear_units: int,
        kernel_size: int,
        dropout_rate: float,
        use_linear_after_conv: bool = False,
        activation: torch.nn.Module = torch.nn.Identity,
        default_norm: any = DEFAULT_NORM,
    ):
        super().__init__()

        self.channel_proj1 = torch.nn.Sequential(
            torch.nn.Linear(size, linear_units), torch.nn.GELU()
        )
        self.csgu = ConvolutionalSpatialGatingUnit(
            size=linear_units,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            use_linear_after_conv=use_linear_after_conv,
            activation=activation,
            default_norm=default_norm,
        )
        self.channel_proj2 = torch.nn.Linear(linear_units // 2, size)

    def forward(self, x, pad_mask):

        if pad_mask is not None: 
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        x = self.channel_proj1(x)  # size -> linear_units
        x = self.csgu(x)  # linear_units -> linear_units/2
        out = self.channel_proj2(x)  # linear_units/2 -> size
        
        return out