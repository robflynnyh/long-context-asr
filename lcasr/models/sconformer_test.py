import torch, torch.nn as nn, torch.nn.functional as F

from torch.utils.checkpoint import checkpoint # # gradient/activation checkpointing
from einops import rearrange
from functools import partial
from lcasr.components import fused_dense, subsampling, convolution, decoder, wrappers
from lcasr.components.positional_encodings import RotaryPositionalEmbedding, apply_rotary, LearnableFourierPosEnc
from lcasr.utils.helpers import exists
from lcasr.components.helpers import get_act
ConformerConvolution = convolution.ConformerConvolution
ConformerFeedForward = fused_dense.FusedMLP
ConvSubsampling, StackingSubsampling = subsampling.ConvSubsampling, subsampling.StackingSubsampling

try: from apex.normalization import FusedRMSNorm as DEFAULT_NORM, FusedRMSNorm as RMSNorm, FusedLayerNorm as LayerNorm
except: 
    from lcasr.components.normalisation import RMSNorm as RMSNorm, RMSNorm as DEFAULT_NORM
    from torch.nn import LayerNorm as LayerNorm

# from lcasr.components.normalisation import RMSNorm as RMSNorm, RMSNorm as DEFAULT_NORM
# from torch.nn import LayerNorm as LayerNorm

PreNorm, Scale = wrappers.PreNorm, wrappers.Scale

from lcasr.components.attention import Attention
from lcasr.models.base import BaseModel
import warnings


# TODO: 
# -. remove caching stuff as it is not used anymore

class SCConformerTest(BaseModel): 
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
        d_model = 768,
        n_heads = 6,
        head_dim = 128,
        expansion_factor = 4,
        dropout_ff = 0.0,
        dropout_conv = 0.0,
        dropout_attn = 0.0,
        checkpoint_every_n_layers = 0,
        conv_kernel_size = 9,
        conv_expansion_factor = 1,
        decoder_norm = False,
        use_rotary = False,
        rotary_interpolation_factor = 1.0, # https://arxiv.org/abs//2306.15595 Extending Context Window of Large Language Models via Positional Interpolation
        learned_rotary = False,
        fourier_pos_enc = False,
        self_conditioning = True,
        default_norm = 'layer_norm',
        sandwich_norm = False,
        bias_in_ff = False,
        transformer=False, # disable convolutions
        legasee_double_norm = True, # norm is applied twice before final output projection, was orignally a bug, kept got compatibility with older checkpoints
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
        self.learned_rotary = learned_rotary
        self.self_conditioning = self_conditioning
        self.sandwich_norm = sandwich_norm
        self.bias_in_ff = bias_in_ff
        self.transformer = transformer

        self.input_bnorm = nn.BatchNorm1d(d_model)
    
        self.legasee_double_norm = legasee_double_norm

        self.checkpoint_subsampling = kwargs.get('checkpoint_subsampling', False) # whether to perform activation checkpointing on subsampling layers

        accepted_norms = ['rms_norm', 'layer_norm']
        accepted_subsampling_acts = ['silu', 'relu', 'gelu', 'none']
        assert subsampling_act in accepted_subsampling_acts, f'subsampling_act must be one of {accepted_subsampling_acts} (got {subsampling_act})'
        assert default_norm in accepted_norms, f'default_norm must be one of {accepted_norms} (got {default_norm})'
        default_norm = RMSNorm if default_norm == 'rms_norm' else LayerNorm

        subsampling_act = get_act(subsampling_act)

        self.flash_attn = kwargs.get('flash_attn', True)
      
        self.checkpoint_every_n_layers = checkpoint_every_n_layers

        self.dropout_ff = dropout_ff
        self.dropout_conv = dropout_conv
        self.dropout_attn = dropout_attn

        self.subsampling_mode = subsampling
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels if subsampling_conv_channels != -1 else d_model

        self.whitelist_weight_decay_modules = (nn.LayerNorm, RMSNorm, LayerNorm, convolution.BatchRenorm1d, nn.GroupNorm) # don't decay
        self.blacklist_weight_decay_modules = (nn.Linear, ConformerFeedForward, nn.Conv1d, nn.Conv2d, RotaryPositionalEmbedding)

        self.decoder_norm = decoder_norm

        self.use_rotary = use_rotary

        self.rotary_pos_emb = None
        if self.use_rotary:
            self.rotary_pos_emb = RotaryPositionalEmbedding(
                dim = head_dim,
                base = kwargs.get('rotary_base_freq', 10000),
                learned_freq = learned_rotary,
                rotary_interpolation_factor = rotary_interpolation_factor
            )
        self.fourier_pos_enc = LearnableFourierPosEnc(d_model) if fourier_pos_enc else nn.Identity()

        self.decoder = decoder.ASRLinearSCDecoder(
            d_model = d_model,
            vocab_size = vocab_size,
            norm = decoder_norm,
            norm_fn = default_norm,
            **kwargs
        )

        subsampling_args = {'subsampling_factor': self.subsampling_factor, 'feat_in': self.feat_in, 'feat_out': self.d_model, 'norm_out': subsampling_norm_out,}
        self.subsampling = \
            ConvSubsampling(subsampling = self.subsampling_mode, conv_channels = self.subsampling_conv_channels, activation = subsampling_act, **subsampling_args) \
                if subsampling != 'stacking' else \
                     StackingSubsampling(norm = True if not subsampling_norm_out else False, default_norm = default_norm, **subsampling_args)

        
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
                default_norm = default_norm,
                sandwich_norm = sandwich_norm,
                bias_in_ff = bias_in_ff,
                transformer = transformer,
                conv_expansion_factor = conv_expansion_factor,
                **kwargs
            )
            self.layers.append(l)

            self.cross_decoder = CrossAttnDecoder(
                vocab_size = None,
                n_layers=n_layers,
                d_model=d_model,
            )
        
    def forward(
            self, 
            audio_signal,
            length = None, 
            cached_kvs = None, 
            cached_kv_lengths = None, 
            return_logits = False
        ):
        '''
        audio_signal: (batch_size, time, feat)
        length: (batch_size,)
        cached_kvs: (kv i.e 2, batch_size, layers, heads, time, head_dim)
        '''
        # if audio_signal.size(-1) < 2048:
        #     to_pad = 2048 - audio_signal.size(-1)
        #     to_pad = torch.zeros(audio_signal.size(0), audio_signal.size(1), to_pad, device=audio_signal.device)
        #     print(audio_signal.device)
        #     audio_signal = torch.cat([audio_signal, to_pad], dim=-1)

        decoder = self.decoder
        max_audio_length: int = audio_signal.size(-1)

        if cached_kvs is not None:
            assert cached_kv_lengths.max() == cached_kvs.shape[1], 'cached kvs must all be the same length'
        
        if length is None:
            length = torch.tensor([max_audio_length] * audio_signal.size(0), device=audio_signal.device)
        
        audio_signal = torch.transpose(audio_signal, 1, 2)
        audio_signal, length = self.subsampling(audio_signal, lengths = length) if not self.checkpoint_subsampling else checkpoint(self.create_custom_forward(self.subsampling), audio_signal, length)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        audio_signal = self.input_bnorm(audio_signal)
        audio_signal = torch.transpose(audio_signal, 1, 2)
        # audio_signal = b, t, f
        

        max_audio_length = audio_signal.size(1)
        ## create masks
        
        mask = torch.arange(max_audio_length, device=audio_signal.device).expand(audio_signal.size(0), max_audio_length) >= length.unsqueeze(1)
    
        rotary_emb_fn = None
   
        full_kv_lengths = length + cached_kv_lengths if cached_kv_lengths is not None else length
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
    
        audio_signal = self.fourier_pos_enc(audio_signal)
        
        for lth, layer in enumerate(self.layers):

            if self.checkpoint_every_n_layers > 0 and lth % self.checkpoint_every_n_layers == 0:
                audio_signal = checkpoint(
                    self.create_custom_forward(layer), 
                    audio_signal, # x
                    att_mask, # att_mask
                    length,
                    pad_mask, # pad_mask
                    self.flash_attn,
                    rotary_emb_fn,
                )
            else:
                audio_signal = layer(
                    x = audio_signal, 
                    attn_mask = att_mask, 
                    length = length,
                    pad_mask = pad_mask,
                    flash_attn = self.flash_attn,
                    rotary_emb_fn = rotary_emb_fn
                )
            
            if lth != len(self.layers) - 1 and self.self_conditioning:
                iterim_post = torch.nn.functional.softmax(decoder(x=audio_signal, logits=True), dim=-1)
                audio_signal = decoder.integrate_projections(audio_signal, decoder.project_back(iterim_post))        

        audio_signal = self.cross_decoder(audio_signal, length)
        audio_signal = decoder.norm(audio_signal) if self.legasee_double_norm else audio_signal
        final_posts = decoder(x = audio_signal, logits = return_logits) # having decoder.norm should have been removed is sortof a bug but probably doesn't matter

        if self.training and self.rotary_pos_emb is not None:
            self.rotary_pos_emb.reset_if_needed()



        return {'final_posteriors': final_posts, 'length': length,}


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
        transformer = False,
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
        self.trasformer = transformer

        
        if not self.trasformer:
        
            self.conv = PreNorm(
                d_model = d_model, 
                fn = ConformerConvolution(
                    d_model = d_model,
                    kernel_size = conv_kernel_size,
                    norm_type = kwargs.get('conv_norm', 'batch_renorm'),
                    exp_factor = conv_expansion_factor,
                ),
                norm = default_norm
            )
            self.do_conv = nn.Dropout(dropout_conv)

        if not self.trasformer:
            self.ff1 = Scale(0.5, PreNorm(
                d_model = d_model, 
                fn = ConformerFeedForward(
                    d_model, 
                    bias1 = bias_in_ff, 
                    bias2 = bias_in_ff,
                    checkpoint_lvl = kwargs.get('ff_checkpoint_lvl', 0)
                ), 
                norm = default_norm, 
                sandwich_norm = sandwich_norm
            ))
        
        self.ff2 = Scale(0.5, PreNorm(
            d_model = d_model, 
            fn = ConformerFeedForward(
                d_model, 
                bias1 = bias_in_ff, 
                bias2 = bias_in_ff,
                checkpoint_lvl = kwargs.get('ff_checkpoint_lvl', 0)
            ), 
            norm = default_norm, 
            sandwich_norm = sandwich_norm
        ))

        self.do_ff = nn.Dropout(dropout_ff)

        self.has_attention = kwargs.get('has_attention', True)

        if self.has_attention:
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
            )
            self.attn_norm_out = default_norm(d_model) if sandwich_norm else lambda x: x
            self.do_attn_out = nn.Dropout(min(dropout_ff, 0.1)) # don't wan't this too large

        self.norm_out = default_norm(d_model)

            

    def forward(self, x, attn_mask, length, pad_mask, flash_attn = True, rotary_emb_fn = None):
        '''
        pad_mask: mask for padding used in conv layers
        attn_mask: attn_mask this should include the cached keys and values
        length: list of lengths of the input sequence
        cached_kv: kvs from previous block-reccurrent time step
        '''

        if not self.trasformer:
            x = self.do_ff(self.ff1(x)) + x

        if self.has_attention:
            x = self.attn_norm_out(self.do_attn_out(self.attend(
                x = x,
                attn_mask = attn_mask,
                length = length,
                pad_mask = pad_mask,
                flash_attn = flash_attn,
                rotary_emb_fn = rotary_emb_fn
            ))) + x
        
        if not self.trasformer:
            x = self.do_conv(self.conv(x, pad_mask = pad_mask)) + x
    
        x = self.do_ff(self.ff2(x)) + x

        x = self.norm_out(x)
        return x

class CrossAttention(nn.Module):
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
        self.flash_attn = kwargs.get('flash_attn', True)

        self.n_feats, self.head_dim, self.n_heads = n_feats, head_dim, n_heads
   
        self.activation = nn.Softmax(dim=-1)

        self.dropout_p = dropout
        self.causal = False
    

        self.qkv_proj = nn.Linear(n_feats, 3 * n_heads * head_dim, bias=bias)
        self.q_proj = nn.Linear(n_feats, n_heads * head_dim, bias=bias)
        self.kv_proj = nn.Linear(n_feats, 2 * n_heads * head_dim, bias=bias)

        self.kv = lambda x: rearrange(self.kv_proj(x), "b n (h d kv) -> kv b n h d", kv=2, h=n_heads, d=head_dim)
        self.q = lambda x: rearrange(self.q_proj(x), "b n (h d) -> b n h d", h=n_heads, d=head_dim)

        self.out_proj = nn.Linear(n_heads * head_dim, n_feats, bias=bias)

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
        
    def forward(self, xq, xkv, kv_mask = None, attn_mask=None, rotary_emb_fn=None):
        H, D = self.n_heads, self.head_dim


        q = self.q(xq)
        k, v = self.kv(xkv)
        kv = torch.stack([k, v], dim=2)
        q, kv = self.apply_rotary(q, kv, rotary_emb_fn)

        k, v = rearrange(kv, "b n kv h d -> kv b h n d", kv=2).contiguous()
        q = q.transpose(1, 2).contiguous()
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0 if not self.training else self.dropout_p, is_causal=False)
        out = rearrange(out, "b h n d -> b n (h d)")


        out = self.out_proj(out)
        
        return out


class CrossAttnDecoder(nn.Module):
    def __init__(
        self,
        vocab_size = 4096,
        n_layers = 3,
        d_model = 768,
        n_heads = 6,
        head_dim = 128,
        expansion_factor = 4,
        dropout_ff = 0.0,
        dropout_attn = 0.0,
        decoder_norm = True,
        use_rotary = True,
        rotary_interpolation_factor = 1.0, # https://arxiv.org/abs//2306.15595 Extending Context Window of Large Language Models via Positional Interpolation
        default_norm = 'rms_norm',
        bias_in_ff = False,
        **kwargs
    ):
        super().__init__()


        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.expansion_factor = expansion_factor
        self.dropout_ff = dropout_ff
        self.dropout_attn = dropout_attn
        self.decoder_norm = decoder_norm
        self.use_rotary = use_rotary
        self.rotary_interpolation_factor = rotary_interpolation_factor
        self.bias_in_ff = bias_in_ff
        self.default_norm = default_norm
        self.flash_attn = kwargs.get('flash_attn', True)

        self.embedding = nn.parameter.Parameter(nn.Embedding(1, d_model).weight / 2)

        self.pos_enc = LearnableFourierPosEnc(d_model, hidden_dim=kwargs.get('fourier_pos_hidden_dim', 64))
        self.dropout_emb = kwargs.get('dropout_emb', 0.0)
        self.ff_out_dropout = kwargs.get('ff_out_dropout', 0.0)

        accepted_norms = ['rms_norm', 'layer_norm']
        assert default_norm in accepted_norms, f'default_norm must be one of {accepted_norms} (got {default_norm})'
        default_norm = RMSNorm if default_norm == 'rms_norm' else LayerNorm
        self.acoustic_norm = default_norm(d_model)

        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(
                    d_model = d_model,
                    fn = Attention(
                        n_feats = d_model,
                        n_heads = n_heads,
                        head_dim = head_dim,
                        bias = bias_in_ff,
                        causal = False,
                        dropout = dropout_attn,
                        **kwargs
                    ),
                    norm = default_norm
                ),
                PreNorm(
                    d_model = d_model, 
                    fn = CrossAttention(
                        n_feats = d_model,
                        n_heads = n_heads,
                        head_dim = head_dim,
                        bias = bias_in_ff,
                        dropout = dropout_attn,
                        **kwargs
                    ), 
                    norm = default_norm
                ),
                PreNorm(
                    d_model = d_model, 
                    fn = ConformerFeedForward(d_model, bias1 = bias_in_ff, bias2 = bias_in_ff),
                    norm = default_norm,
                )
            ]))
            

   

        self.rotary_pos_emb = None
        if self.use_rotary:
            self.rotary_pos_emb = RotaryPositionalEmbedding(
                dim = head_dim,
                base = kwargs.get('rotary_base_freq', 10000),
                learned_freq = False,
                rotary_interpolation_factor = rotary_interpolation_factor
            )

    def forward(
            self,
            a_hidden: torch.Tensor, 
            a_lengths: torch.Tensor,
        ):
        '''
        tokens: (batch, seq_len) - target text sequence
        a_hidden: (batch, seq_len, dim) - encoder output
        pos_enc_module: positional encoding module - instance of PosEnc
        '''
        tokens = self.embedding.unsqueeze(0).expand(a_hidden.size(0), a_hidden.size(1), -1)
        lengths = a_lengths
        x = tokens # self.pos_enc(tokens, lengths=lengths)
        x = F.dropout(x, p=self.dropout_emb, training=self.training)
        
        
        if a_lengths.max() == a_lengths.min(): kv_mask = None # if all the same length don't bother with the mask
        else: kv_mask = ~(torch.arange(a_hidden.shape[1], device=a_hidden.device).expand(a_hidden.size(0), a_hidden.shape[1]) >= a_lengths.unsqueeze(1))

        rotary_emb_fn = None
        if self.use_rotary:
            max_seq_len = tokens.shape[1]
            q_offset = 0 
            cos, sin = self.rotary_pos_emb(max_seq_len, tokens.device)
            rotary_emb_fn = apply_rotary(cos = cos, sin = sin, q_offset = q_offset, learned = self.rotary_pos_emb.learned_freq)

        attn_mask = None
        if kv_mask is not None:
            attn_mask = ~(rearrange(~kv_mask, 'b n -> b () n ()') * rearrange(~kv_mask, 'b n -> b () () n'))

        
        for lth, (self_attn, cross_attn, ff_out) in enumerate(self.layers):
            x = self_attn(x, length=lengths, flash_attn=False, rotary_emb_fn=rotary_emb_fn) + x
            x = cross_attn(x, xkv = a_hidden, kv_mask=kv_mask, attn_mask=attn_mask, rotary_emb_fn=rotary_emb_fn) + x
            x = F.dropout(ff_out(x), p=self.ff_out_dropout, training=self.training) + x


        return x



if __name__ == '__main__':
    # run test
    model = SCConformerXL(vocab_size=4096, head_dim=256, n_heads=3, attention_window_size=128)
    audio = torch.randn(2, 80, 1000)
    lengths = torch.tensor([1000, 500])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    audio = audio.to(device)
    lengths = lengths.to(device)
    out = model(audio, length=lengths)
    print(out['final_posteriors'].shape)
    