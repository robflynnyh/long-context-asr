import torch, torch.nn as nn, torch.nn.functional as F
import apex
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
DEFAULT_NORM, RMSNorm, LayerNorm = apex.normalization.FusedRMSNorm, apex.normalization.FusedRMSNorm, apex.normalization.FusedLayerNorm
PreNorm, Scale = wrappers.PreNorm, wrappers.Scale
from flash_attn.flash_attention import FlashAttention
from flash_attn.modules.mha import FlashCrossAttention
from flash_attn.bert_padding import unpad_input, pad_input
from lcasr.models.base import BaseModel
import warnings



# TODO: 
# -. remove caching stuff as it is not used anymore
# -. change name of class to just conforer or something
# -. remove intermediate losses stuff as it is not used anymore 

class SCConformerXL(BaseModel): 
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

        if kwargs.get('padded_forward_pad_to', None) is not None: self.use_padded_forward(kwargs['padded_forward_pad_to'])

    def padded_forward(self, audio_signal):
        '''
            audio_signal: (batch_size, feat, time)
            for inference when we want the input to be equal to a certain length
        '''
        assert audio_signal.shape[0] == 1, 'padded forward only implemented for batch size 1!'
        if self.pad_to <= audio_signal.shape[-1]:
            return self.main_forward(audio_signal)
        else:
            #print(f' input shape: {audio_signal.shape} ')
            pad_amount = self.pad_to - audio_signal.shape[-1] 
            # pad equal amount on both sides
            pad_left = pad_amount // 2
            pad_right = pad_amount - pad_left
        
            audio_signal = F.pad(audio_signal, (pad_left, pad_right), 'constant', 0)
            #print(f' padded input shape: {audio_signal.shape} ')
            out =  self.main_forward(audio_signal)
            final_posteriors = out['final_posteriors']
            out_n = final_posteriors.shape[1]
            #print(audio_signal.shape, final_posteriors.shape)
            downsample_factor = audio_signal.shape[-1] // out_n
            remove_left, remove_right = pad_left // downsample_factor, pad_right // downsample_factor
            final_posteriors = final_posteriors[:, remove_left: -remove_right]
            
            length = torch.LongTensor([final_posteriors.shape[1]])
            return {
                'final_posteriors': final_posteriors,
                'length': length,
            }

    def use_padded_forward(self, pad_to): 
        self.main_forward = self.forward
        self.pad_to = pad_to
        self.forward = self.padded_forward
        
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

        decoder = self.decoder
        max_audio_length: int = audio_signal.size(-1)

        if cached_kvs is not None:
            assert cached_kv_lengths.max() == cached_kvs.shape[1], 'cached kvs must all be the same length'

        if length is None:
            length = torch.tensor([max_audio_length] * audio_signal.size(0), device=audio_signal.device)
            
        audio_signal = torch.transpose(audio_signal, 1, 2)
        audio_signal, length = self.subsampling(audio_signal, lengths = length) if not self.checkpoint_subsampling else checkpoint(self.create_custom_forward(self.subsampling), audio_signal, length)

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
        
        kvs_to_cache = []

        audio_signal = self.fourier_pos_enc(audio_signal)
        
        for lth, layer in enumerate(self.layers):
            current_layer_kvs = cached_kvs[:,:,lth] if cached_kvs is not None else None

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
            kvs_to_cache.append(kv_to_cache)
            
            if lth != len(self.layers) - 1 and self.self_conditioning:
                iterim_post = torch.nn.functional.softmax(decoder(x=audio_signal, logits=True), dim=-1)
                audio_signal = decoder.integrate_projections(audio_signal, decoder.project_back(iterim_post))        

        # stack the posteriors along the first dimension (height, batch, seq_len, dim)
        kvs_to_cache = torch.stack(kvs_to_cache, dim=0)
        kvs_to_cache = rearrange(kvs_to_cache, 'l kv b h n d -> kv b l h n d')
        
        audio_signal = decoder.norm(audio_signal) if self.legasee_double_norm else audio_signal
        final_posts = decoder(x = audio_signal, logits = return_logits) # having decoder.norm should have been removed is sortof a bug but probably doesn't matter

        if self.training and self.rotary_pos_emb is not None:
            self.rotary_pos_emb.reset_if_needed()

        return {
            'final_posteriors': final_posts,
            'kvs_to_cache': kvs_to_cache,
            'length': length,
            'full_kv_lengths': full_kv_lengths, # kv cache is returned, however we don't use this and is left over from prior experiments
        }



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

            

    def forward(self, x, attn_mask, pad_mask, length, cached_kv = None, flash_attn = True, rotary_emb_fn = None):
        '''
        pad_mask: mask for padding used in conv layers
        attn_mask: attn_mask this should include the cached keys and values
        length: list of lengths of the input sequence
        cached_kv: kvs from previous block-reccurrent time step
        '''

        if not self.trasformer:
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

        self.num_position_embeddings = kwargs.get('num_position_embeddings', -1)
        if self.num_position_embeddings > 0: # # B, N, KV, H, D
            self.position_embeddings = nn.Parameter(torch.empty(1, self.num_position_embeddings, 2, n_heads, head_dim))
            nn.init.uniform_(self.position_embeddings, -0.02, 0.02)


        self.n_feats, self.head_dim, self.n_heads = n_feats, head_dim, n_heads
   
        self.activation = nn.Softmax(dim=-1)

        self.dropout_p = dropout

        # softmax_scale is set to None but will default to 1/sqrt(d_k) in FlashAttention
        self.flash_attn_fn = FlashAttention(softmax_scale = None, attention_dropout = dropout)
        self.flash_attn_c_fn = FlashCrossAttention(softmax_scale = None, attention_dropout = dropout)

        self.qkv_proj = nn.Linear(n_feats, 3 * n_heads * head_dim, bias=bias)

        self.qkv = lambda x: rearrange(self.qkv_proj(x), "b n (h d qkv) -> qkv b n h d", qkv=3, h=n_heads, d=head_dim)

        self.out_proj = nn.Linear(n_heads * head_dim, n_feats, bias=bias)

    
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

        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0)

        q, k, v = self.qkv(x)
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
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=False)
            out = rearrange(out, "b h n d -> b n (h d)")
      
        if pad_mask != None:
            out = out.masked_fill(pad_mask.unsqueeze(-1), 0)

        out = self.out_proj(out)
        
        return out, kv_to_cache



