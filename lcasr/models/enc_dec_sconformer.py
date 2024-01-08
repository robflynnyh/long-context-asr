import torch, torch.nn as nn, torch.nn.functional as F
import apex
from torch.utils.checkpoint import checkpoint # # gradient/activation checkpointing
from einops import rearrange
from functools import partial
from lcasr.components import fused_dense, subsampling, convolution, decoder, wrappers
from lcasr.components.rotary_emb import RotaryPositionalEmbedding, apply_rotary
from lcasr.utils.helpers import exists
from lcasr.components.wrappers import Scale
ConformerConvolution = convolution.ConformerConvolution
ConformerFeedForward = fused_dense.FusedMLP
ConvSubsampling, StackingSubsampling = subsampling.ConvSubsampling, subsampling.StackingSubsampling
DEFAULT_NORM, RMSNorm, LayerNorm = apex.normalization.FusedRMSNorm, apex.normalization.FusedRMSNorm, apex.normalization.FusedLayerNorm
PreNorm, Scale = wrappers.PreNorm, wrappers.Scale
from flash_attn.flash_attention import FlashAttention
from flash_attn.modules.mha import FlashCrossAttention
from flash_attn.bert_padding import unpad_input, pad_input
from lcasr.components.helpers import get_act
from lcasr.models.base import BaseModel
import math

import warnings

def add_eos(tokens, eos_id, token_lens):
    tokens[torch.arange(tokens.shape[0], device=tokens.device, dtype=torch.long), (token_lens - 1).to(torch.long)] = eos_id 
    return tokens

def token_lens_to_mask(token_lens, max_len=None):
    max_len = token_lens.max() if max_len is None else max_len
    mask = torch.arange(max_len, device=token_lens.device)[None, :] < token_lens[:, None]
    return mask

def mark_padding(targets, mask, pad_id):
    targets[~mask] = pad_id
    return targets

class LearnableFourierPosEnc(torch.nn.Module): # code taken from espnet: https://espnet.github.io/espnet/_modules/espnet/nets/pytorch_backend/transformer/embedding.html#LearnableFourierPosEnc
    """Learnable Fourier Features for Positional Encoding.

    See https://arxiv.org/pdf/2106.02795.pdf

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
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
        max_len=5000,
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
        self.max_len = max_len

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

    def extend_pe(self, x):
        """Reset the positional encodings."""
        position_v = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1).to(x)

        cosine = torch.cos(torch.matmul(position_v, self.w_r))
        sine = torch.sin(torch.matmul(position_v, self.w_r))
        pos_enc = torch.cat((cosine, sine), -1)
        pos_enc /= math.sqrt(self.d_model)

        if self.hidden_dim is None:
            return pos_enc.unsqueeze(0)
        else:
            return self.mlp(pos_enc.unsqueeze(0))


    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        pe = self.extend_pe(x)
        x = x * self.xscale + pe
        return self.dropout(x)


class EncDecSconformer(BaseModel): 
    def __init__(
        self,
        vocab_size = 4096,
        feat_in = 80,
        subsampling = 'dw_striding',
        subsampling_factor = 8,
        subsampling_conv_channels = 256,
        subsampling_act = 'silu',
        subsampling_norm_out = False,
        self_condition_subsampling = False,
        n_layers = 3,
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
        decoder_norm = True,
        use_rotary = True,
        rotary_interpolation_factor = 1.0, # https://arxiv.org/abs//2306.15595 Extending Context Window of Large Language Models via Positional Interpolation
        learned_rotary = False,
        self_conditioning = True,
        default_norm = 'layer_norm',
        sandwich_norm = False,
        bias_in_ff = False,
        transformer=False, # disable convolutions
        ctc_loss_weight = 0.5,
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
        self.self_conditioning = self_conditioning if ctc_loss_weight > 0 else False # no ctc decoder
        self.sandwich_norm = sandwich_norm
        self.bias_in_ff = bias_in_ff
        self.transformer = transformer
        self.self_condition_subsampling = self_condition_subsampling
        self.legasee_double_norm = legasee_double_norm

        self.ctc_loss_weight = ctc_loss_weight

        # self.abs_pos_enc = PosEnc(d_model)
        self.pos_enc = LearnableFourierPosEnc(d_model, hidden_dim=64)

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

        self.ctc_decoder = decoder.ASRLinearSCDecoder(
            d_model = d_model,
            vocab_size = vocab_size,
            norm = decoder_norm,
            norm_fn = default_norm,
        ) if ctc_loss_weight > 0 else None

        self.language_model_decoder = CrossAttnDecoder(
            vocab_size = vocab_size,
            n_layers = n_layers,
            d_model = d_model,
            n_heads = n_heads,
            head_dim = head_dim,
            expansion_factor = expansion_factor,
            dropout_attn = dropout_attn,
            dropout_ff = dropout_ff,
            decoder_norm=decoder_norm,
            bias_in_ff=bias_in_ff,
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

    @staticmethod
    def create_custom_forward(module): # for activation checkpointing allow passing dictionary as the argument to the module
        def custom_forward(*args, **kwargs):
            return module(*args, **kwargs)
        return custom_forward


    def calc_loss(
            self, 
            audio_signal,
            text_sequence,
            a_lengths,
            t_lengths,
        ):
        # add bos to text sequence
        text_sequence_bos = F.pad(text_sequence, (1, 0), value=0)
        target_lengths_bos = t_lengths + 1
        
        out = self.forward(audio_signal, text_sequence_bos, a_lengths)
        ctc_out, lm_out, a_length_out = out['final_posteriors_ctc'], out['final_posteriors_lm'], out['length']

        if self.ctc_loss_weight > 0.0:
            ctc_loss = F.ctc_loss(
                log_probs = rearrange(ctc_out, 'b n c -> n b c'),
                targets = text_sequence,
                input_lengths = a_length_out,
                target_lengths = t_lengths,
                reduction = 'sum',
                blank = ctc_out.shape[-1] - 1
            )
            a_sum = a_lengths.sum()
            ctc_loss_to_show = (ctc_loss / a_sum).item() * 100
            ctc_loss_to_bwd = ctc_loss / (ctc_out.shape[1] * ctc_out.shape[0]) * 100
        else:
            ctc_loss_to_show, ctc_loss_to_bwd = 0, 0

        targets = text_sequence_bos.clone()
        targets[:, :-1] = targets[:, 1:]
        if target_lengths_bos.max() == target_lengths_bos.min(): targets[:, -1] = 0
        else:
            targets = add_eos(targets, eos_id = 0, token_lens = target_lengths_bos)
        mask = token_lens_to_mask(target_lengths_bos)
        targets = mark_padding(targets, mask, pad_id = -100)

        predictions = lm_out
        lm_loss = F.cross_entropy(
            input = rearrange(predictions, 'b n c -> (b n) c'),
            target = rearrange(targets, 'b n -> (b n)'),
            ignore_index = -100,
            reduction = 'sum'
        )
        lm_loss_to_show = (lm_loss / t_lengths.sum()).item()
        lm_loss_to_bwd = lm_loss / (predictions.shape[0] * predictions.shape[1])

        loss_to_show = ctc_loss_to_show * self.ctc_loss_weight + lm_loss_to_show * (1 - self.ctc_loss_weight)
        loss = ctc_loss_to_bwd * self.ctc_loss_weight + lm_loss_to_bwd * (1 - self.ctc_loss_weight) 

        wandb_log_data = {
            'loss': loss_to_show,
            'ctc_loss': ctc_loss_to_show,
            'lm_loss': lm_loss_to_show,
        }

        return {
            'loss': loss,
            'display_losses': wandb_log_data,
            'ctc_posteriors': ctc_out,
            'lm_posteriors': lm_out,
            'length': a_length_out,
        }
        



    def forward(
            self, 
            audio_signal,
            text_sequence = None, 
            length = None, 
            cached_kvs = None, 
            cached_kv_lengths = None, 
            return_logits = False
        ):
        max_audio_length: int = audio_signal.size(-1)

        if cached_kvs is not None:
            assert cached_kv_lengths.max() == cached_kvs.shape[1], 'cached kvs must all be the same length'

        if length is None:
            length = torch.tensor([max_audio_length] * audio_signal.size(0), device=audio_signal.device)
            
        audio_signal = torch.transpose(audio_signal, 1, 2)
        audio_signal, length = self.subsampling(audio_signal, lengths = length) if not self.checkpoint_subsampling else checkpoint(self.create_custom_forward(self.subsampling), audio_signal, length)
        #audio_signal = self.abs_pos_enc(audio_signal, scale = self.pos_enc_scale)
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

        audio_signal = self.pos_enc(audio_signal)

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
                iterim_post = torch.nn.functional.softmax(self.ctc_decoder(x=audio_signal, logits=True), dim=-1)
                audio_signal = self.ctc_decoder.integrate_projections(audio_signal, self.ctc_decoder.project_back(iterim_post))        

        # stack the posteriors along the first dimension (height, batch, seq_len, dim)
        kvs_to_cache = torch.stack(kvs_to_cache, dim=0)
        kvs_to_cache = rearrange(kvs_to_cache, 'l kv b h n d -> kv b l h n d')
        
        final_posts_ctc = None
        if self.ctc_loss_weight > 0:
            final_posts_ctc = self.ctc_decoder(x = self.ctc_decoder.norm(audio_signal) if self.legasee_double_norm else audio_signal, logits = return_logits) # having decoder.norm should have been removed is sortof a bug but probably doesn't matter

        final_posts_lm = None
        if text_sequence is not None:
            final_posts_lm = self.language_model_decoder(
                tokens = text_sequence,
                a_hidden = audio_signal,
                a_lengths = length,
            )


        return {
            'final_posteriors_ctc': final_posts_ctc,
            'final_posteriors_lm': final_posts_lm,
            'length': length,
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
        causal=False,
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
        self.causal = causal
        # softmax_scale is set to None but will default to 1/sqrt(d_k) in FlashAttention
        self.flash_attn_fn = FlashAttention(softmax_scale = None, attention_dropout = dropout)
        self.flash_attn_c_fn = FlashCrossAttention(softmax_scale = None, attention_dropout = dropout, causal = causal)

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
                out = self.flash_attn_fn(qkv, attn_mask, causal=self.causal)[0] #!!! !!!!!!
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

class CrossAttentionPosEnc(nn.Module):
    def __init__(
        self,
        heads,
        d_head,
        seq_len=64,
    ):
        super().__init__()
        self.heads = heads
        self.d_head = d_head
        self.d_pos = d_head 
        
        
        self.pos_emb_q = nn.Parameter(torch.randn(1, seq_len, heads, self.d_pos))
        self.pos_emb_k = nn.Parameter(torch.randn(1, seq_len, heads, self.d_pos)) 
        self.pos_emb_q.data.normal_(mean=0.0, std=0.05)
        self.pos_emb_k.data.normal_(mean=0.0, std=0.05)

    def extend_pos_emb(self, cur_seq_len_q:int, cur_seq_len_k:int, device: torch.device = torch.device('cuda')):
        if cur_seq_len_q > self.pos_emb_q.shape[1]:
            if self.training:
                diff = cur_seq_len_q - self.pos_emb_q.shape[1]
                pos_ext = torch.randn(1, diff, self.heads, self.d_pos)
                pos_ext.data.normal_(mean=0.0, std=self.pos_emb_q.std().item())
                self.pos_emb_q = nn.Parameter(torch.cat([self.pos_emb_q, pos_ext.to(device)], dim=1))
            else:
                warnings.warn(f'cur_seq_len_q ({cur_seq_len_q}) > self.pos_emb_q.shape[1] ({self.pos_emb_q.shape[1]}), this will cause an error at inference time')
        if cur_seq_len_k > self.pos_emb_k.shape[1]:
            if self.training:
                diff = cur_seq_len_k - self.pos_emb_k.shape[1]
                pos_ext = torch.randn(1, diff, self.heads, self.d_pos)
                pos_ext.data.normal_(mean=0.0, std=self.pos_emb_k.std().item())
                self.pos_emb_k = nn.Parameter(torch.cat([self.pos_emb_k, pos_ext.to(device)], dim=1))
            else:
                warnings.warn(f'cur_seq_len_k ({cur_seq_len_k}) > self.pos_emb_k.shape[1] ({self.pos_emb_k.shape[1]}), this will cause an error at inference time')

    def forward(self, q, k):
        # b, n, h, d = q.shape
        cur_seq_len_q, cur_seq_len_k = q.shape[1], k.shape[1]
        self.extend_pos_emb(cur_seq_len_q, cur_seq_len_k, device = q.device)
      
        return q + self.pos_emb_q[:, :cur_seq_len_q, :, :].clone(), k + self.pos_emb_k[:, :cur_seq_len_k, :, :].clone()


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

        self.n_feats, self.head_dim, self.n_heads = n_feats, head_dim, n_heads
   
        self.activation = nn.Softmax(dim=-1)

        self.dropout_p = dropout
        self.causal = False
        # softmax_scale is set to None but will default to 1/sqrt(d_k) in FlashAttention
        self.flash_attn_c_fn = FlashCrossAttention(softmax_scale = None, attention_dropout = dropout, causal = False)

        self.qkv_proj = nn.Linear(n_feats, 3 * n_heads * head_dim, bias=bias)
        self.q_proj = nn.Linear(n_feats, n_heads * head_dim, bias=bias)
        self.kv_proj = nn.Linear(n_feats, 2 * n_heads * head_dim, bias=bias)

        #self.kv = lambda x: rearrange(self.kv_proj(x), "b n (h d kv) -> b n kv h d", kv=2, h=n_heads, d=head_dim)
        self.kv = lambda x: rearrange(self.kv_proj(x), "b n (h d kv) -> kv b n h d", kv=2, h=n_heads, d=head_dim)
        self.q = lambda x: rearrange(self.q_proj(x), "b n (h d) -> b n h d", h=n_heads, d=head_dim)

        self.out_proj = nn.Linear(n_heads * head_dim, n_feats, bias=bias)


        
    def forward(self, xq, xkv, kv_mask = None):
        H, D = self.n_heads, self.head_dim

        flash_attn = True # not implemented otherwise !

        q = self.q(xq)
        k, v = self.kv(xkv)
        kv = torch.stack([k, v], dim=2)

        ### Flash attention stuff 
        if xq.device.type == 'cuda' and flash_attn:
            q, kv = q.contiguous(), kv.contiguous()
            if q.dtype == torch.float32:
                q, kv = q.half(), kv.half()

            out = self.flash_attn_c_fn(q, kv)
            if kv_mask is None:
                out = self.flash_attn_c_fn(q, kv)
            else:
                q_attn_mask = torch.ones((q.shape[0], q.shape[1]), device=q.device).bool()
                kv_attn_mask = kv_mask

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
            out = out.to(xq.dtype) 
            out = rearrange(out, "b n h d -> b n (h d)")
        else:
            raise NotImplementedError # just need to implemenent masking for cross attention sdp
            k, v = rearrange(kv, "b n kv h d -> kv b h n d", kv=2).contiguous()
            q = q.transpose(1, 2).contiguous()
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=False)
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

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = LearnableFourierPosEnc(d_model, hidden_dim=64)
        
        self.embed_dropout = nn.Dropout(dropout_attn)

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
                        causal = True,
                        dropout = dropout_attn,),
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
                    ), 
                    norm = default_norm
                ),
                PreNorm(
                    d_model = d_model, 
                    fn = ConformerFeedForward(d_model, bias1 = bias_in_ff, bias2 = bias_in_ff),
                    norm = default_norm,
                )
            ]))
            

        self.out_proj = nn.Sequential(
            default_norm(d_model) if decoder_norm else nn.Identity(),
            nn.Linear(d_model, vocab_size)
        )

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
            tokens: torch.Tensor, 
            a_hidden: torch.Tensor, 
            a_lengths: torch.Tensor,
        ):
        '''
        tokens: (batch, seq_len) - target text sequence
        a_hidden: (batch, seq_len, dim) - encoder output
        pos_enc_module: positional encoding module - instance of PosEnc
        '''
        x = self.pos_enc(self.embed(tokens))
        a_hidden = self.acoustic_norm(a_hidden)
        
        lengths = torch.LongTensor([x.shape[1]] * x.shape[0]).to(x.device)
        
        if a_lengths.max() == a_lengths.min(): a_mask = None # if all the same length don't bother with the mask
        else: a_mask = ~(torch.arange(a_hidden.shape[1], device=a_hidden.device).expand(a_hidden.size(0), a_hidden.shape[1]) >= a_lengths.unsqueeze(1))

        for lth, (self_attn, cross_attn, ff_out) in enumerate(self.layers):
            x = self_attn(x, length=lengths)[0] + x
            x = cross_attn(x, xkv = a_hidden, kv_mask=a_mask) + x
            x = ff_out(x) + x

        return self.out_proj(x)



if __name__ == '__main__':
    model = EncDecSconformer() # default model
    model.print_total_params()

    vocab_size = 4096
    audio_seq = torch.randn(2, 80, 100, device='cuda')
    text = torch.randint(0, vocab_size, (2, 10), device='cuda')
    model.to('cuda')
    print(text.shape, audio_seq.shape)

    out = model(audio_seq, text)
    print([o.shape for o in out.values()])

    model.calc_loss(
        audio_signal = audio_seq,
        text_sequence = text,
        a_length = torch.tensor([100, 100], device='cuda'),
        t_lengths = torch.tensor([5, 10], device='cuda')
    )



                    