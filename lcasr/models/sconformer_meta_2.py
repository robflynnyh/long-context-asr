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
import random
from lcasr.components.attention import Attention
from lcasr.models.base import BaseModel
import warnings
from lcasr.components.batchrenorm import BatchRenorm1d
import os
from contextlib import nullcontext

class metadecoder(nn.Module):
    def __init__(self, d_model, norm=nn.LayerNorm, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.norm = norm(d_model)

        self.glu = nn.Sequential(*[ConformerFeedForward(d_model) for _ in range(kwargs.get('meta_glu_layers', 0))])
        self.ff = nn.Linear(d_model, 1) # ConformerFeedForward(d_model)

    def forward(self, x):
        x = self.norm(x)
        x = self.glu(x)
        return self.ff(x)


class EMAGradModule(nn.Module):
    def __init__(self, ema_decay=0.99, init_val=None):
        super(EMAGradModule, self).__init__()
        self.ema_decay = ema_decay
        self.current_step = 0
        self.current_val = init_val

    def forward(self, x):
        self.current_step += 1
        if self.current_val is None:
            self.current_val = x
        else:
            self.current_val = self.current_val.to(x.device)
            self.current_val = self.ema_decay * self.current_val + (1 - self.ema_decay) * x
        return self.current_val

class Combine(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model*2, d_model)
        self.ff2 = nn.Linear(d_model*2, d_model)
        self.ff3 = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()

    def forward(self, x1, x2):
        x = torch.cat([self.norm1(x1), self.norm2(x2)], dim=-1)
        x = self.ff1(x) * self.act(self.ff2(x))
        x = self.ff3(x)
        return x


class SCConformerMeta(BaseModel): 
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
        load_pretrained_from=None,
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



        self.inference_iterations = kwargs.get('inference_iterations',10)
        self.inference_lr = kwargs.get('inference_lr', 0.05)

        self.legasee_double_norm = legasee_double_norm

        self.checkpoint_subsampling = kwargs.get('checkpoint_subsampling', False) # whether to perform activation checkpointing on subsampling layers

        accepted_norms = ['rms_norm', 'layer_norm']
        accepted_subsampling_acts = ['silu', 'relu', 'gelu', 'none']
        assert subsampling_act in accepted_subsampling_acts, f'subsampling_act must be one of {accepted_subsampling_acts} (got {subsampling_act})'
        assert default_norm in accepted_norms, f'default_norm must be one of {accepted_norms} (got {default_norm})'
        default_norm = nn.LayerNorm #RMSNorm if default_norm == 'rms_norm' else LayerNorm

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

        # layer_delta = torch.zeros(n_layers, 1, 1, d_model)
        # # register buffer for layer delta
        # self.register_buffer('layer_delta', layer_delta)

        subsampling_args = {'subsampling_factor': self.subsampling_factor, 'feat_in': self.feat_in, 'feat_out': self.d_model, 'norm_out': subsampling_norm_out,}
        self.subsampling = \
            ConvSubsampling(subsampling = self.subsampling_mode, conv_channels = self.subsampling_conv_channels, activation = subsampling_act, **subsampling_args) \
                if subsampling != 'stacking' else \
                     StackingSubsampling(norm = True if not subsampling_norm_out else False, default_norm = default_norm, **subsampling_args)

        
        self.layers = nn.ModuleList()
        self.reprs = None
        self.grad_preds = None
        self.output_signal = None

        self.meta_mode = kwargs.get('meta_mode', 1)

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

        if exists(load_pretrained_from):
            if os.path.exists(load_pretrained_from):
                # load from path using torch.load
                checkpoint = torch.load(load_pretrained_from, map_location='cpu')
                
                self.load_state_dict(checkpoint['model'])
                print(f"Loaded model from {load_pretrained_from} !")

        self.meta_decoder = metadecoder(d_model, **kwargs)
        self.combine = Combine(1)

        self.embedding = nn.Embedding(vocab_size + 1, d_model)
        self.output_pred = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.normal_(self.output_pred, mean=0, std=0.1)

        self.meta_layers = nn.ModuleList()
        for i in range(kwargs.get('n_meta_layers', 2)):
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
                transformer = True,
                conv_expansion_factor = conv_expansion_factor,
                **kwargs
            )
            self.meta_layers.append(l)

        #PreNorm(d_model, ConformerFeedForward(d_model), norm = default_norm)
        for param in self.parameters():
            param.requires_grad = False
            
        for param in self.meta_decoder.parameters():
            param.requires_grad = True

        for param in self.meta_layers.parameters():
            param.requires_grad = True

        for param in self.combine.parameters():
            param.requires_grad = True

        #self.emas = [EMAGradModule(ema_decay=0.99, init_val=p.data) for p in self.layers[0].parameters()]


    
    def main_layers(self, audio_signal, att_mask, length, pad_mask, rotary_emb_fn):
        for lth, layer in enumerate(self.layers):

            if self.checkpoint_every_n_layers > 0 and lth % self.checkpoint_every_n_layers == 0:
                audio_signal = checkpoint(
                    self.create_custom_forward(layer), 
                    audio_signal, # x
                    att_mask, # att_mask
                    length,
                    pad_mask, # pad_mask
                    self.flash_attn,
                    rotary_emb_fn
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
                iterim_post = torch.nn.functional.softmax(self.decoder(x=audio_signal, logits=True), dim=-1)
                audio_signal = self.decoder.integrate_projections(audio_signal, self.decoder.project_back(iterim_post))       

        return audio_signal

    @torch.set_grad_enabled(True)
    def forward(
            self, 
            audio_signal,
            length = None, 
            cached_kvs = None, 
            cached_kv_lengths = None, 
            return_logits = False,
        ):
        '''
        audio_signal: (batch_size, time, feat)
        length: (batch_size,)
        cached_kvs: (kv i.e 2, batch_size, layers, heads, time, head_dim)
        '''
        audio_signal.requires_grad = True

             # freeze all layers
        if self.training:
            for param in self.parameters():
                param.requires_grad = False
            for param in self.meta_decoder.parameters():
                param.requires_grad = True
            for param in self.meta_layers.parameters():
                param.requires_grad = True
            for param in self.combine.parameters():
                param.requires_grad = True
        else:
            for param in self.parameters():
                param.requires_grad = True


        decoder = self.decoder
        max_audio_length: int = audio_signal.size(-1)

        if cached_kvs is not None:
            assert cached_kv_lengths.max() == cached_kvs.shape[1], 'cached kvs must all be the same length'

        if length is None:
            length = torch.tensor([max_audio_length] * audio_signal.size(0), device=audio_signal.device)
            
        audio_signal = torch.transpose(audio_signal, 1, 2)

        with torch.no_grad():
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
                qmask = torch.arange(max_audio_length, device=audio_signal.device).expand(audio_signal.size(0), max_audio_length) >= length.unsqueeze(1) 
                kmask = ~full_kv_mask
                att_mask = ~(rearrange(qmask, 'b n -> b () n ()') * rearrange(kmask, 'b n -> b () () n'))
                att_mask = att_mask.to(audio_signal.dtype) * -torch.finfo(audio_signal.dtype).max

        pad_mask = mask 
    
        audio_signal = self.fourier_pos_enc(audio_signal)
        self.reprs = None
        self.grad_preds = None
        self.output_signal = None

        was_training = self.training

        iterations = 1 #random.randint(1, 3)
        if not self.training:
            audio_signal.requires_grad = True 
            iterations = 1

        
        self.constant_initial_signal = audio_signal.clone()
        self.initial_signal = audio_signal.clone()
        for i in range(iterations):
            audio_signal = self.initial_signal.clone()

            with torch.no_grad() if self.training else nullcontext():
                audio_signal = self.main_layers(audio_signal, att_mask, length, pad_mask, rotary_emb_fn)
            
                
            self.reprs = audio_signal.clone()


            final_posts = decoder(x = decoder.norm(self.reprs) if self.legasee_double_norm else self.reprs, logits = True)
            final_probs_initial = final_posts.softmax(dim=-1)

            # get entropy of the final posteriors
            #entropy = -(final_probs * final_probs.log()).sum(dim=-1).mean()

            # convert softmax to embeddings
            final_probs_emb = torch.matmul(final_probs_initial, self.embedding.weight)
            
            
            initial_signal = self.initial_signal
            if was_training:
                final_probs_emb = final_probs_emb.detach()
                initial_signal = initial_signal.detach()

            audio_signal = torch.cat([initial_signal, final_probs_emb], dim=1)
            
            comb_length = torch.tensor([audio_signal.size(1)] * audio_signal.size(0), device=audio_signal.device)
            if self.use_rotary:
                max_seq_len = comb_length.max()
                q_offset = 0 
                cos, sin = self.rotary_pos_emb(max_seq_len, audio_signal.device)
                rotary_emb_fn_comb = apply_rotary(cos = cos, sin = sin, q_offset = q_offset, learned = self.rotary_pos_emb.learned_freq)
            

            for lth, layer in enumerate(self.meta_layers):
                if self.checkpoint_every_n_layers > 0 and lth % self.checkpoint_every_n_layers == 0:
                    audio_signal = checkpoint(
                        self.create_custom_forward(layer), 
                        audio_signal, # x
                        None, # att_mask
                        length,
                        None, # pad_mask
                        self.flash_attn,
                        rotary_emb_fn_comb
                    )
                else:
                    audio_signal = layer(
                        x = audio_signal, 
                        attn_mask = None, 
                        length = length,
                        pad_mask = None,
                        flash_attn = self.flash_attn,
                        rotary_emb_fn = rotary_emb_fn_comb
                    )
            
            final_pred = decoder(x = decoder.norm(audio_signal[:, audio_signal.size(1) // 2:]) if self.legasee_double_norm else audio_signal[:, audio_signal.size(1) // 2:], logits = True)
            #final_posts = final_pred.log_softmax(dim=-1)
            # meta_pred = self.meta_decoder(audio_signal[:, audio_signal.size(1) // 2:])
            # self.grad_preds = meta_pred
         
            # if not was_training and i < iterations - 1:
            #     input_grad = torch.autograd.grad(meta_pred, self.initial_signal)[0]
            #     self.initial_signal = self.initial_signal - input_grad * 100#* 0.01 #* 20000

            # if not was_training and 1==1 and i == iterations - 1:
            #     a = final_probs_initial
            #     c = final_pred.softmax(dim=-1)
          
            #     meta_grad = a - c
            #     params = [p for p in self.layers[0].parameters()]
            #     weight_grad = torch.autograd.grad(final_probs_initial, params, grad_outputs=meta_grad)
            #     for p, g in zip(params, weight_grad):
            #         p.data = p.data - g * 1e-3
                
                #if meta_pred.item() > 0.0:
                # params = [p for p in self.layers[0].parameters()]
                # # get all biases
                # # for p in self.layers.parameters():
                # #     if p.dim() == 1:
                # #         params.append(p) 
                # #params += [p for p in self.layers[1].parameters()]
                # #params += [p for p in self.layers[2].parameters()]
            
                # weight_grad = torch.autograd.grad(meta_pred, params)
                # # update weights
                # for p, g in zip(params, weight_grad):
                    # p.data = p.data - g * 1e-2
            
                
                #self.initial_signal = self.initial_signal - input_grad * 20000
            
        #print('---')
   
        
        final_posts = final_posts.log_softmax(dim=-1)

        if self.training and self.rotary_pos_emb is not None:
            self.rotary_pos_emb.reset_if_needed()

        return {'final_posteriors': final_posts, 'length': length}


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

            

    def forward(self, x, attn_mask, length, pad_mask, flash_attn = True, rotary_emb_fn = None):
        '''
        pad_mask: mask for padding used in conv layers
        attn_mask: attn_mask this should include the cached keys and values
        length: list of lengths of the input sequence
        cached_kv: kvs from previous block-reccurrent time step
        '''

        if not self.trasformer:
            x = self.do_ff(self.ff1(x)) + x

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




if __name__ == '__main__':
    # run test
    model = SCConformerMeta(vocab_size=4096, head_dim=256, n_heads=3, attention_window_size=128)
    audio = torch.randn(2, 80, 1000)
    lengths = torch.tensor([1000, 500])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    audio = audio.to(device)
    lengths = lengths.to(device)
    out = model(audio, length=lengths)
    print(out['final_posteriors'].shape)
    