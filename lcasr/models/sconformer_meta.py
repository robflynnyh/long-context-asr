import torch, torch.nn as nn, torch.nn.functional as F

from torch.utils.checkpoint import checkpoint # # gradient/activation checkpointing
from einops import rearrange, einsum
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

PreNorm, Scale = wrappers.PreNorm, wrappers.Scale
import random
from lcasr.components.attention import Attention
from lcasr.models.base import BaseModel
import warnings
from lcasr.components.batchrenorm import BatchRenorm1d
import os
from contextlib import nullcontext

try: from vector_quantize_pytorch import VectorQuantize
except: VectorQuantize = None

from torch.func import functional_call


class metadecoder(nn.Module):
    def __init__(self, d_model, classes, norm=nn.LayerNorm, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.norm = norm(d_model)

        #self.glu = nn.Sequential(*[ConformerFeedForward(d_model) for _ in range(kwargs.get('meta_glu_layers', 0))])
        self.ff = nn.Linear(d_model, classes, bias=False)
        self.encode = combiner(d_model)#nn.Sequential(nn.Linear(4096, d_model), nn.LayerNorm(d_model))

        #self.layer_wise_lr = torch.nn.Parameter(torch.ones(6)*1e-3)
      


    def forward(self, x):
        x = self.norm(x)
        x = self.ff(x)
    
        return x
    

class combiner(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.d_model = d_model
        # module to combine two representations
  
        self.ff1 = nn.Sequential(
            nn.Linear(4096, d_model),
            nn.LayerNorm(d_model),
        )
        self.ff2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.out = nn.Linear(d_model*2, d_model)

    def forward(self, x1, x2):
        x1 = self.ff1(x1)
        x2 = self.ff2(x2)
        x = torch.cat((x1, x2), dim=-1)
        x = self.out(x)
        return x


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

        assert VectorQuantize is not None, 'VectorQuantize module must be installed to use this model (pip install vector_quantize_pytorch)'
        
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
        #self.combiner = combiner(d_model)

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


        codebook_classes = kwargs.get('codebook_classes', 16384)
        self.meta_decoder = metadecoder(d_model, classes=codebook_classes, **kwargs)

        self.grad_vq = VectorQuantize(
            dim = d_model,
            kmeans_init = True,
            decay = kwargs.get('vq_decay', 0.98),
            codebook_size = codebook_classes,
            threshold_ema_dead_code = 0.0,
            commitment_weight = 0.0,
        )

        self.meta_layers = nn.ModuleList()
        for i in range(kwargs.get('n_meta_layers', 1)):
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
            self.meta_layers.append(l)

        #PreNorm(d_model, ConformerFeedForward(d_model), norm = default_norm)
        for param in self.parameters():
            param.requires_grad = False
            
        for param in self.meta_decoder.parameters():
            param.requires_grad = True

        for param in self.meta_layers.parameters():
            param.requires_grad = True

        

        #self.emas = [EMAGradModule(ema_decay=0.99, init_val=p.data) for p in self.layers[0].parameters()]



    
    def main_layers(self, audio_signal, att_mask, length, pad_mask, rotary_emb_fn, params=None):
        
        for lth, layer in enumerate(self.layers):  
            
            if self.checkpoint_every_n_layers > 0 and lth % self.checkpoint_every_n_layers == 0:
                audio_signal = checkpoint(
                    self.create_custom_forward(functional_call),
                    layer,
                    dict(layer.named_parameters()) if params is None else params[lth],
                    (audio_signal, # x
                    att_mask, # att_mask
                    length,
                    pad_mask, # pad_mask
                    self.flash_attn,
                    rotary_emb_fn),
                    use_reentrant = False
                )
            else:
                audio_signal = functional_call(
                    layer,
                    dict(layer.named_parameters()) if params is None else params[lth],
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
            # for param in self.layers.parameters():
            #     param.requires_grad = True
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

        audio_signal.requires_grad = True
        iterations = 1
        if not self.training:
            audio_signal.requires_grad = True 
            iterations = 1

        was_training = self.training
        checkpoint_val = self.checkpoint_every_n_layers 
        if was_training: self.checkpoint_every_n_layers = 1

  
        new_params = None
        
        self.static_initial_signal = audio_signal.clone()
        self.initial_signal = audio_signal.clone()
        for i in range(iterations):
            #print(i)
            #no_grad = was_training and i < iterations - 1
         
            audio_signal = self.initial_signal.clone().detach().requires_grad_(True)
            
            #with torch.no_grad() if was_training else nullcontext():
            audio_signal = self.main_layers(audio_signal, att_mask, length, pad_mask, rotary_emb_fn, params=new_params)

            self.reprs = audio_signal
            self.reprs.retain_grad()
            audio_signal = self.reprs

            logits = self.decoder(x = decoder.norm(audio_signal) if self.legasee_double_norm else audio_signal, logits = True)

            # if was_training:
            #     logits = logits.detach()
            #     logits.requires_grad = True
            # if was_training: audio_signal = audio_signal.detach()
            # if was_training: audio_signal.requires_grad = True
    

            
            # if self.training:
            #     scale = torch.FloatTensor(logits.shape[0], 1, 1).uniform_(0.01, 2.5)
            #     scale = scale.to(logits.device)
            #     logits = logits * scale # 

            probs = logits.softmax(dim=-1)

            self.original_probs = probs.log()

            audio_signal = self.meta_decoder.encode(logits, self.static_initial_signal)

            for lth, layer in enumerate(self.meta_layers):
                if (self.checkpoint_every_n_layers > 0 and lth % self.checkpoint_every_n_layers == 0) or not self.training:
                    audio_signal = checkpoint(
                        
                        self.create_custom_forward(layer), 
                        audio_signal, # x
                        att_mask, # att_mask
                        length,
                        pad_mask, # pad_mask
                        self.flash_attn,
                        rotary_emb_fn,
                        use_reentrant = False
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
            
            
            meta_pred = self.meta_decoder(audio_signal)            
            self.grad_pred = meta_pred
            #print(self.meta_decoder.layer_wise_lr)

            # if self.training and i < iterations - 1:
            #     vals, indices = meta_pred.softmax(-1).max(dim=-1, keepdim=True)
            #     out = torch.zeros_like(meta_pred).scatter_(-1, indices, 1.0)
            #     out = out * vals
            #     grads = einsum(out, self.grad_vq.codebook, 'b n c, c d -> b n d')
            #     params = dict(self.layers.named_parameters())
            #     params_tensors, params_names = list(params.values()), list(params.keys())
            #     weight_grad = torch.autograd.grad(outputs = self.reprs, inputs = params_tensors, grad_outputs = grads, retain_graph=True)
                
            #     new_params = [{} for _ in range(len(self.layers))]
            #     for p, g, n in zip(params_tensors, weight_grad, params_names):
            #         layer_idx = int(n.split('.')[0])
            #         name = ".".join(n.split('.')[1:])
            #         new_params[layer_idx][name] = p.data - g * self.meta_decoder.layer_wise_lr[layer_idx]

                    
                

            # if not was_training:
            #     vals, indices = meta_pred.softmax(-1).max(dim=-1, keepdim=True)
            #     # print(indices)
            #     # print((vals > 0.6).sum() / vals.size(-2))
            #     # print((vals > 0.6).shape, indices.shape)
            #     out = torch.zeros_like(meta_pred).scatter_(-1, indices, 1.0)
            #     #print( (vals < 0.2).sum() / vals.size(-2))
            #     #out = out * (vals < 0.2).float()                    
            #     out = out * vals
            #     #print (out.max()), print (out.shape), print (self.grad_vq.codebook.shape)
            #     grads = einsum(out, self.grad_vq.codebook, 'b n c, c d -> b n d')
            #     print (vals.mean())

            #     params = [p for p in self.layers[0].parameters() if p.requires_grad]
            #     #print(self.reprs)
            #     weight_grad = torch.autograd.grad(outputs = self.reprs, inputs = params, grad_outputs = grads, retain_graph=True)

            #     for p, g in zip(params, weight_grad):
            #         p.data = p.data - g * 1e-3


            # if i < iterations - 1 and not was_training:
            #     grad_pred = meta_pred
            #     param_inputs = [self.initial_signal]
            #     all_param_grads = torch.autograd.grad(outputs=logits, inputs=param_inputs, grad_outputs=meta_pred, retain_graph=False)
            #     initial_signal_grad = all_param_grads[0]
            #     print(initial_signal_grad)
            #     self.initial_signal = self.initial_signal - initial_signal_grad * 400 #* 100
            #     self.initial_signal = self.initial_signal.detach()
            #     self.initial_signal.requires_grad = True

            # if i == iterations - 1 and not was_training:
            #     params = [p for p in self.layers[0].parameters() if p.requires_grad]
                
            #     weight_grad = torch.autograd.grad(outputs=logits, inputs=params, grad_outputs=meta_pred, retain_graph=False)
            #     #print(weight_grad)
            #     for p, g in zip(params, weight_grad):
            #         p.data = p.data - g * 1e-3

        # if was_training: audio_signal = self.reprs - meta_pred
        # else: audio_signal = self.reprs   
        final_posts = (probs.log())# - meta_pred.detach() * self.meta_decoder.i_weight.abs()).log_softmax(dim=-1)
        #final_posts = decoder(x = decoder.norm(audio_signal) if self.legasee_double_norm else audio_signal, logits = True)

        # print entropy of the final posteriors
        #posts = final_posts.softmax(dim=-1)
       # entropy = -(posts * posts.log()).sum(dim=-1).mean()
        #print(entropy)

        #final_posts = decoder(x = decoder.norm(audio_signal) if self.legasee_double_norm else audio_signal, logits = return_logits)
        
        self.checkpoint_every_n_layers = checkpoint_val

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
    