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
import random
from lcasr.components.batchrenorm import BatchRenorm1d
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

from matplotlib import pyplot as plt
# TODO: 
# -. remove caching stuff as it is not used anymore


class MelDiscriminator1D(nn.Module):
    def __init__(self, mel_bins=80):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(mel_bins, 128, kernel_size=15, stride=1, padding=7),
            BatchRenorm1d(128),
            nn.LeakyReLU(0.2),

            nn.Conv1d(128, 128, kernel_size=41, stride=4, groups=4, padding=20),
            BatchRenorm1d(128),
            nn.LeakyReLU(0.2),

            nn.Conv1d(128, 256, kernel_size=41, stride=4, groups=4, padding=20),
            BatchRenorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Conv1d(256, 512, kernel_size=41, stride=4, groups=4, padding=20),
            BatchRenorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Conv1d(512, 1024, kernel_size=5, stride=1, padding=2),
            BatchRenorm1d(1024),
            nn.LeakyReLU(0.2),

            nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)  # Output: [B, 1, T']

class SCConformerTest(BaseModel): 
    def __init__(
        self,
        feat_in = 80,
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
        use_rotary = False,
        rotary_interpolation_factor = 1.0, # https://arxiv.org/abs//2306.15595 Extending Context Window of Large Language Models via Positional Interpolation
        learned_rotary = False,
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
    
        self.calc_loss_includes_backward = True
        self.legasee_double_norm = legasee_double_norm

        self.checkpoint_subsampling = kwargs.get('checkpoint_subsampling', False) # whether to perform activation checkpointing on subsampling layers

        accepted_norms = ['rms_norm', 'layer_norm']
        accepted_subsampling_acts = ['silu', 'relu', 'gelu', 'none']
        assert default_norm in accepted_norms, f'default_norm must be one of {accepted_norms} (got {default_norm})'
        default_norm = RMSNorm if default_norm == 'rms_norm' else LayerNorm


        self.flash_attn = kwargs.get('flash_attn', True)
      
        self.checkpoint_every_n_layers = checkpoint_every_n_layers

        self.dropout_ff = dropout_ff
        self.dropout_conv = dropout_conv
        self.dropout_attn = dropout_attn


        self.whitelist_weight_decay_modules = (nn.LayerNorm, RMSNorm, LayerNorm, convolution.BatchRenorm1d, nn.GroupNorm) # don't decay
        self.blacklist_weight_decay_modules = (nn.Linear, ConformerFeedForward, nn.Conv1d, nn.Conv2d, RotaryPositionalEmbedding)

        self.use_rotary = use_rotary
        self.rotary_pos_emb = None
        if self.use_rotary:
            self.rotary_pos_emb = RotaryPositionalEmbedding(
                dim = head_dim,
                base = kwargs.get('rotary_base_freq', 1500000),
                learned_freq = learned_rotary,
                rotary_interpolation_factor = rotary_interpolation_factor
            )
     
        self.layers = nn.ModuleList()

        self.in_proj = nn.Linear(feat_in, d_model)
        self.out_proj = nn.Linear(d_model, feat_in)

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

        self.discriminator = MelDiscriminator1D(mel_bins=feat_in)
        
    def forward(
            self, 
            audio_signal = None,
            length = None, 
            *args,
            **kwargs
        ):
        '''
        audio_signal: (batch_size, time, feat)
        length: (batch_size,)
        cached_kvs: (kv i.e 2, batch_size, layers, heads, time, head_dim)
        '''

        max_audio_length: int = audio_signal.size(-1)

        if length is None: length = torch.tensor([max_audio_length] * audio_signal.size(0), device=audio_signal.device)
            
        audio_signal = torch.transpose(audio_signal, 1, 2)
        audio_signal = self.in_proj(audio_signal)

        ## create masks
        
        mask = torch.arange(max_audio_length, device=audio_signal.device).expand(audio_signal.size(0), max_audio_length) >= length.unsqueeze(1)
    
        rotary_emb_fn = None
   
        full_kv_lengths = length 
        if self.use_rotary:
            max_seq_len = full_kv_lengths.max()
            q_offset = 0 

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

        audio_signal = self.out_proj(audio_signal)

        if self.training and self.rotary_pos_emb is not None:
            self.rotary_pos_emb.reset_if_needed()


        return {'out': audio_signal, 'length': length}
    
    def calc_loss(self, audio_signal, a_lengths, *args, **kwargs):
        B,C,T= audio_signal.shape
        length = a_lengths
        if length is None: length = torch.tensor([T] * B, device=audio_signal.device)
        start_mask_a = torch.LongTensor([0]*B).to(audio_signal.device)
     
        end_mask_a = (start_mask_a + length // 4)
        end_mask_b = (length)
        start_mask_b = (end_mask_b - length // 4)

        start_mask_a = start_mask_a.view(B, 1, 1)
        end_mask_a = end_mask_a.view(B, 1, 1)
        start_mask_b = start_mask_b.view(B, 1, 1)
        end_mask_b = end_mask_b.view(B, 1, 1)

        target = audio_signal.clone()

        time = torch.arange(T).to(audio_signal.device).unsqueeze(0).unsqueeze(0).expand(B, 1, T)
        mask_a = (time >= start_mask_a) & (time < end_mask_a)
        mask_b = (time >= start_mask_b) & (time < end_mask_b)
        mask = mask_a | mask_b
        audio_signal = audio_signal.masked_fill(mask, 0)
        prediction = self.forward(audio_signal, length)['out'].transpose(1, 2)

        if kwargs.get("wandb", False) and random.random() < 0.1:
            wandb = kwargs.get("wandb")
            fig = plt.imshow(prediction[0].cpu().detach().numpy(), aspect='auto', vmin=-1, vmax=1, cmap='inferno')
            wandb.log({"prediction": fig}, commit=False)

        optimizer = kwargs.get("optimizer")

        loss_recon = F.l1_loss(prediction, target, reduction='mean')

        loss_G = self.discriminator(prediction).mean(dim=(-1,-2)) 
        loss_G = ((loss_G - 1) ** 2).mean()
        loss = loss_G 

        G_params = list(self.layers.parameters()) + list(self.in_proj.parameters()) + list(self.out_proj.parameters())
        torch.nn.utils.clip_grad_norm_(G_params, 0.1)
        optimizer.zero_grad()
        loss.backward(inputs=G_params)
        optimizer.step()
        optimizer.zero_grad()

        all_samples = torch.cat([prediction.detach(), target.detach()], dim=0)
        disc_out = self.discriminator(all_samples).mean(dim=(-1,-2))
        disc_out_fake = disc_out[:B]
        disc_out_real = disc_out[B:B*2]
        loss_D_real = (disc_out_real - 1) ** 2
        loss_D_fake = disc_out_fake ** 2
        loss_D = (loss_D_real.mean() + loss_D_fake.mean()) / 2
        loss_D *= 0.1

        disc_params = list(self.discriminator.parameters())
        torch.nn.utils.clip_grad_norm_(disc_params, 0.1)
        loss_D.backward(inputs=disc_params)
        optimizer.step()

        display_losses = {
            'loss_D': loss_D.item(),
            'loss_G': loss_G.item(),
            'loss_recon': loss_recon.item(),
            'loss': loss.item(),
        }
        
        return {'loss': loss, 'display_losses': display_losses}



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
    