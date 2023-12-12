import torch, torch.nn as nn, torch.nn.functional as F
import apex
from torch.utils.checkpoint import checkpoint # # gradient/activation checkpointing
from einops import rearrange
from functools import partial
from lcasr.components import fused_dense, subsampling, decoder, wrappers
from lcasr.components.rotary_emb import RotaryPositionalEmbedding, apply_rotary
from lcasr.utils.helpers import exists
from lcasr.components.bidirectional_mamba import MambaBlock
DEFAULT_NORM, RMSNorm, LayerNorm = apex.normalization.FusedRMSNorm, apex.normalization.FusedRMSNorm, apex.normalization.FusedLayerNorm
import math
ConvSubsampling, StackingSubsampling = subsampling.ConvSubsampling, subsampling.StackingSubsampling



# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class Mamba(nn.Module): 
    def __init__(
        self,
        vocab_size = 128,
        feat_in = 80,
        subsampling = 'dw_striding',
        subsampling_factor = 8,
        subsampling_conv_channels = 256,
        subsampling_act = 'silu',
        subsampling_norm_out = False,
        self_conditioning = True,
        n_layers = 6,
        d_model = 768,
        **kwargs
    ):
        super().__init__()
        
        self.feat_in = feat_in
        self.n_layers = n_layers
        self.d_model = d_model        
        self.self_conditioning = self_conditioning
        
        accepted_subsampling_acts = ['silu', 'relu', 'gelu', 'none']
        assert subsampling_act in accepted_subsampling_acts, f'subsampling_act must be one of {accepted_subsampling_acts} (got {subsampling_act})'
    
        if subsampling_act == 'silu':
            subsampling_act = nn.SiLU()
        elif subsampling_act == 'relu':
            subsampling_act = nn.ReLU()
        elif subsampling_act == 'gelu':
            subsampling_act = nn.GELU()
        elif subsampling_act == 'none':
            subsampling_act = nn.Identity()

        self.subsampling_mode = subsampling
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels if subsampling_conv_channels != -1 else d_model


        self.decoder = decoder.ASRLinearSCDecoder(
            d_model = d_model,
            vocab_size = vocab_size,
            norm = True
        )

        subsampling_args = {'subsampling_factor': self.subsampling_factor, 'feat_in': self.feat_in, 'feat_out': self.d_model, 'norm_out': subsampling_norm_out,}
        self.subsampling = \
            ConvSubsampling(subsampling = self.subsampling_mode, conv_channels = self.subsampling_conv_channels, activation = subsampling_act, **subsampling_args) \
                if subsampling != 'stacking' else \
                     StackingSubsampling(norm = True if not subsampling_norm_out else False, default_norm = RMSNorm, **subsampling_args)

        
        self.layers = nn.ModuleList()


        for i in range(n_layers):
            l = MambaBlock(
                dim = d_model,
                norm_cls = RMSNorm,
            )
            self.layers.append(l)

        self.apply(partial(_init_weights, n_layer=n_layers))

    @staticmethod
    def create_custom_forward(module): # for activation checkpointing allow passing dictionary as the argument to the module
        def custom_forward(*args, **kwargs):
            return module(*args, **kwargs)
        return custom_forward

    def forward(
            self, 
            audio_signal, 
            length = None,
            return_logits = False,
        ):
        '''
        audio_signal: (batch_size, time, feat)
        length: (batch_size,)
        cached_kvs: (kv i.e 2, batch_size, layers, heads, time, head_dim)
        '''
        return self.forward_for_export(audio_signal=audio_signal, decoder=self.decoder, length=length, return_logits=return_logits)



    def forward_for_export(self, audio_signal, decoder, length = None, cached_kvs = None, cached_kv_lengths = None, return_logits = False):
        max_audio_length: int = audio_signal.size(-1)

        if cached_kvs is not None:
            assert cached_kv_lengths.max() == cached_kvs.shape[1], 'cached kvs must all be the same length'

        if length is None:
            length = torch.tensor([max_audio_length] * audio_signal.size(0), device=audio_signal.device)
            
        audio_signal = torch.transpose(audio_signal, 1, 2)
        audio_signal, length = self.subsampling(audio_signal, lengths = length)

        max_audio_length = audio_signal.size(1)
        ## create masks
        
        mask = torch.arange(max_audio_length, device=audio_signal.device).expand(audio_signal.size(0), max_audio_length) >= length.unsqueeze(1)
    
        rotary_emb_fn = None


        length_to_pass, mask = None, None

        if length.max() != length.min():
            length_to_pass = length
            mask = ~(torch.arange(length.max(), device=length.device) < length.unsqueeze(1)) 
    



        for lth, layer in enumerate(self.layers):
            audio_signal = layer(
                hidden_states = audio_signal,
                lengths = length_to_pass,
                mask = mask,
            )
            
            if lth != len(self.layers) - 1 and self.self_conditioning:
                iterim_post = torch.nn.functional.softmax(decoder(x=audio_signal, logits=True), dim=-1)
                audio_signal = decoder.integrate_projections(audio_signal, decoder.project_back(iterim_post))        

        
        audio_signal = decoder.norm(audio_signal) 
        final_posts = decoder(x = audio_signal, logits = return_logits) # having decoder.norm should have been removed is sortof a bug but probably doesn't matter


        return {
            'final_posteriors': final_posts,
            'length': length,
        }

    def print_total_params(self, only_trainable = False):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad) if only_trainable else sum(p.numel() for p in self.parameters())
        pstr = 'Total trainable params: ' if only_trainable else 'Total params: '
        print(f'{pstr}: ', total/1e6, 'M')
        return total



if __name__ == '__main__':
    model = Mamba(n_layers=7, d_model=768, vocab_size=4096)
    model.print_total_params()
    x = torch.randn(4, 80, 8192, device='cuda')
    model = model.to(x.device)
    y = model(x)
    print(y['final_posteriors'].shape)

