import torch, torch.nn as nn, torch.nn.functional as F

from nemo.collections.asr.parts.submodules.dynamic_positions import DynamicPositionBiasXL
from nemo.collections.asr.parts.submodules.conformer_modules import (
    ConformerFeedForward,
    ConformerConvolution
)

from nemo.collections.asr.parts.utils.activations import ReLUSquared
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType, SpectrogramType, LogprobsType
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import adapter_mixins
from nemo.core.classes.module import NeuralModule
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling

from torch.utils.checkpoint import checkpoint # # gradient/activation checkpointing

from einops import rearrange, repeat
from torch import einsum


class SelfConditionedConformerXL(NeuralModule, Exportable):
    def __init__(
        self,
        feat_in = 80,
        n_layers = 12,
        d_model = 256,
        n_heads = 8,
        head_dim = 32,
        expansion_factor = 4,
        dropout_ff = 0.1,
        dropout_conv = 0.1,
        dropout_attn = 0.3,
        checkpoint_every_n_layers = 1,
        subsampling_factor = 4,
        conv_kernel_size = 31,
        commitment_loss = 0.25,
        codebook_size = 64,
        vq_cosine_sim = True,
        codebook_dim = 8,
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
        
        self.dropout_ff = dropout_ff
        self.dropout_conv = dropout_conv
        self.dropout_attn = dropout_attn

        self.position_bias = DynamicPositionBiasXL(heads = n_heads, **DynamicPositionBiasXL.fetch_module_kwargs(kwargs))

        from vector_quantize_pytorch import VectorQuantize
        self.VQ = VectorQuantize(
            dim = head_dim,
            codebook_size = codebook_size,
            use_cosine_sim = vq_cosine_sim,
            commitment_weight = commitment_loss,
            codebook_dim = codebook_dim
        )

        self.subsampling = ConvSubsampling(
            subsampling = 'striding',
            subsampling_factor = subsampling_factor,
            feat_in = feat_in,
            feat_out = d_model,
            conv_channels = d_model,
            activation = nn.ReLU()
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
            decoder, 
            length = None,
            cached_kvs = None,
            cached_kv_lengths = None
        ):
        return self.forward_for_export(audio_signal=audio_signal, decoder=decoder, length=length, cached_kvs=cached_kvs, cached_kv_lengths=cached_kv_lengths)

    def forward_for_export(self, audio_signal, decoder, length = None, cached_kvs = None, cached_kv_lengths = None):
        max_audio_length: int = audio_signal.size(-1)
        pos_emb = self.position_bias
        
        if length is None:
            length = audio_signal.new_full(
                audio_signal.size(0), max_audio_length, dtype=torch.int32, device=self.seq_range.device
            )
            
        audio_signal = torch.transpose(audio_signal, 1, 2)
        audio_signal, length = self.subsampling(audio_signal, lengths = length)
        max_audio_length = audio_signal.size(1)

        ## create masks
        mask = torch.arange(max_audio_length, device=audio_signal.device).expand(audio_signal.size(0), max_audio_length) >= length.unsqueeze(1)
        cached_kv_lengths = cached_kv_lengths if cached_kv_lengths is not None else None
        cached_kv_pad_mask = None
        if cached_kv_lengths != None:
            cached_kv_pad_mask = torch.arange(cached_kvs.shape[-2], device=audio_signal.device).expand(audio_signal.size(0), cached_kvs.shape[-2]) >= cached_kv_lengths.unsqueeze(1)
      
        full_kv_lengths = length + cached_kv_lengths if cached_kv_lengths is not None else length
        full_kv_mask = torch.arange(full_kv_lengths.max(), device=audio_signal.device).expand(audio_signal.size(0), full_kv_lengths.max()) >= full_kv_lengths.unsqueeze(1)
        qmask, kmask = ~mask, ~full_kv_mask
        att_mask = ~(rearrange(qmask, 'b n -> b () n ()') * rearrange(kmask, 'b n -> b () () n'))
        pad_mask = mask

        cached_kv_indices = None
        if cached_kvs is not None and cached_kv_lengths != None:
            cached_kv_indices = self.get_cache_indices(x_lens=length, cache_lens=cached_kv_lengths, cache_kv=cached_kvs, x=audio_signal)


        iterim_posteriors = []
        commit_losses = []
        kvs_to_cache = []
        for lth, layer in enumerate(self.layers):
            # cached kvs for the layer up
            layer_up_kvs = cached_kvs[:,:,lth + 1] if lth + 1 < self.n_layers and cached_kvs is not None else None
            current_layer_kvs = cached_kvs[:,:,lth] if cached_kvs is not None else None
            layer_up_kvs = current_layer_kvs if layer_up_kvs is None else layer_up_kvs
            cached_kvs_to_pass = None

            if current_layer_kvs is not None:
                layer_up_kvs, commit_loss = self.vector_quantize_cached_states(layer_up_kvs, pad_mask=cached_kv_pad_mask, device=audio_signal.device, dtype=audio_signal.dtype)
                commit_losses.append(commit_loss)
                cached_kvs_to_pass = torch.stack([current_layer_kvs, layer_up_kvs], dim=0)


            if self.checkpoint_every_n_layers > 0 and lth % self.checkpoint_every_n_layers == 0:
                audio_signal, kv_to_cache = checkpoint(
                    self.create_custom_forward(layer), 
                    audio_signal, # x
                    att_mask, # att_mask
                    pos_emb, # pos_emb
                    pad_mask, # pad_mask
                    cached_kvs_to_pass, # cached_kv
                    cached_kv_indices # cached_kv_indices
                )
            else:
                audio_signal, kv_to_cache = layer(
                    x = audio_signal, 
                    att_mask = att_mask, 
                    pos_emb = pos_emb, 
                    pad_mask = pad_mask,
                    cached_kv = cached_kvs_to_pass,
                    cached_kv_indices = cached_kv_indices
                )

            kvs_to_cache.append(kv_to_cache) # possibly detach and move to cpu ?    
            
            if lth != len(self.layers) - 1:
                iterim_logits = decoder(encoder_output=audio_signal.transpose(1, 2), logits=True)
                iterim_post = torch.nn.functional.softmax(iterim_logits, dim=-1)
                iterim_logposteriors = torch.log(iterim_post)
                iterim_posteriors.append(iterim_logposteriors)
                audio_signal = decoder.integrate_projections(audio_signal, decoder.project_back(iterim_post))        

        audio_signal = torch.transpose(audio_signal, 1, 2) # (batch, seq_len, d_model) -> (batch, d_model, seq_len) 

        # stack the posteriors along the first dimension (height, batch, seq_len, dim)
        iterim_posteriors = torch.stack(iterim_posteriors, dim=0)
        kvs_to_cache = torch.stack(kvs_to_cache, dim=0)
        kvs_to_cache = rearrange(kvs_to_cache, 'l kv b h n d -> kv b l h n d')
        
        commit_losses = torch.stack(commit_losses, dim=0).sum() if len(commit_losses) > 0 else 0
        
        return audio_signal, iterim_posteriors, kvs_to_cache, length, full_kv_lengths, {'commit_loss': commit_losses}

    def vector_quantize_cached_states(self, cached_kv, pad_mask, device, dtype):
        KV, B, H, N, D = cached_kv.shape
    
        cached_kv = cached_kv.to(device=device, dtype=dtype)
        cached_kv = rearrange(cached_kv, 'kv b h n d -> (kv b h) n d')
        pad_mask = repeat(pad_mask, 'b n -> (kv b h) n', kv=KV, b=B, h=H)
        vq_cached_kv, indices, commit_loss = self.VQ(cached_kv, mask=~pad_mask)
        
        vq_cached_kv = rearrange(vq_cached_kv, '(kv b h) n d -> kv b h n d', kv=KV, b=B, h=H, n=N, d=D)
        return vq_cached_kv, commit_loss
        
    @staticmethod
    def get_cache_indices(x_lens, cache_lens, cache_kv, x):  # replace with a vmap
        # used later w/ gather to remove padding when cache is concatenated with current input
        max_new_len = (x_lens + cache_lens).max()

        B, H, N, D = x.shape[0], 1, (x.shape[1] + cache_kv.shape[-2]), cache_kv.shape[-1]

        indices = []
        for i in range(B):
            cache_indices = torch.arange(cache_lens[i], device='cpu')
            total_length = cache_lens[i] + x_lens[i]
            diff_from_max_len = max_new_len - total_length
            x_indices = torch.arange(x_lens[i]+diff_from_max_len, device='cpu') + cache_kv.shape[-2]
            if diff_from_max_len > 0:
                x_indices[-diff_from_max_len:] = N 
            new_indices = torch.cat([cache_indices, x_indices])
            indices.append(new_indices)

        indices = torch.stack(indices, dim=0)
        
        indices = rearrange(indices, 'b n -> () () b () n ()').expand(2, 2,B,H,-1,D)
        return indices.to(x.device)

class PreNorm(nn.Module): # applies normalization before fn
    def __init__(self, d_model, fn):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

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
        expansion_factor,
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

        d_ff = d_model * expansion_factor

        self.conv = PreNorm(
            d_model = d_model, 
            fn = ConformerConvolution(
                d_model = d_model,
                kernel_size = conv_kernel_size,
                norm_type = 'batch_renorm'
            ),
        )

        self.do_conv = nn.Dropout(dropout_conv)

        ff_args = {'d_model': d_model, 'd_ff': d_ff, 'dropout': dropout_ff}
        self.ff1 = Scale(0.5, PreNorm(d_model = d_model, fn = ConformerFeedForward(**ff_args)))
        self.ff2 = Scale(0.5, PreNorm(d_model = d_model, fn = ConformerFeedForward(**ff_args)))
        self.do_ff = nn.Dropout(dropout_ff)

        self.attend = PreNorm(
            d_model = d_model, 
            fn = CosineAttention(
                n_feats = d_model,
                head_dim = head_dim,
                n_heads = n_heads,
                dropout = dropout_attn,
                bias = False,
                temperature = kwargs.get('temperature', 15.5),
                activation = 'softmax',
                layer_idx = layer_idx,
            )
        )

        self.do_attn_out = nn.Dropout(min(dropout_ff, 0.1)) # don't wan't this too large

        self.norm_out = nn.LayerNorm(d_model)

            

    def forward(self, x, attn_mask, pos_emb, pad_mask, cached_kv = None, cached_kv_indices = None):
        '''
        pad_mask: mask for padding used in conv layers
        attn_mask: attn_mask this should include the cached keys and values
        cached_kv: kvs from previous block-reccurrent time step
        cached_kv_indices: indices used to remove padding in between prev kvs and current kvs, 
            these only need to be calculated once hence they are passed to each layer
        '''

        x = self.do_ff(self.ff1(x)) + x

        attn_out, kv_to_cache = self.attend(
            x = x,
            pos_fn = pos_emb,
            attn_mask = attn_mask,
            cached_kv = cached_kv,
            cached_kv_indices = cached_kv_indices
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

class SpatialAttnDropout(nn.Module): # this is a inefficient way to do this, use triton?
    def __init__(self, p = 0.25, spatial_size = 5):
        super().__init__()
        self.p = p
        self.spatial_size = spatial_size
        self.drop = nn.Dropout(p = p)

    def forward(self, x): 
        if not self.training or self.p == 0.:
            return x
        if x.shape[-1] // 2 <= self.spatial_size: # normal dropout if sequence is too short
            return self.drop(x)
        B,H,I,J = x.shape
        dropmask = torch.ones((B,H,I,J//self.spatial_size), device = x.device, dtype = torch.half if x.device.type == 'cuda' else x.dtype)
        dropmask = self.drop(dropmask)
        dropmask = dropmask.repeat_interleave(self.spatial_size, dim = -1)
        # pad the end of the mask to match the length of the input (with dropmask.max() (bcos of scale up from dropout))
        dropmask = torch.nn.functional.pad(dropmask, (0, J - dropmask.shape[-1]), value=dropmask.max())
        return x * dropmask

class CosineAttention(nn.Module):
    def __init__(
        self,
        n_feats,
        head_dim,
        n_heads,
        dropout=0.1,
        bias=False,
        temperature=15.5,
        activation='softmax',
        **kwargs
    ):
        super().__init__()
        assert activation in ['relusq', 'softmax']
        self.shared_kv = kwargs.get('shared_kv', True)
        self.talking_heads = kwargs.get('talking_heads', 'pre')
        spatial_attn_dropout = kwargs.get('spatial_attn_dropout', False)
        self.layer_idx = kwargs.get('layer_idx', None)

        #self.history_vector = torch.nn.Parameter(torch.zeros(2, 1, 1, 1, head_dim), requires_grad=True)

        self.n_feats, self.head_dim, self.n_heads = n_feats, head_dim, n_heads
        self.dropout = nn.Dropout(dropout) if not spatial_attn_dropout else SpatialAttnDropout(p = dropout)
        
        if self.talking_heads:
            if self.talking_heads == 'pre' or self.talking_heads == 'both':
                self._head_proj = nn.Conv2d(n_heads, n_heads, (1, 1))
            if self.talking_heads == 'post' or self.talking_heads == 'both':
                self._head_proj_post = nn.Conv2d(n_heads, n_heads, (1, 1))
            

        self.temperature = torch.nn.Parameter(torch.tensor(temperature), requires_grad=True) if isinstance(temperature, float) else temperature

        self.activation = ReLUSquared() if activation == 'relusq' else nn.Softmax(dim=-1)

        if not self.shared_kv:
            self.qkv_proj = nn.Linear(n_feats, 3 * n_heads * head_dim, bias=bias)
            self.qkv = lambda x: rearrange(self.qkv_proj(x), "b n (h d qkv) -> qkv b h n d", qkv=3, h=n_heads, d=head_dim)
        else:
            self.q_proj, self.kv_proj = [nn.Linear(n_feats, el, bias=bias) for el in [n_heads * head_dim, 2 * head_dim]]
            map_q, map_kv = lambda q: rearrange(q, 'b n (h d) -> b h n d', h=n_heads), lambda kv: rearrange(kv, 'b n (kv d) -> kv b () n d', kv=2, d=head_dim)
            self.qkv = lambda x: (map_q(self.q_proj(x)), *map_kv(self.kv_proj(x)))

        self.out_proj = nn.Linear(n_heads * head_dim, n_feats, bias=bias)
    
    def head_proj(self, dots, mode='pre'):
        if mode == 'pre' and (self.talking_heads == 'pre' or self.talking_heads == 'both'):
            dots = self._head_proj(dots)
        if mode == 'post' and (self.talking_heads == 'post' or self.talking_heads == 'both'):
            dots = self._head_proj_post(dots)
        return dots      

    def attend(self, query, key, value, attn_mask, pos_fn):
        dots = einsum('bhid,bhjd->bhij', query, key) * self.temperature
        dots = self.head_proj(dots, mode='pre')

        pos = pos_fn(i=dots.shape[-2], j=dots.shape[-1], device=dots.device, dtype=dots.dtype)
        #import pickle as pkl
        #pkl.dump(pos.detach().to('cpu'), open(f'bias.pkl', 'wb'))
        dots += pos
        
        dots.masked_fill_(attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.activation(dots)
        attn = self.head_proj(attn, mode='post')
    
        '''if self.layer_idx == 0:
            print(attn.shape)
        
        if attn.shape[-1] != attn.shape[-2]:
            import pickle as pkl
            pkl.dump(attn.detach().to('cpu'), open(f'attn_{self.layer_idx}.pkl', 'wb'))
            if self.layer_idx == 11:
                exit()'''
        

        attn = self.dropout(attn)
        return einsum("bhij,bhjd->bhid", attn, value)

    
    def attatch_cache(self, kv, cached_kv, cached_kv_indices):
        kv = torch.stack(kv, dim=0)
        if cached_kv is None or cached_kv_indices is None:
            return kv, kv

        # zero vector for padding in the gather # kv is shape [2, B, H, N, D] so zero vector is [2, B, H, 1, D] # add extra dimension for kvs from different layers
        kv = kv.unsqueeze(0).expand(2, *kv.shape) # expand as cached_kv contatins current and above layer # these are coupled so gather only needs to be done once
        zero_vector = torch.zeros_like(kv[:, :, :, :, :1, :]) # used as the padding index in the gather
        #cached_kv[1] += self.history_vector.expand_as(cached_kv[1])
        new_kv = torch.cat([cached_kv, kv, zero_vector], dim=-2)
        new_kv = torch.gather(new_kv, dim=-2, index=cached_kv_indices)

        return new_kv

    def forward(self, x, pos_fn, attn_mask, cached_kv=None, cached_kv_indices=None):
        assert pos_fn is not None, 'pls provide a position function'
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim
        #print(x.shape, mask.shape)
       
        q, k, v = self.qkv(x)
        q, k = map(l2norm, (q, k))
        cur_l_kv, shifted_history_kv = self.attatch_cache([k, v], cached_kv, cached_kv_indices)
      
        k, v = shifted_history_kv

        out = self.attend(q, k, v, attn_mask, pos_fn)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        #print(cur_l_kv.shape)
        # permute cur_l_kv to mix up the batch dimension (test)
        #cur_l_kv = cur_l_kv[:, torch.randperm(cur_l_kv.shape[1]), :, :, :] + torch.randn_like(cur_l_kv)*100
        #print(cur_l_kv.shape)
        return out, cur_l_kv.half() 