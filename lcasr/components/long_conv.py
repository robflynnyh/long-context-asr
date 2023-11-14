# Taken from https://github.com/HazyResearch/safari/tree/main https://arxiv.org/abs/2302.06646
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import opt_einsum as oe
from einops import repeat
from functools import partial
from lcasr.utils.helpers import exists
import math


fftconv_funcs = None



optimized = True
if optimized:
    contract = oe.contract
else:
    contract = torch.einsum


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """ X: (batch, dim, lengths...) """
        if self.training:
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            return X
        return X

def get_initializer(name, activation=None):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        nonlinearity = 'linear'
    elif activation in ['relu', 'tanh', 'sigmoid']:
        nonlinearity = activation
    elif activation in ['gelu', 'swish', 'silu']:
        nonlinearity = 'relu' # Close to ReLU so approximate with ReLU's gain
    else:
        raise NotImplementedError(f"get_initializer: activation {activation} not supported")

    if name == 'uniform':
        initializer = partial(torch.nn.init.kaiming_uniform_, nonlinearity=nonlinearity)
    elif name == 'normal':
        initializer = partial(torch.nn.init.kaiming_normal_, nonlinearity=nonlinearity)
    elif name == 'xavier':
        initializer = torch.nn.init.xavier_normal_
    elif name == 'zero':
        initializer = partial(torch.nn.init.constant_, val=0)
    elif name == 'one':
        initializer = partial(torch.nn.init.constant_, val=1)
    else:
        raise NotImplementedError(f"get_initializer: initializer type {name} not supported")

    return initializer

def LinearActivation(
        d_input, d_output, bias=True,
        zero_bias_init=False,
        transposed=False,
        initializer=None,
        activation=None,
        activate=False, # Apply activation as part of this module
        weight_norm=False,
        **kwargs,
    ):
    """ Returns a linear nn.Module with control over axes order, initialization, and activation """

    # Construct core module
    # linear_cls = partial(nn.Conv1d, kernel_size=1) if transposed else nn.Linear
    linear_cls = TransposedLinear if transposed else nn.Linear
    if activation == 'glu': d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    # Initialize weight
    if initializer is not None:
        get_initializer(initializer, activation)(linear.weight)

    # Initialize bias
    if bias and zero_bias_init:
        nn.init.zeros_(linear.bias)

    # Weight norm
    if weight_norm:
        linear = nn.utils.weight_norm(linear)

    if activate and activation is not None:
        activation = Activation(activation, d_output, dim=1 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear

class SquaredReLU(nn.Module):
    def forward(self, x):
        # return F.relu(x)**2
        return torch.square(F.relu(x))  # Could this be faster?

def laplace(x, mu=0.707107, sigma=0.282095):
    x = (x - mu).div(sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + torch.erf(x))

class Laplace(nn.Module):
    def __init__(self, mu=0.707107, sigma=0.282095):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return laplace(x, mu=self.mu, sigma=self.sigma)


class TransposedLinear(nn.Module):
    """ Linear module on the second-to-last dimension
    Assumes shape (B, D, L), where L can be 1 or more axis
    """

    def __init__(self, d_input, d_output, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(d_output, d_input))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # nn.Linear default init
        # nn.init.kaiming_uniform_(self.weight, nonlinearity='linear') # should be equivalent

        if bias:
            self.bias = nn.Parameter(torch.empty(d_output))
            bound = 1 / math.sqrt(d_input)
            nn.init.uniform_(self.bias, -bound, bound)
            setattr(self.bias, "_optim", {"weight_decay": 0.0})
        else:
            self.bias = 0.0

    def forward(self, x):
        num_axis = len(x.shape[2:])  # num_axis in L, for broadcasting bias
        y = contract('b u ..., v u -> b v ...', x, self.weight) + self.bias.view(-1, *[1]*num_axis)
        return y


class TransposedLN(nn.Module):
    """ LayerNorm module over second dimension
    Assumes shape (B, D, L), where L can be 1 or more axis

    This is slow and a dedicated CUDA/Triton implementation shuld provide substantial end-to-end speedup
    """
    def __init__(self, d, scalar=True):
        super().__init__()
        self.scalar = scalar
        if self.scalar:
            self.m = nn.Parameter(torch.zeros(1))
            self.s = nn.Parameter(torch.ones(1))
            setattr(self.m, "_optim", {"weight_decay": 0.0})
            setattr(self.s, "_optim", {"weight_decay": 0.0})
        else:
            self.ln = nn.LayerNorm(d)

    def forward(self, x):
        if self.scalar:
            # calc. stats over D dim / channels
            s, m = torch.std_mean(x, dim=1, unbiased=False, keepdim=True)
            y = (self.s/s) * (x-m+self.m)
        else:
            # move channel to last axis, apply layer_norm, then move channel back to second axis
            _x = self.ln(rearrange(x, 'b d ... -> b ... d'))
            y = rearrange(_x, 'b ... d -> b d ...')
        return y

def Activation(activation=None, size=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'softplus':
        return nn.Softplus()
    elif activation in ['sqrelu', 'relu2']:
        return SquaredReLU()
    elif activation == 'laplace':
        return Laplace()
    elif activation == 'ln':
        return TransposedLN(dim)
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))


class LongConvKernel(nn.Module):
    def __init__(
        self, 
        H, 
        L,
        channels=1, 
        learning_rate=None, 
        lam=0.001, 
        causal=True, 
        kernel_dropout=0,
        weight_init="random",
        use_ma_smoothing = False,
        ma_window_len = 7,
        smooth_freq = False,
        **kwargs
    ):
        super().__init__()
       
        self.drop = torch.nn.Dropout(p=kernel_dropout)
        self.H = H
        self.weight_init = weight_init
        self.causal = causal
        self.L = L*2 if not causal else L
        
        self.channels = channels
        self.lam = lam
        self.kernel = torch.nn.Parameter(self._parameter_initialization()) #(c,H,L) 

        
        self.use_ma_smoothing=use_ma_smoothing
        self.smooth_freq = smooth_freq
        self.ma_window_len = ma_window_len
        if self.use_ma_smoothing:
            if smooth_freq:
                weight = torch.arange(ma_window_len, dtype = self.kernel.dtype)
                weight = torch.exp(-0.5 * torch.abs(weight - ma_window_len // 2) ** 2)
                weight = repeat(weight, 'l -> h1 h2 l', h1 = self.H, h2 = 1)
                weight = weight.type(torch.fft.rfft(self.kernel).dtype)
                self.smooth_weight = weight
            else:
                self.ma_window_len = ma_window_len
                assert self.ma_window_len%2!=0, "window size must be odd"
                padding = (self.ma_window_len//2)
                self.smooth = torch.nn.AvgPool1d(kernel_size=self.ma_window_len,stride=1,padding=padding)

    def _parameter_initialization(self):
        if self.weight_init=="random":
            return torch.randn(self.channels, self.H, self.L) * 0.002
        elif self.weight_init=="double_exp":
            K = torch.randn(self.channels, self.H, self.L,dtype=torch.float32) * 0.02
            double_exp = torch.zeros((self.H,self.L),dtype=torch.float32) 
            for i in range(self.H):
                for j in range(self.L):
                    double_exp[i,j] = torch.exp(-(j/self.L)*torch.pow(torch.tensor(int(self.H/2)),torch.tensor(i/self.H)))
            K = torch.einsum("c h l, h l -> c h l",K,double_exp)
            return K
        else: raise NotImplementedError(f"{self.weight_init} is not valid") 

    def forward(self, **kwargs):
        k = self.kernel
        if self.use_ma_smoothing:
            if self.smooth_freq:
                k_f = torch.fft.rfft(k, dim=-1)
                k_f = F.conv1d(k_f, self.smooth_weight.to(k_f.device), padding='same', groups=self.H)
                k = torch.fft.irfft(k_f, dim=-1)
            else:
                k = self.smooth(k)
        k = F.relu(torch.abs(k)-self.lam)*torch.sign(k)
        k = self.drop(k)
        return k, None

    @property
    def d_output(self):
        return self.H


from torch import nn
import torch
from einops import rearrange

class PositionKernel(nn.Module):
    '''
    Rather than having a parameter for each position in the kernel/seq we just predict a value for the kernel at each positon given the position in the sequence
    Similar to dynamic position embeddings in transformers
    '''
    def __init__(
        self,
        H,
        L,
        channels=1,
        intermediate_dim=32,
        activation=nn.ReLU,
        **kwargs
    ):
        super().__init__()
        self.H = H
        self.L = L
        self.channels = channels
        self.kernel = nn.Sequential(
            nn.Linear(3, intermediate_dim),
            activation(),
            nn.Linear(intermediate_dim, self.H*self.channels)
        )
        # init linear layers to be small i.e normal(0,0.002)
        for layer in self.kernel:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=0.002)
                layer.bias.data.normal_(mean=0.0, std=0.002)

        self.base_rates = torch.nn.Parameter(torch.tensor([0.01, 1.0, 1.0]))
     
    def forward(self, L, rate=1.0, state=None):
        # create linear sequence of float of length L increasing by rate
        L = min(L, self.L)
        x = torch.arange(L, dtype=torch.float32, device=self.kernel[0].weight.device)[:, None].repeat(1,3) + 1
   
        a, b, c = x.transpose(-1,-2)
        a = a*self.base_rates[0]
        b = (b*self.base_rates[1]).log()
        c = torch.sin(c*self.base_rates[2])
        x = torch.stack([a,b,c],dim=1)
    
        # get kernel
        k = self.kernel(x) # (L, H*C)
        k = rearrange(k, 'l (c h) -> c h l', c=self.channels) # (C, H, L)
        
        #k = (self.squash((self.squash(k + self.limit) - self.limit) * -1 + self.limit) - self.limit) * -1
        return k, None
        

class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim: int, seq_len: int, **kwargs): 
        """Complex exponential positional embeddings for Hyena filters."""  
        super().__init__()
        
        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None] # 1, L, 1
        
        if emb_dim > 1:
            bands = (emb_dim - 1) // 2            
        # To compute the right embeddings we use the "proper" linspace 
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len # 1, L, 1 
        
        f = torch.linspace(1e-4, bands - 1, bands)[None, None] 
        z = torch.exp(-1j * f * w)
        self.z = torch.nn.Parameter(torch.cat([t, z.real, z.imag], dim=-1), requires_grad=True) # add backwards hook to decreae lr
        self.t = torch.nn.Parameter(t, requires_grad=False)
        
    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(nn.Module):
    def __init__(
        self,
        d_model,
        channels=1,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulation_lr=0.0,
        modulate: bool=True,
        shift: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None].repeat(channels, 1, 1)
        self.deltas = nn.Parameter(deltas, requires_grad=True) # add backwards hook to decreae lr
     
        
    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs()) 
            x = x * (decay + self.shift)
        return x                  

class Sin(nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)

class HyenaFilter(nn.Module):
    def __init__(
            self, 
            d_model,
            channels=1,
            emb_dim=3, # dim of input to MLP, augments with positional encoding
            order=16, # width of the implicit MLP 
            fused_fft_conv=False,
            seq_len=1024, 
            w=1, # frequency of periodic activations 
            wd=0, # weight decay of kernel parameters 
            bias=True,
            num_inner_mlps=2,
            normalized=False,
            **kwargs
        ):
        """
        Implicit long filter with modulation.
        
        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP
        """
        super().__init__()
        self.d_model = d_model
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        
        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert emb_dim % 2 != 0 and emb_dim >= 3, "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.seq_len = seq_len
  
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len)
        
        self.implicit_filter = nn.Sequential(
            nn.Linear(emb_dim, order),
            act,
        )
        for i in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order))
            self.implicit_filter.append(act)

        self.implicit_filter.append(nn.Linear(order, d_model*channels, bias=False))
        self.channels = channels
            
        self.modulation = ExponentialModulation(d_model, **kwargs)
        
        self.normalized = normalized
    

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = rearrange(h, '1 l (c h) -> (1 c) l h', c=self.channels) # (C, L, H)
        h = self.modulation(t, h).transpose(-1, -2) # (C, H, L)
        return h

    def forward(self, L, **kwargs):
        k = self.filter(L)
        return k

class LongConv(nn.Module):
    def __init__(
            self,
            d_model,
            l_max=1024,
            channels=1,
            bidirectional=False,
            position_kernel=True,
            # Arguments for position-wise feedforward components
            #activation='gelu', # activation between conv and FF
            postact='glu', # activation after FF (glu)
            initializer=None, # initializer on FF
            weight_norm=False, # weight normalization on FF
            dropout=0.0, tie_dropout=False,
            transposed=True, # axis ordering (B, L, D) or (B, D, L)
            verbose=False,
            # SSM Kernel arguments
            **kernel_args,
        ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum kernel length, also denoted by L
        channels: can be interpreted as a number of "heads"; the SSM is a map from a 1-dim to C-dim sequence. It's not recommended to change this unless desperate for things to tune; instead, increase d_model for larger models
        bidirectional: if True, convolution kernel will be two-sided

        Position-wise feedforward components:
        --------------------
        activation: activation in between SS and FF
        postact: activation after FF ('id' for no activation, None to remove FF layer)
        initializer: initializer on FF
        weight_norm: weight normalization on FF
        dropout: standard dropout argument. tie_dropout=True ties the dropout mask across the sequence length, emulating nn.Dropout1d

        Other arguments:
        --------------------
        transposed: choose backbone axis ordering of (B, L, H) (if False) or (B, H, L) (if True) [B=batch size, L=sequence length, H=hidden dimension]
        """

        super().__init__()
        # if verbose:
        #     import src.utils.train
        #     log = src.utils.train.get_logger(__name__)
        #     log.info(f"Constructing Long Conv (H, L) = ({d_model}, {l_max})")

        self.d_model = d_model
        self.H = d_model
        self.L = l_max
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(channels, self.H))

        if self.bidirectional:
            channels *= 2

        #SSM Kernel
        if not position_kernel:
            self.kernel = LongConvKernel(self.H, L=self.L, channels=channels, verbose=verbose, **kernel_args)
        else:
            self.kernel = PositionKernel(self.H, L=self.L, channels=channels, **kernel_args)
        # self.kernel = HyenaFilter(d_model=self.H, channels=channels, seq_len=self.L)

        # Pointwise
        self.activation = Activation('gelu')
        # dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout # Broken in torch==1.11
        dropout_fn = DropoutNd if tie_dropout else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        if postact is None:
            self.output_linear = nn.Identity()
        else:
            self.output_linear = LinearActivation(
                self.d_model * self.channels,
                self.d_model,
                # self.H*self.channels,
                # self.d_model*(1 if self.gate is None else self.gate),
                transposed=self.transposed,
                initializer=initializer,
                activation=postact,
                activate=True,
                weight_norm=weight_norm,
            )


    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32) # rfft doesn't support bf16 and doesnt fully support fp16 (requires seq + kernel length to be power of 2)
    def forward(self, u, state=None, rate=1.0, lengths=None, **kwargs): # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed, remnant from state spaces repo

        Returns: same shape as u
        """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)
        # Mask out padding tokens
        # TODO handle option for mask - instead of lengths, which assumes suffix padding
        if isinstance(lengths, int):
            if lengths != L:
                lengths = torch.tensor(lengths, dtype=torch.long, device=u.device)
            else:
                lengths = None
        if lengths is not None:
            assert isinstance(lengths, torch.Tensor) and lengths.ndim == 1 and lengths.size(0) in [1, u.size(0)]
            mask = torch.where(torch.arange(L, device=lengths.device) < lengths[:, None, None], 1., 0.)
            u = u * mask

        # Compute SS Kernel
        L_kernel = L if self.L is None else min(L, round(self.L / rate))
        k, _ =  self.kernel(L=L_kernel, rate=rate, state=state) # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, L)) \
                    + F.pad(k1.flip(-1), (L, 0))
            # this pads k0 with zeros on the right and k1 with zeros on the left to a length of L + L_kernel
        
        k_f = torch.fft.rfft(k, n=L_kernel+L) # (C H L)
        u_f = torch.fft.rfft(u, n=L_kernel+L) # (B H L)
        y_f = contract('bhl,chl->bchl', u_f, k_f) 
     
        y = torch.fft.irfft(y_f, n=L_kernel+L)[..., :L] # (B C H L)
        # Compute skip connection
        y = y + contract('bhl,ch->bchl', u, self.D)

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        if not self.transposed: y = y.transpose(-1, -2)
        y = self.activation(y)
        y = self.dropout(y)

        y = self.output_linear(y)

        return y

    @property
    def d_state(self):
        return self.H

    @property
    def d_output(self):
        return self.d_model


from time import time
if __name__ == '__main__':

    # f = HyenaFilter(d_model=768, channels=2, seq_len=1024)
    # f(256)
    # exit()
    print('oki')
    hascuda = torch.cuda.is_available()
    lcm = LongConv(
        d_model=256,
        l_max=8192,
        bidirectional=True,
        transposed=False,
        channels = 1,
        weight_init = 'random'
    )
    print(f'total params: {sum(p.numel() for p in lcm.parameters()) / 1e6} (M)')
    lcm.to('cuda' if hascuda else 'cpu')
    print('HERE')
    B, L, H = 5, 118192, 256
    stime = time()
    
    u = torch.randn(B, L, H).to('cuda' if hascuda else 'cpu')
    #u = rearrange(u, 'b l (h 1) -> (b h) l 1', h=H)
    print(u.device)
    y = lcm(u)
    print(time() - stime)
    print(y.shape)

    