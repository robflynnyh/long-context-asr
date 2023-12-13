import torch, torch.nn as nn
from lcasr.components.batchrenorm import BatchRenorm1d

try:
    from flashfftconv.depthwise_1d import conv1d_forward, conv1d_backward
    class conv1dFunc(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, input, weights, bias, padding, is_bhl=True):
            outputs = conv1d_forward(input, weights, bias, padding, is_bhl)
            ctx.padding = padding
            ctx.is_bhl = is_bhl
            ctx.save_for_backward(input, weights, bias)
            return outputs

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, dout):
            input, weight, bias = ctx.saved_tensors
            dout  = dout.contiguous()
            du, dk, dbias = conv1d_backward(dout, input, weight, bias, ctx.padding, ctx.is_bhl)
            return du, dk, dbias, None, None
except:
    conv1dFunc = None




def get_norm(norm_type, d_model):
    if norm_type == 'batch_norm':
        return nn.BatchNorm1d(d_model)
    elif norm_type == 'layer_norm':
        return nn.LayerNorm(d_model)
    elif norm_type == 'group_norm':
        return nn.GroupNorm(num_groups=32, num_channels=d_model)
    elif norm_type == 'batch_renorm':
        return BatchRenorm1d(d_model)
    elif norm_type == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"conv_norm_type={norm_type} is not valid!")

class ConformerConvolution(nn.Module):
    """The convolution module for the Conformer model.
    Args:
        d_model (int): hidden dimension
        kernel_size (int): kernel size for depthwise convolution
    """

    def __init__(
            self, 
            d_model, 
            kernel_size, 
            norm_type='batch_renorm', 
            exp_factor=1,
            **kwargs
            ):

        super(ConformerConvolution, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model
     
        inner_dim = int(d_model * exp_factor)
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=inner_dim * 2, kernel_size=1, stride=1, padding=0, bias=True
        )

        dw_conv = nn.Conv1d # if weight_standardization == False else WSConv1d
        self.depthwise_conv = dw_conv(
            in_channels=inner_dim,
            out_channels=inner_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=inner_dim,
            bias=True,
        )
        self.conv_kernel_size = kernel_size
        self.conv_padding = (kernel_size - 1) // 2
        
        self.batch_norm = get_norm(norm_type, inner_dim)

        self.use_fft_conv = kwargs.get('use_fft_conv', False) and conv1dFunc is not None

        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=inner_dim, out_channels=d_model, kernel_size=1, stride=1, padding=0, bias=True
        )

    def exec_conv(self, x):
        is_cuda = x.device.type == 'cuda'
        if self.use_fft_conv and is_cuda:
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                return conv1dFunc.apply(
                    x.contiguous(),
                    self.depthwise_conv.weight.squeeze().contiguous(),
                    self.depthwise_conv.bias.contiguous(),
                    self.conv_padding,
                    True
                )
        else:
            return self.depthwise_conv(x)


    def forward(self, x, pad_mask=None, **kwargs):
  
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)

        if pad_mask is not None: 
            x = x.float().masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.exec_conv(x)

        if isinstance(self.batch_norm, nn.LayerNorm):
            x = x.transpose(1, 2)
            x = self.batch_norm(x) 
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        return x




if __name__ == '__main__':
    # runtest
    
    B, N, D = 16, 1024, 768
    dtype = torch.float32
    torch.manual_seed(0)
    x = torch.randn(B, N, D, device = 'cuda', dtype=dtype)
    
    conv_module = ConformerConvolution(
        d_model = D,
        kernel_size = 9,
        use_fft_conv = True
    )
    conv_module = conv_module.to(dtype=x.dtype, device=x.device)

    with torch.autocast(device_type='cuda', dtype=dtype):
        out1 = conv_module(x)

    conv_module.use_fft_conv = False

    with torch.autocast(device_type='cuda', dtype=dtype):
        out2 = conv_module(x)

    print(x.shape, out1.shape, out2.shape)
    print(out1[0,0,:10], out2[0,0,:10])
    print(torch.allclose(out1, out2))