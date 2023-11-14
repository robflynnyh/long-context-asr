import torch, torch.nn as nn
from lcasr.components.batchrenorm import BatchRenorm1d



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
        
        self.batch_norm = get_norm(norm_type, inner_dim)


        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=inner_dim, out_channels=d_model, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x, pad_mask=None, **kwargs):
  
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)

        if pad_mask is not None: 
            x = x.float().masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x)

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




