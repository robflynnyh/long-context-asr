from torch import nn

def get_act(act:str):
    if act == 'silu':
        return nn.SiLU()
    elif act == 'relu':
        return  nn.ReLU()
    elif act == 'gelu':
        return  nn.GELU()
    elif act == 'none':
        return  nn.Identity()
    else:
        raise ValueError(f'Activation {act} not supported.')
            

class ResidualBlock(nn.Module):
    def __init__(
            self,
            module:nn.Module,
            ) -> None:
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        return x + self.module(x, *args, **kwargs)
        
