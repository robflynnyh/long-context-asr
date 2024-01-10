from torch import nn
from lcasr.components.batchrenorm import BatchRenorm1d

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
        

        
