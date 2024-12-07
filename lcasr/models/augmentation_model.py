import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor
from lcasr.components.convolution import GatedConv1d
from lcasr.components.helpers import ResidualBlock
from lcasr.components.batchrenorm import BatchRenorm1d
from lcasr.models.base import BaseModel

class SoftMaskNN(BaseModel):
    def __init__(self, layers:int=4) -> None:
        super().__init__()
    
        self.network = nn.Sequential(
            *[ResidualBlock(
                nn.Sequential(
                    GatedConv1d(input_dim=80, output_dim=80, expansion_factor=2, kernel_size=(9,9), stride=(1,1), padding=("same", "same")),
                    BatchRenorm1d(80)
                )
            ) for _ in range(layers)], 
            nn.Conv1d(in_channels=80, out_channels=80, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x:Tensor, apply_mask:bool=True) -> Tensor:
        return x * self.network(x) if apply_mask else self.network(x)



if __name__ == '__main__':
    # run test
    smnn = SoftMaskNN(layers=4)
    x = torch.rand(2, 80, 4096)
    print(smnn(x, apply_mask=True).shape)
    smnn.print_total_params()