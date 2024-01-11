import torch
from lcasr.components.rotary_emb import RotaryPositionalEmbedding, apply_rotary
import math

class LearnableFourierPosEnc(torch.nn.Module): # code taken from espnet: https://espnet.github.io/espnet/_modules/espnet/nets/pytorch_backend/transformer/embedding.html#LearnableFourierPosEnc
    """Learnable Fourier Features for Positional Encoding.

    See https://arxiv.org/pdf/2106.02795.pdf

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        gamma (float): init parameter for the positional kernel variance
            see https://arxiv.org/pdf/2106.02795.pdf.
        apply_scaling (bool): Whether to scale the input before adding the pos encoding.
        hidden_dim (int): if not None, we modulate the pos encodings with
            an MLP whose hidden layer has hidden_dim neurons.
    """

    def __init__(
        self,
        d_model,
        dropout_rate=0.0,
        gamma=1.0,
        apply_scaling=False,
        hidden_dim=None,
    ):
        """Initialize class."""
        super(LearnableFourierPosEnc, self).__init__()

        self.d_model = d_model

        if apply_scaling:
            self.xscale = math.sqrt(self.d_model)
        else:
            self.xscale = 1.0

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.gamma = gamma
        if self.gamma is None:
            self.gamma = self.d_model // 2

        assert (
            d_model % 2 == 0
        ), "d_model should be divisible by two in order to use this layer."
        self.w_r = torch.nn.Parameter(torch.empty(1, d_model // 2))
        self._reset()  # init the weights

        self.hidden_dim = hidden_dim
        if self.hidden_dim is not None:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(d_model, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, d_model),
            )

    def _reset(self):
        self.w_r.data = torch.normal(
            0, (1 / math.sqrt(self.gamma)), (1, self.d_model // 2)
        )

    def extend_pe(self, x):
        """Reset the positional encodings."""
        position_v = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1).to(x)

        cosine = torch.cos(torch.matmul(position_v, self.w_r))
        sine = torch.sin(torch.matmul(position_v, self.w_r))
        pos_enc = torch.cat((cosine, sine), -1)
        pos_enc /= math.sqrt(self.d_model)

        if self.hidden_dim is None:
            return pos_enc.unsqueeze(0)
        else:
            return self.mlp(pos_enc.unsqueeze(0))


    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        pe = self.extend_pe(x)
        x = x * self.xscale + pe
        return self.dropout(x)