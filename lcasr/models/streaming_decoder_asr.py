import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from lcasr.components.helpers import get_act
from lcasr.components.subsampling import ConvSubsampling, calc_length
from lcasr.models.base import BaseModel

try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except Exception:
    from torch.nn import LayerNorm


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, qkv_bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        dropout_p = self.dropout_p if self.training else 0.0
        if key_padding_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            causal = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
            attn_mask = torch.zeros(length, length, dtype=x.dtype, device=x.device)
            attn_mask = attn_mask.masked_fill(causal, -torch.finfo(x.dtype).max)
            attn_mask = attn_mask.view(1, 1, length, length)

            key_mask = key_padding_mask.view(batch, 1, 1, length)
            key_bias = torch.zeros(batch, 1, 1, length, dtype=x.dtype, device=x.device)
            attn_mask = attn_mask + key_bias.masked_fill(key_mask, -torch.finfo(x.dtype).max)

            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )
        out = out.transpose(1, 2).contiguous().view(batch, length, width)
        return self.out(out)


class CausalDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expansion_factor: int = 4,
        dropout_ff: float = 0.0,
        dropout_attn: float = 0.0,
        activation: str = "silu",
    ):
        super().__init__()
        self.attn_norm = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout_attn)
        self.ff_norm = LayerNorm(d_model)
        hidden = d_model * expansion_factor
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden, bias=False),
            get_act(activation),
            nn.Dropout(dropout_ff),
            nn.Linear(hidden, d_model, bias=False),
            nn.Dropout(dropout_ff),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), key_padding_mask=key_padding_mask)
        x = x + self.ff(self.ff_norm(x))
        return x


class StreamingDecoderASR(BaseModel):
    def __init__(
        self,
        vocab_size: int = 4095,
        feat_in: int = 80,
        subsampling: str = "dw_striding",
        subsampling_factor: int = 8,
        subsampling_conv_channels: int = 256,
        subsampling_act: str = "silu",
        subsampling_norm_out: bool = True,
        n_layers: int = 14,
        d_model: int = 768,
        n_heads: int = 6,
        expansion_factor: int = 4,
        dropout_ff: float = 0.0,
        dropout_attn: float = 0.0,
        decoder_norm: bool = True,
        previous_token_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.silence_id = vocab_size
        self.num_classes = vocab_size + 1
        self.subsampling_factor = subsampling_factor
        self.previous_token_dropout = previous_token_dropout

        self.subsampling = ConvSubsampling(
            subsampling=subsampling,
            subsampling_factor=subsampling_factor,
            feat_in=feat_in,
            feat_out=d_model,
            conv_channels=subsampling_conv_channels,
            activation=get_act(subsampling_act),
            is_causal=True,
            norm_out=subsampling_norm_out,
            default_norm=LayerNorm,
        )
        self.prev_token_embedding = nn.Embedding(self.num_classes, d_model)
        self.layers = nn.ModuleList(
            [
                CausalDecoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    expansion_factor=expansion_factor,
                    dropout_ff=dropout_ff,
                    dropout_attn=dropout_attn,
                    activation=kwargs.get("ff_activation", "silu"),
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = LayerNorm(d_model) if decoder_norm else nn.Identity()
        self.decoder = nn.Linear(d_model, self.num_classes, bias=False)

    def get_silence_id(self) -> int:
        return self.silence_id

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return calc_length(
            lengths=lengths,
            all_paddings=self.subsampling._left_padding + self.subsampling._right_padding,
            kernel_size=self.subsampling._kernel_size,
            stride=self.subsampling._stride,
            ceil_mode=self.subsampling._ceil_mode,
            repeat_num=self.subsampling._sampling_num,
        )

    def _previous_targets(self, frame_targets: Optional[torch.Tensor], batch: int, length: int, device) -> torch.Tensor:
        if frame_targets is None:
            return torch.full((batch, length), self.silence_id, dtype=torch.long, device=device)

        prev = frame_targets[:, :length].clone()
        prev = prev.masked_fill(prev < 0, self.silence_id)
        prev = torch.cat(
            [
                torch.full((prev.size(0), 1), self.silence_id, dtype=torch.long, device=prev.device),
                prev[:, :-1],
            ],
            dim=1,
        )
        if self.training and self.previous_token_dropout > 0:
            drop = torch.rand_like(prev, dtype=torch.float) < self.previous_token_dropout
            prev = prev.masked_fill(drop, self.silence_id)
        return prev

    def forward(
        self,
        audio_signal: torch.Tensor,
        length: Optional[torch.Tensor] = None,
        frame_targets: Optional[torch.Tensor] = None,
        return_logits: bool = True,
    ) -> dict:
        if length is None:
            length = torch.full((audio_signal.size(0),), audio_signal.size(-1), device=audio_signal.device)

        x = audio_signal.transpose(1, 2)
        x, out_lengths = self.subsampling(x, lengths=length)
        prev_ids = self._previous_targets(frame_targets, x.size(0), x.size(1), x.device)
        x = x + self.prev_token_embedding(prev_ids)

        key_padding_mask = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1) >= out_lengths.unsqueeze(1)
        key_padding_mask = key_padding_mask if key_padding_mask.any() else None
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        logits = self.decoder(self.norm(x))
        output = {"logits": logits, "length": out_lengths}
        if not return_logits:
            output["final_posteriors"] = F.log_softmax(logits, dim=-1)
        return output

    def calc_loss(
        self,
        audio_signal: torch.Tensor,
        length: torch.Tensor,
        frame_targets: torch.Tensor,
    ) -> dict:
        out = self.forward(audio_signal=audio_signal, length=length, frame_targets=frame_targets, return_logits=True)
        logits = out["logits"]
        if frame_targets.size(1) != logits.size(1):
            if frame_targets.size(1) < logits.size(1):
                pad = logits.size(1) - frame_targets.size(1)
                frame_targets = F.pad(frame_targets, (0, pad), value=-100)
            else:
                frame_targets = frame_targets[:, : logits.size(1)]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            frame_targets.reshape(-1).to(logits.device),
            ignore_index=-100,
        )
        with torch.no_grad():
            valid = frame_targets != -100
            silence = (frame_targets == self.silence_id) & valid
            silence_fraction = silence.sum().float() / valid.sum().clamp_min(1).float()
        return {
            **out,
            "loss": loss,
            "display_losses": {
                "loss": float(loss.detach().cpu()),
                "silence_fraction": float(silence_fraction.detach().cpu()),
            },
        }
