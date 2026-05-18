import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from lcasr.components.attention import Attention
from lcasr.components.helpers import get_act
from lcasr.components.subsampling import ConvSubsampling, calc_length
from lcasr.models.base import BaseModel

try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except Exception:
        from torch.nn import LayerNorm


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
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.attn_norm = LayerNorm(d_model)
        self.attn = Attention(
            n_feats=d_model,
            head_dim=d_model // n_heads,
            n_heads=n_heads,
            dropout=dropout_attn,
            causal=True,
            qkv_bias=False,
            bias=False,
        )
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
        x_norm = self.attn_norm(x)
        attn_mask = None if key_padding_mask is None else ~key_padding_mask
        attn_lengths = None if attn_mask is None else attn_mask.sum(dim=-1)
        x = x + self.attn(
            x_norm,
            attn_mask=attn_mask,
            length=attn_lengths,
            pad_mask=key_padding_mask,
            flash_attn=True,
        )
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
        subsampling_conv_chunking_factor: int = 1,
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
            subsampling_conv_chunking_factor=subsampling_conv_chunking_factor,
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
        self.silence_head = nn.Linear(d_model, 2, bias=False)
        self.text_head = nn.Linear(d_model, vocab_size, bias=False)

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

    def _combined_logits(self, silence_logits: torch.Tensor, text_logits: torch.Tensor) -> torch.Tensor:
        silence_score = silence_logits[..., 0:1]
        token_scores = text_logits + silence_logits[..., 1:2]
        return torch.cat([token_scores, silence_score], dim=-1)

    def _step_predictions(self, hidden: torch.Tensor, sample: bool = False, temperature: float = 1.0) -> torch.Tensor:
        return self._predict_ids(
            self.silence_head(hidden),
            self.text_head(hidden),
            sample_silence=sample,
            silence_temperature=temperature,
        )

    def _predict_ids(
        self,
        silence_logits: torch.Tensor,
        text_logits: torch.Tensor,
        sample_silence: bool = False,
        silence_temperature: float = 1.0,
    ) -> torch.Tensor:
        if sample_silence:
            silence_temperature = max(float(silence_temperature), 1e-6)
            silence_pred = torch.multinomial(
                (silence_logits / silence_temperature).softmax(dim=-1).reshape(-1, 2),
                num_samples=1,
            ).view(silence_logits.shape[:-1])
        else:
            silence_pred = silence_logits.argmax(dim=-1)
        text_pred = text_logits.argmax(dim=-1)
        return torch.where(
            silence_pred.bool(),
            text_pred,
            torch.full_like(text_pred, self.silence_id),
        )

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
        x = self.norm(x)
        silence_logits = self.silence_head(x)
        text_logits = self.text_head(x)
        logits = self._combined_logits(silence_logits, text_logits)
        output = {
            "logits": logits,
            "silence_logits": silence_logits,
            "text_logits": text_logits,
            "length": out_lengths,
        }
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
        frame_targets = frame_targets.to(logits.device)

        valid = frame_targets != -100
        non_silence = (frame_targets != self.silence_id) & valid
        silence_log_probs = F.log_softmax(out["silence_logits"], dim=-1)
        silence_targets = non_silence.long()
        silence_loss = F.nll_loss(
            silence_log_probs.reshape(-1, 2),
            silence_targets.reshape(-1),
            reduction="none",
        ).view_as(silence_targets)
        silence_loss = silence_loss[valid].mean() if valid.any() else silence_loss.sum() * 0.0
        if non_silence.any():
            text_loss = F.cross_entropy(
                out["text_logits"][non_silence],
                frame_targets[non_silence].to(logits.device),
            )
        else:
            text_loss = out["text_logits"].sum() * 0.0
        loss = silence_loss + text_loss

        with torch.no_grad():
            silence = (frame_targets == self.silence_id) & valid
            silence_fraction = silence.sum().float() / valid.sum().clamp_min(1).float()
            non_silence_fraction = non_silence.sum().float() / valid.sum().clamp_min(1).float()
            predictions = self._predict_ids(out["silence_logits"], out["text_logits"])
            predicted_non_silence = (predictions != self.silence_id) & valid
            predicted_non_silence_fraction = (
                predicted_non_silence.sum().float() / valid.sum().clamp_min(1).float()
            )
        return {
            **out,
            "loss": loss,
            "predictions": predictions,
            "display_losses": {
                "loss": float(loss.detach().cpu()),
                "silence_loss": float(silence_loss.detach().cpu()),
                "non_silence_loss": float(text_loss.detach().cpu()),
                "text_loss": float(text_loss.detach().cpu()),
                "silence_fraction": float(silence_fraction.detach().cpu()),
                "non_silence_fraction": float(non_silence_fraction.detach().cpu()),
                "predicted_non_silence_fraction": float(predicted_non_silence_fraction.detach().cpu()),
            },
        }

    @torch.no_grad()
    def greedy_decode(
        self,
        audio_signal: torch.Tensor,
        length: Optional[torch.Tensor] = None,
        max_frames: Optional[int] = None,
    ) -> dict:
        was_training = self.training
        self.eval()
        if length is None:
            length = torch.full((audio_signal.size(0),), audio_signal.size(-1), device=audio_signal.device)

        x = audio_signal.transpose(1, 2)
        x, out_lengths = self.subsampling(x, lengths=length)
        if max_frames is not None and x.size(1) > max_frames:
            x = x[:, :max_frames]
            out_lengths = out_lengths.clamp(max=max_frames)

        key_padding_mask = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1) >= out_lengths.unsqueeze(1)
        key_padding_mask = key_padding_mask if key_padding_mask.any() else None
        prev_ids = torch.full((x.size(0), x.size(1)), self.silence_id, dtype=torch.long, device=x.device)
        predictions = []

        for step in range(x.size(1)):
            h = x + self.prev_token_embedding(prev_ids)
            if key_padding_mask is not None:
                h = h.masked_fill(key_padding_mask.unsqueeze(-1), 0)
            for layer in self.layers:
                h = layer(h, key_padding_mask=key_padding_mask)
            step_h = self.norm(h[:, step])
            step_prediction = self._step_predictions(step_h, sample=False)
            predictions.append(step_prediction)
            if step + 1 < x.size(1):
                prev_ids[:, step + 1] = step_prediction

        if was_training:
            self.train()
        return {"predictions": torch.stack(predictions, dim=1), "length": out_lengths}

    @torch.no_grad()
    def sample_decode(
        self,
        audio_signal: torch.Tensor,
        length: Optional[torch.Tensor] = None,
        max_frames: Optional[int] = None,
        temperature: float = 1.0,
    ) -> dict:
        was_training = self.training
        self.eval()
        if length is None:
            length = torch.full((audio_signal.size(0),), audio_signal.size(-1), device=audio_signal.device)

        x = audio_signal.transpose(1, 2)
        x, out_lengths = self.subsampling(x, lengths=length)
        if max_frames is not None and x.size(1) > max_frames:
            x = x[:, :max_frames]
            out_lengths = out_lengths.clamp(max=max_frames)

        key_padding_mask = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1) >= out_lengths.unsqueeze(1)
        key_padding_mask = key_padding_mask if key_padding_mask.any() else None
        prev_ids = torch.full((x.size(0), x.size(1)), self.silence_id, dtype=torch.long, device=x.device)
        predictions = []

        for step in range(x.size(1)):
            h = x + self.prev_token_embedding(prev_ids)
            if key_padding_mask is not None:
                h = h.masked_fill(key_padding_mask.unsqueeze(-1), 0)
            for layer in self.layers:
                h = layer(h, key_padding_mask=key_padding_mask)
            step_h = self.norm(h[:, step])
            step_prediction = self._step_predictions(step_h, sample=True, temperature=temperature)
            predictions.append(step_prediction)
            if step + 1 < x.size(1):
                prev_ids[:, step + 1] = step_prediction

        if was_training:
            self.train()
        return {"predictions": torch.stack(predictions, dim=1), "length": out_lengths}
