import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional

from lcasr.components.attention import Attention
from lcasr.components.helpers import get_act
from lcasr.components.positional_encodings import RotaryPositionalEmbedding, apply_rotary
from lcasr.components.subsampling import ConvSubsampling, calc_length
from lcasr.models.base import BaseModel

try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except Exception:
        from torch.nn import LayerNorm


class CausalDecoderLayer(nn.Module):
    """Causal self-attention plus feed-forward block for streaming decoder ASR."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expansion_factor: int = 4,
        dropout_ff: float = 0.0,
        dropout_attn: float = 0.0,
        activation: str = "silu",
    ):
        """Initialize one causal decoder layer with residual attention and FFN paths."""
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

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb_fn=None,
        cached_kv: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        max_cache_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply causal self-attention and feed-forward residual updates to `[B, T, D]`."""
        x_norm = self.attn_norm(x)
        attn_out = self.attn(
            x_norm,
            flash_attn=True,
            rotary_emb_fn=rotary_emb_fn,
            cached_kv=cached_kv,
            use_cache=use_cache,
            max_cache_length=max_cache_length,
        )
        next_cache = None
        if use_cache:
            attn_out, next_cache = attn_out
        x = x + attn_out
        x = x + self.ff(self.ff_norm(x))
        if use_cache:
            return x, next_cache
        return x


class StreamingDecoderASR(BaseModel):
    """Decoder-only streaming ASR model with causal acoustic subsampling."""

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
        use_rotary: bool = True,
        rotary_base_freq: int = 1_500_000,
        rotary_interpolation_factor: float = 1.0,
        **kwargs,
    ):
        """Build the streaming decoder, previous-token embedding, and two prediction heads."""
        super().__init__()
        self.vocab_size = vocab_size
        self.silence_id = vocab_size
        self.num_classes = vocab_size + 1
        self.subsampling_factor = subsampling_factor
        self.previous_token_dropout = previous_token_dropout
        self.use_rotary = use_rotary
        self.rotary_base_freq = rotary_base_freq
        self.rotary_interpolation_factor = rotary_interpolation_factor

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
        self.rotary_pos_emb = None
        if self.use_rotary:
            self.rotary_pos_emb = RotaryPositionalEmbedding(
                dim=d_model // n_heads,
                base=rotary_base_freq,
                rotary_interpolation_factor=rotary_interpolation_factor,
            )
            self._mark_rotary_buffers_non_persistent()

    def _mark_rotary_buffers_non_persistent(self) -> None:
        """Keep deterministic RoPE buffers out of checkpoints for old-state compatibility."""
        if self.rotary_pos_emb is None:
            return
        self.rotary_pos_emb._non_persistent_buffers_set.update(
            {"inv_freq", "rotary_interpolation_factor"}
        )

    def _rotary_emb_fn(self, seq_len: int, device: torch.device, offset: int = 0):
        """Build the shared attention rotary callback for the current sequence length."""
        if self.rotary_pos_emb is None:
            return None
        cos, sin = self.rotary_pos_emb(seq_len + offset, device)
        if offset:
            cos = cos[:, offset : offset + seq_len]
            sin = sin[:, offset : offset + seq_len]
        return apply_rotary(cos=cos, sin=sin, learned=self.rotary_pos_emb.learned_freq)

    def get_silence_id(self) -> int:
        """Return the extra class id used for frame-level silence predictions."""
        return self.silence_id

    def _decode_prediction_ids(
        self,
        tokenizer,
        prediction_ids: Iterable[int],
        max_tokens: Optional[int] = None,
    ) -> str:
        """Drop silence frame ids and decode the remaining text ids with the tokenizer."""
        tokens = []
        for idx in prediction_ids:
            idx = int(idx)
            if idx == self.silence_id:
                continue
            tokens.append(idx)
            if max_tokens is not None and len(tokens) >= max_tokens:
                break
        return "" if not tokens else tokenizer.decode(tokens)

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """Compute post-subsampling frame lengths for input spectrogram lengths."""
        return calc_length(
            lengths=lengths,
            all_paddings=self.subsampling._left_padding + self.subsampling._right_padding,
            kernel_size=self.subsampling._kernel_size,
            stride=self.subsampling._stride,
            ceil_mode=self.subsampling._ceil_mode,
            repeat_num=self.subsampling._sampling_num,
        )

    def kv_cache_length_from_spectrogram_length(self, spectrogram_length: int) -> int:
        """Convert an input spectrogram-frame window into cached decoder-frame length."""
        if spectrogram_length <= 0:
            raise ValueError("spectrogram_length must be positive")
        length = torch.tensor([spectrogram_length], dtype=torch.long)
        return int(self.output_lengths(length)[0].item())

    def _previous_targets(self, frame_targets: Optional[torch.Tensor], batch: int, length: int, device) -> torch.Tensor:
        """Shift teacher frame targets right to form previous-token decoder inputs."""
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
        """Combine two-head outputs as log `P(token, not-silence)` plus log `P(silence)`."""
        silence_log_probs = F.log_softmax(silence_logits, dim=-1)
        text_log_probs = F.log_softmax(text_logits, dim=-1)
        silence_score = silence_log_probs[..., 0:1]
        token_scores = text_log_probs + silence_log_probs[..., 1:2]
        return torch.cat([token_scores, silence_score], dim=-1)

    def _step_predictions(self, hidden: torch.Tensor, sample: bool = False, temperature: float = 1.0) -> torch.Tensor:
        """Predict the next frame ids from hidden states for one autoregressive step."""
        return self._predict_ids(
            self.silence_head(hidden),
            self.text_head(hidden),
            sample=sample,
            temperature=temperature,
        )

    def _predict_ids(
        self,
        silence_logits: torch.Tensor,
        text_logits: torch.Tensor,
        sample: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Return frame ids using the joint silence/text distribution."""
        combined_logits = self._combined_logits(silence_logits, text_logits)
        if not sample:
            return combined_logits.argmax(dim=-1)
        temperature = max(float(temperature), 1e-6)
        probs = (combined_logits / temperature).softmax(dim=-1)
        return torch.multinomial(probs.reshape(-1, self.num_classes), num_samples=1).view(
            combined_logits.shape[:-1]
        )

    def forward(
        self,
        audio_signal: torch.Tensor,
        length: Optional[torch.Tensor] = None,
        frame_targets: Optional[torch.Tensor] = None,
        return_logits: bool = True,
    ) -> dict:
        """Run teacher-forced streaming decoding over `[B, F, T]` spectrogram batches."""
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
        rotary_emb_fn = self._rotary_emb_fn(x.size(1), x.device)
        for layer in self.layers:
            x = layer(x, rotary_emb_fn=rotary_emb_fn)
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
        """Compute the two-head training loss for frame-synchronous ASR targets."""
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
        use_kv_cache: bool = False,
        max_kv_cache_length: Optional[int] = None,
        max_kv_cache_spectrogram_length: Optional[int] = None,
    ) -> dict:
        """Autoregressively decode by greedy two-head prediction at each output frame."""
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

        if use_kv_cache:
            if x.size(0) != 1:
                raise ValueError("KV-cache greedy decoding currently supports batch size 1")
            if max_kv_cache_length is not None and max_kv_cache_spectrogram_length is not None:
                raise ValueError(
                    "Pass either max_kv_cache_length or max_kv_cache_spectrogram_length, not both"
                )
            effective_max_kv_cache_length = max_kv_cache_length
            if max_kv_cache_spectrogram_length is not None:
                effective_max_kv_cache_length = self.kv_cache_length_from_spectrogram_length(
                    max_kv_cache_spectrogram_length
                )
            caches = [None for _ in self.layers]
            prev_id = torch.full((x.size(0),), self.silence_id, dtype=torch.long, device=x.device)
            for step in range(x.size(1)):
                h = x[:, step : step + 1] + self.prev_token_embedding(prev_id).unsqueeze(1)
                rotary_emb_fn = self._rotary_emb_fn(1, h.device, offset=step)
                for layer_idx, layer in enumerate(self.layers):
                    h, caches[layer_idx] = layer(
                        h,
                        rotary_emb_fn=rotary_emb_fn,
                        cached_kv=caches[layer_idx],
                        use_cache=True,
                        max_cache_length=effective_max_kv_cache_length,
                    )
                step_h = self.norm(h[:, 0])
                step_prediction = self._step_predictions(step_h, sample=False)
                predictions.append(step_prediction)
                prev_id = step_prediction

            if was_training:
                self.train()
            return {"predictions": torch.stack(predictions, dim=1), "length": out_lengths}

        for step in range(x.size(1)):
            h = x + self.prev_token_embedding(prev_ids)
            if key_padding_mask is not None:
                h = h.masked_fill(key_padding_mask.unsqueeze(-1), 0)
            rotary_emb_fn = self._rotary_emb_fn(h.size(1), h.device)
            for layer in self.layers:
                h = layer(h, rotary_emb_fn=rotary_emb_fn)
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
        use_kv_cache: bool = False,
        max_kv_cache_length: Optional[int] = None,
        max_kv_cache_spectrogram_length: Optional[int] = None,
    ) -> dict:
        """Autoregressively decode while sampling joint silence/text frame ids."""
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

        if use_kv_cache:
            if x.size(0) != 1:
                raise ValueError("KV-cache sampling currently supports batch size 1")
            if max_kv_cache_length is not None and max_kv_cache_spectrogram_length is not None:
                raise ValueError(
                    "Pass either max_kv_cache_length or max_kv_cache_spectrogram_length, not both"
                )
            effective_max_kv_cache_length = max_kv_cache_length
            if max_kv_cache_spectrogram_length is not None:
                effective_max_kv_cache_length = self.kv_cache_length_from_spectrogram_length(
                    max_kv_cache_spectrogram_length
                )
            caches = [None for _ in self.layers]
            prev_id = torch.full((x.size(0),), self.silence_id, dtype=torch.long, device=x.device)
            for step in range(x.size(1)):
                h = x[:, step : step + 1] + self.prev_token_embedding(prev_id).unsqueeze(1)
                rotary_emb_fn = self._rotary_emb_fn(1, h.device, offset=step)
                for layer_idx, layer in enumerate(self.layers):
                    h, caches[layer_idx] = layer(
                        h,
                        rotary_emb_fn=rotary_emb_fn,
                        cached_kv=caches[layer_idx],
                        use_cache=True,
                        max_cache_length=effective_max_kv_cache_length,
                    )
                step_h = self.norm(h[:, 0])
                step_prediction = self._step_predictions(
                    step_h,
                    sample=True,
                    temperature=temperature,
                )
                predictions.append(step_prediction)
                prev_id = step_prediction

            if was_training:
                self.train()
            return {"predictions": torch.stack(predictions, dim=1), "length": out_lengths}

        for step in range(x.size(1)):
            h = x + self.prev_token_embedding(prev_ids)
            if key_padding_mask is not None:
                h = h.masked_fill(key_padding_mask.unsqueeze(-1), 0)
            rotary_emb_fn = self._rotary_emb_fn(h.size(1), h.device)
            for layer in self.layers:
                h = layer(h, rotary_emb_fn=rotary_emb_fn)
            step_h = self.norm(h[:, step])
            step_prediction = self._step_predictions(step_h, sample=True, temperature=temperature)
            predictions.append(step_prediction)
            if step + 1 < x.size(1):
                prev_ids[:, step + 1] = step_prediction

        if was_training:
            self.train()
        return {"predictions": torch.stack(predictions, dim=1), "length": out_lengths}

    @torch.no_grad()
    def transcribe(
        self,
        audio_spec,
        tokenizer,
        device=None,
        decode_mode: str = "greedy",
        temperature: float = 1.0,
        max_sequence_length: Optional[int] = None,
        max_output_frames: Optional[int] = None,
        max_tokens: Optional[int] = None,
        use_kv_cache: bool = False,
        max_kv_cache_length: Optional[int] = None,
        max_kv_cache_spectrogram_length: Optional[int] = None,
        return_metadata: bool = False,
        **kwargs,
    ):
        """Transcribe one spectrogram, or a list of spectrograms, through streaming decoding."""
        kwargs.pop("verbose", None)
        if kwargs:
            raise TypeError(f"Unsupported StreamingDecoderASR.transcribe kwargs: {sorted(kwargs)}")
        if isinstance(audio_spec, (list, tuple)):
            return [
                self.transcribe(
                    item,
                    tokenizer,
                    device=device,
                    decode_mode=decode_mode,
                    temperature=temperature,
                    max_sequence_length=max_sequence_length,
                    max_output_frames=max_output_frames,
                    max_tokens=max_tokens,
                    use_kv_cache=use_kv_cache,
                    max_kv_cache_length=max_kv_cache_length,
                    max_kv_cache_spectrogram_length=max_kv_cache_spectrogram_length,
                    return_metadata=return_metadata,
                )
                for item in audio_spec
            ]

        if device is None:
            device = next(self.parameters()).device
        device = torch.device(device)
        if audio_spec.dim() == 3 and audio_spec.size(0) == 1:
            audio_spec = audio_spec.squeeze(0)
        if audio_spec.dim() != 2:
            raise ValueError(f"Expected audio spectrogram with shape [features, frames], got {tuple(audio_spec.shape)}")

        if max_sequence_length is not None and max_sequence_length > 0:
            audio_spec = audio_spec[..., :max_sequence_length]

        model_dtype = next(self.parameters()).dtype
        audio_signal = audio_spec.to(device=device, dtype=model_dtype).unsqueeze(0)
        length = torch.tensor([audio_signal.shape[-1]], dtype=torch.long, device=device)

        if decode_mode == "sample":
            decoded = self.sample_decode(
                audio_signal=audio_signal,
                length=length,
                max_frames=max_output_frames,
                temperature=temperature,
                use_kv_cache=use_kv_cache,
                max_kv_cache_length=max_kv_cache_length,
                max_kv_cache_spectrogram_length=max_kv_cache_spectrogram_length,
            )
        elif decode_mode == "greedy":
            decoded = self.greedy_decode(
                audio_signal=audio_signal,
                length=length,
                max_frames=max_output_frames,
                use_kv_cache=use_kv_cache,
                max_kv_cache_length=max_kv_cache_length,
                max_kv_cache_spectrogram_length=max_kv_cache_spectrogram_length,
            )
        else:
            raise ValueError(f"Unsupported decode_mode: {decode_mode}")

        pred_len = int(decoded["length"][0].item())
        prediction_ids = decoded["predictions"][0, :pred_len].detach().cpu().tolist()
        text = self._decode_prediction_ids(tokenizer, prediction_ids, max_tokens=max_tokens)
        if not return_metadata:
            return text
        return {
            "text": text,
            "prediction_ids": prediction_ids,
            "output_frames": pred_len,
            "pred_non_silence_fraction": sum(idx != self.silence_id for idx in prediction_ids) / max(len(prediction_ids), 1),
        }
