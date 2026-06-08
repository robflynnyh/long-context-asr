import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
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

    @staticmethod
    def _append_kv_cache(
        kv: torch.Tensor,
        cached_kv: Optional[torch.Tensor] = None,
        max_cache_length: Optional[int] = None,
    ) -> torch.Tensor:
        if cached_kv is not None:
            kv = torch.cat([cached_kv, kv], dim=1)
        if max_cache_length is not None and max_cache_length > 0 and kv.size(1) > max_cache_length:
            kv = kv[:, -max_cache_length:].contiguous()
        return kv

    def _cached_attention(
        self,
        x: torch.Tensor,
        rotary_emb_fn=None,
        cached_kv: Optional[torch.Tensor] = None,
        max_cache_length: Optional[int] = None,
    ):
        q, k, v = self.attn.qkv(x)
        kv = torch.stack([k, v], dim=2)
        kv = self._append_kv_cache(kv, cached_kv=cached_kv, max_cache_length=max_cache_length)
        if rotary_emb_fn is not None:
            k = kv[:, :, 0]
            v = kv[:, :, 1]
            q, k = rotary_emb_fn.apply(q, k)
            attn_kv = torch.stack([k, v], dim=2)
        else:
            attn_kv = kv

        assert self.attn.left_window == -1 and self.attn.right_window == -1, (
            "windowed cached attention is not supported"
        )
        k, v = rearrange(attn_kv, "b n kv h d -> kv b h n d", kv=2).contiguous()
        q = q.transpose(1, 2).contiguous()
        q_len = q.size(2)
        kv_len = k.size(2)
        attn_mask = None
        if q_len > 1:
            cached_len = max(kv_len - q_len, 0)
            query_positions = torch.arange(q_len, device=q.device) + cached_len
            key_positions = torch.arange(kv_len, device=q.device)
            future_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
            attn_mask = torch.zeros((q_len, kv_len), device=q.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(future_mask, torch.finfo(q.dtype).min)
        dropout_p = self.attn.dropout_p if self.training else 0.0
        if not self.attn.return_attention_weights:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )
        else:
            out, _ = self.attn.return_attention_module(q, k, v, None, causal=False)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.attn.out_proj(out), kv

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
        if use_cache:
            attn_out, next_cache = self._cached_attention(
                x_norm,
                rotary_emb_fn=rotary_emb_fn,
                cached_kv=cached_kv,
                max_cache_length=max_cache_length,
            )
        else:
            attn_out = self.attn(
                x_norm,
                flash_attn=True,
                rotary_emb_fn=rotary_emb_fn,
            )
            next_cache = None
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

    def _rotary_emb_fn(
        self,
        seq_len: int,
        device: torch.device,
        offset: int = 0,
        q_offset: int = 0,
        trim_k: bool = False,
    ):
        """Build the shared attention rotary callback for the current sequence length."""
        if self.rotary_pos_emb is None:
            return None
        cos, sin = self.rotary_pos_emb(seq_len + offset, device)
        if offset:
            cos = cos[:, offset : offset + seq_len]
            sin = sin[:, offset : offset + seq_len]
        return apply_rotary(
            cos=cos,
            sin=sin,
            q_offset=q_offset,
            learned=self.rotary_pos_emb.learned_freq,
            trim_k=trim_k,
        )

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

    def _previous_targets(
        self,
        frame_targets: Optional[torch.Tensor],
        batch: int,
        length: int,
        device,
        initial_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Shift teacher frame targets right to form previous-token decoder inputs."""
        if frame_targets is None:
            prev = torch.full((batch, length), self.silence_id, dtype=torch.long, device=device)
            if initial_targets is not None and length > 0:
                prev[:, 0] = initial_targets.to(device=device, dtype=torch.long)
            return prev

        if frame_targets.size(1) < length:
            frame_targets = F.pad(frame_targets, (0, length - frame_targets.size(1)), value=-100)
        prev = frame_targets[:, :length].clone()
        prev = prev.masked_fill(prev < 0, self.silence_id)
        if initial_targets is None:
            first = torch.full((prev.size(0), 1), self.silence_id, dtype=torch.long, device=prev.device)
        else:
            first = initial_targets.to(device=prev.device, dtype=torch.long).view(-1, 1)
        prev = torch.cat(
            [
                first,
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

    def _step_predictions(
        self,
        hidden: torch.Tensor,
        sample: bool = False,
        temperature: float = 1.0,
        sample_silence_only: bool = False,
    ) -> torch.Tensor:
        """Predict the next frame ids from hidden states for one autoregressive step."""
        return self._predict_ids(
            self.silence_head(hidden),
            self.text_head(hidden),
            sample=sample,
            temperature=temperature,
            sample_silence_only=sample_silence_only,
        )

    def _predict_ids(
        self,
        silence_logits: torch.Tensor,
        text_logits: torch.Tensor,
        sample: bool = False,
        temperature: float = 1.0,
        sample_silence_only: bool = False,
    ) -> torch.Tensor:
        """Return frame ids using the joint silence/text distribution."""
        if sample_silence_only and not sample:
            raise ValueError("sample_silence_only requires sample=True")
        if sample_silence_only:
            temperature = max(float(temperature), 1e-6)
            silence_probs = (silence_logits / temperature).softmax(dim=-1)
            silence_draw = torch.multinomial(silence_probs.reshape(-1, 2), num_samples=1).view(
                silence_logits.shape[:-1]
            )
            text_prediction = text_logits.argmax(dim=-1)
            silence_prediction = torch.full_like(text_prediction, self.silence_id)
            return torch.where(silence_draw == 0, silence_prediction, text_prediction)

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

    @staticmethod
    def _slice_feature_span(
        x: torch.Tensor,
        out_lengths: torch.Tensor,
        feature_start: int = 0,
        feature_length: Optional[int] = None,
    ):
        if feature_start < 0:
            raise ValueError("feature_start must be non-negative")
        if feature_length is not None and feature_length < 0:
            raise ValueError("feature_length must be non-negative")
        if feature_start:
            x = x[:, feature_start:]
            out_lengths = (out_lengths - feature_start).clamp(min=0)
        if feature_length is not None:
            x = x[:, :feature_length]
            out_lengths = out_lengths.clamp(max=feature_length)
        return x, out_lengths

    def forward_with_cache(
        self,
        audio_signal: torch.Tensor,
        length: Optional[torch.Tensor] = None,
        frame_targets: Optional[torch.Tensor] = None,
        cached_kvs: Optional[Iterable[Optional[torch.Tensor]]] = None,
        max_cache_length: Optional[int] = None,
        feature_start: int = 0,
        feature_length: Optional[int] = None,
        initial_frame_targets: Optional[torch.Tensor] = None,
        return_cache_slice: Optional[tuple] = None,
        detach_cache: bool = False,
        return_logits: bool = True,
    ) -> dict:
        """Teacher-forced forward pass that carries bounded decoder KV state."""
        if length is None:
            length = torch.full((audio_signal.size(0),), audio_signal.size(-1), device=audio_signal.device)
        if cached_kvs is None:
            cached_kvs = [None for _ in self.layers]
        else:
            cached_kvs = list(cached_kvs)
        if len(cached_kvs) != len(self.layers):
            raise ValueError("cached_kvs must have one entry per decoder layer")

        x = audio_signal.transpose(1, 2)
        x, out_lengths = self.subsampling(x, lengths=length)
        x, out_lengths = self._slice_feature_span(
            x=x,
            out_lengths=out_lengths,
            feature_start=feature_start,
            feature_length=feature_length,
        )
        prev_ids = self._previous_targets(
            frame_targets,
            x.size(0),
            x.size(1),
            x.device,
            initial_targets=initial_frame_targets,
        )
        x = x + self.prev_token_embedding(prev_ids)

        key_padding_mask = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1) >= out_lengths.unsqueeze(1)
        key_padding_mask = key_padding_mask if key_padding_mask.any() else None
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0)

        cached_len = 0
        for cache in cached_kvs:
            if cache is not None:
                cached_len = cache.size(1)
                break
        effective_kv_len = cached_len + x.size(1)
        if max_cache_length is not None and max_cache_length > 0:
            effective_kv_len = min(effective_kv_len, max_cache_length)
        q_offset = max(effective_kv_len - x.size(1), 0)
        rotary_emb_fn = self._rotary_emb_fn(effective_kv_len, x.device, q_offset=q_offset)

        next_caches = []
        for layer_idx, layer in enumerate(self.layers):
            x, next_cache = layer(
                x,
                rotary_emb_fn=rotary_emb_fn,
                cached_kv=cached_kvs[layer_idx],
                use_cache=True,
                max_cache_length=max_cache_length,
            )
            if return_cache_slice is not None:
                retain_start, retain_end = return_cache_slice
                retain_start = max(0, min(int(retain_start), x.size(1)))
                retain_end = max(retain_start, min(int(retain_end), x.size(1)))
                current_start = max(next_cache.size(1) - x.size(1), 0)
                next_cache = next_cache[:, current_start + retain_start : current_start + retain_end].contiguous()
            if detach_cache:
                next_cache = next_cache.detach()
            next_caches.append(next_cache)
        x = self.norm(x)
        silence_logits = self.silence_head(x)
        text_logits = self.text_head(x)
        logits = self._combined_logits(silence_logits, text_logits)
        output = {
            "logits": logits,
            "silence_logits": silence_logits,
            "text_logits": text_logits,
            "length": out_lengths,
            "cache": next_caches,
        }
        if not return_logits:
            output["final_posteriors"] = F.log_softmax(logits, dim=-1)
        return output

    def _calc_loss_from_output(self, out: dict, frame_targets: torch.Tensor) -> dict:
        """Compute the two-head training loss for frame-synchronous ASR targets."""
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

    def calc_loss(
        self,
        audio_signal: torch.Tensor,
        length: torch.Tensor,
        frame_targets: torch.Tensor,
    ) -> dict:
        """Compute the two-head training loss for frame-synchronous ASR targets."""
        out = self.forward(audio_signal=audio_signal, length=length, frame_targets=frame_targets, return_logits=True)
        return self._calc_loss_from_output(out, frame_targets)

    def calc_loss_with_cache(
        self,
        audio_signal: torch.Tensor,
        length: torch.Tensor,
        frame_targets: torch.Tensor,
        cached_kvs: Optional[Iterable[Optional[torch.Tensor]]] = None,
        max_cache_length: Optional[int] = None,
        feature_start: int = 0,
        feature_length: Optional[int] = None,
        initial_frame_targets: Optional[torch.Tensor] = None,
        return_cache_slice: Optional[tuple] = None,
        detach_cache: bool = False,
    ) -> dict:
        """Compute teacher-forced loss while carrying bounded decoder cache."""
        out = self.forward_with_cache(
            audio_signal=audio_signal,
            length=length,
            frame_targets=frame_targets,
            cached_kvs=cached_kvs,
            max_cache_length=max_cache_length,
            feature_start=feature_start,
            feature_length=feature_length,
            initial_frame_targets=initial_frame_targets,
            return_cache_slice=return_cache_slice,
            detach_cache=detach_cache,
            return_logits=True,
        )
        return self._calc_loss_from_output(out, frame_targets)

    def _prepare_decode_features(
        self,
        audio_signal: torch.Tensor,
        length: Optional[torch.Tensor] = None,
        max_frames: Optional[int] = None,
    ):
        if length is None:
            length = torch.full((audio_signal.size(0),), audio_signal.size(-1), device=audio_signal.device)

        x = audio_signal.transpose(1, 2)
        x, out_lengths = self.subsampling(x, lengths=length)
        if max_frames is not None and x.size(1) > max_frames:
            x = x[:, :max_frames]
            out_lengths = out_lengths.clamp(max=max_frames)

        key_padding_mask = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1) >= out_lengths.unsqueeze(1)
        key_padding_mask = key_padding_mask if key_padding_mask.any() else None
        return x, out_lengths, key_padding_mask

    def _effective_max_kv_cache_length(
        self,
        max_kv_cache_length: Optional[int] = None,
        max_kv_cache_spectrogram_length: Optional[int] = None,
    ) -> Optional[int]:
        if max_kv_cache_length is not None and max_kv_cache_spectrogram_length is not None:
            raise ValueError("Pass either max_kv_cache_length or max_kv_cache_spectrogram_length, not both")
        if max_kv_cache_spectrogram_length is None:
            return max_kv_cache_length
        return self.kv_cache_length_from_spectrogram_length(max_kv_cache_spectrogram_length)

    @staticmethod
    def _require_positive_int(value: int, name: str) -> int:
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _retain_current_chunk_cache(
        caches: Iterable[Optional[torch.Tensor]],
        current_chunk_cache_start: int,
        max_current_cache_length: int,
    ):
        retained = []
        for cache in caches:
            if cache is None:
                retained.append(None)
                continue
            retain_end = cache.size(1)
            retain_start = max(int(current_chunk_cache_start), retain_end - int(max_current_cache_length))
            retained.append(cache[:, retain_start:retain_end].contiguous())
        return retained

    def _prepare_chunked_decode_features(
        self,
        audio_signal: torch.Tensor,
        length: torch.Tensor,
        chunk_start: int,
        chunk_size: int,
        subsampling_history_length: int,
        final_flush_length: int = 0,
    ):
        total_frames = min(int(length[0].item()), audio_signal.size(-1))
        chunk_end = min(total_frames, int(chunk_start) + int(chunk_size))
        history_start = max(0, int(chunk_start) - int(subsampling_history_length))
        history_frames = int(chunk_start) - history_start
        segment = audio_signal[:, :, history_start:chunk_end]
        final_flush_length = max(0, int(final_flush_length))
        if final_flush_length:
            segment = F.pad(segment, (0, final_flush_length), value=0.0)
        segment_length = torch.tensor(
            [segment.size(-1)],
            dtype=length.dtype,
            device=length.device,
        )
        x, out_lengths = self.subsampling(segment.transpose(1, 2), lengths=segment_length)
        if history_frames > 0:
            history_length = torch.tensor([history_frames], dtype=length.dtype, device=length.device)
            history_output_len = int(self.output_lengths(history_length)[0].item())
            x = x[:, history_output_len:]
            out_lengths = (out_lengths - history_output_len).clamp(min=0)
        return x, out_lengths

    def _decode_with_chunked_kv_cache(
        self,
        audio_signal: torch.Tensor,
        length: torch.Tensor,
        raw_chunk_size: int,
        subsampling_history_length: int,
        decoder_history_length: int,
        max_output_frames: Optional[int],
        sample: bool,
        temperature: float,
        sample_silence_only: bool,
        max_kv_cache_length: Optional[int] = None,
        final_flush_spectrogram_length: Optional[int] = None,
    ) -> torch.Tensor:
        if audio_signal.size(0) != 1:
            raise ValueError("chunked KV-cache streaming decoding currently supports batch size 1")
        raw_chunk_size = self._require_positive_int(raw_chunk_size, "raw_chunk_size")
        subsampling_history_length = self._require_positive_int(
            subsampling_history_length,
            "subsampling_history_length",
        )
        decoder_history_length = self._require_positive_int(
            decoder_history_length,
            "decoder_history_length",
        )
        decoder_cache_cap = (
            int(max_kv_cache_length)
            if max_kv_cache_length is not None
            else self.kv_cache_length_from_spectrogram_length(decoder_history_length)
        )
        decoder_cache_cap = self._require_positive_int(decoder_cache_cap, "decoder_cache_cap")
        final_flush_spectrogram_length = 0 if final_flush_spectrogram_length is None else int(final_flush_spectrogram_length)
        if final_flush_spectrogram_length < 0:
            raise ValueError("final_flush_spectrogram_length must be non-negative")

        caches = [None for _ in self.layers]
        prev_id = torch.full((audio_signal.size(0),), self.silence_id, dtype=torch.long, device=audio_signal.device)
        predictions = []
        total_frames = min(int(length[0].item()), audio_signal.size(-1))
        max_output_frames = None if max_output_frames is None or max_output_frames <= 0 else int(max_output_frames)

        chunk_starts = list(range(0, total_frames, raw_chunk_size))
        if total_frames == 0 and final_flush_spectrogram_length > 0:
            chunk_starts = [0]

        for chunk_start in chunk_starts:
            is_final_chunk = int(chunk_start) + raw_chunk_size >= total_frames
            x, out_lengths = self._prepare_chunked_decode_features(
                audio_signal=audio_signal,
                length=length,
                chunk_start=chunk_start,
                chunk_size=raw_chunk_size,
                subsampling_history_length=subsampling_history_length,
                final_flush_length=final_flush_spectrogram_length if is_final_chunk else 0,
            )
            current_output_len = int(out_lengths[0].item())
            if max_output_frames is not None:
                remaining = max_output_frames - len(predictions)
                if remaining <= 0:
                    break
                current_output_len = min(current_output_len, remaining)
            if current_output_len <= 0:
                continue

            current_chunk_cache_start = 0 if caches[0] is None else caches[0].size(1)
            for step in range(current_output_len):
                h = x[:, step : step + 1] + self.prev_token_embedding(prev_id).unsqueeze(1)
                cached_len = 0 if caches[0] is None else caches[0].size(1)
                rotary_emb_fn = self._rotary_emb_fn(cached_len + 1, h.device, q_offset=cached_len)
                for layer_idx, layer in enumerate(self.layers):
                    h, caches[layer_idx] = layer(
                        h,
                        rotary_emb_fn=rotary_emb_fn,
                        cached_kv=caches[layer_idx],
                        use_cache=True,
                    )
                step_h = self.norm(h[:, 0])
                prev_id = self._step_predictions(
                    step_h,
                    sample=sample,
                    temperature=temperature,
                    sample_silence_only=sample_silence_only,
                )
                predictions.append(prev_id)

            caches = self._retain_current_chunk_cache(
                caches,
                current_chunk_cache_start=current_chunk_cache_start,
                max_current_cache_length=decoder_cache_cap,
            )

        if not predictions:
            return torch.empty((audio_signal.size(0), 0), dtype=torch.long, device=audio_signal.device)
        return torch.stack(predictions, dim=1)

    def _decode_with_kv_cache(
        self,
        x: torch.Tensor,
        max_cache_length: Optional[int],
        sample: bool,
        temperature: float,
        sample_silence_only: bool,
    ) -> torch.Tensor:
        if x.size(0) != 1:
            raise ValueError("KV-cache streaming decoding currently supports batch size 1")
        caches = [None for _ in self.layers]
        prev_id = torch.full((x.size(0),), self.silence_id, dtype=torch.long, device=x.device)
        predictions = []
        for step in range(x.size(1)):
            h = x[:, step : step + 1] + self.prev_token_embedding(prev_id).unsqueeze(1)
            cached_len = 0 if caches[0] is None else caches[0].size(1)
            effective_kv_len = cached_len + h.size(1)
            if max_cache_length is not None and max_cache_length > 0:
                effective_kv_len = min(effective_kv_len, max_cache_length)
            q_offset = max(effective_kv_len - h.size(1), 0)
            rotary_emb_fn = self._rotary_emb_fn(effective_kv_len, h.device, q_offset=q_offset)
            for layer_idx, layer in enumerate(self.layers):
                h, caches[layer_idx] = layer(
                    h,
                    rotary_emb_fn=rotary_emb_fn,
                    cached_kv=caches[layer_idx],
                    use_cache=True,
                    max_cache_length=max_cache_length,
                )
            step_h = self.norm(h[:, 0])
            prev_id = self._step_predictions(
                step_h,
                sample=sample,
                temperature=temperature,
                sample_silence_only=sample_silence_only,
            )
            predictions.append(prev_id)
        return torch.stack(predictions, dim=1)

    def _decode_with_full_context(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        sample: bool,
        temperature: float,
        sample_silence_only: bool,
    ) -> torch.Tensor:
        prev_ids = torch.full((x.size(0), x.size(1)), self.silence_id, dtype=torch.long, device=x.device)
        predictions = []

        for step in range(x.size(1)):
            h = x + self.prev_token_embedding(prev_ids)
            if key_padding_mask is not None:
                h = h.masked_fill(key_padding_mask.unsqueeze(-1), 0)
            rotary_emb_fn = self._rotary_emb_fn(h.size(1), h.device)
            for layer in self.layers:
                h = layer(h, rotary_emb_fn=rotary_emb_fn)
            step_h = self.norm(h[:, step])
            step_prediction = self._step_predictions(
                step_h,
                sample=sample,
                temperature=temperature,
                sample_silence_only=sample_silence_only,
            )
            predictions.append(step_prediction)
            if step + 1 < x.size(1):
                prev_ids[:, step + 1] = step_prediction
        return torch.stack(predictions, dim=1)

    @torch.no_grad()
    def _autoregressive_decode(
        self,
        audio_signal: torch.Tensor,
        length: Optional[torch.Tensor] = None,
        max_frames: Optional[int] = None,
        sample: bool = False,
        temperature: float = 1.0,
        use_kv_cache: bool = False,
        max_kv_cache_length: Optional[int] = None,
        max_kv_cache_spectrogram_length: Optional[int] = None,
        chunked_kv_cache: bool = False,
        kv_cache_chunk_spectrogram_length: Optional[int] = None,
        subsampling_history_spectrogram_length: Optional[int] = None,
        decoder_history_spectrogram_length: Optional[int] = None,
        final_flush_spectrogram_length: Optional[int] = None,
        sample_silence_only: bool = False,
    ) -> dict:
        """Shared inference-only autoregressive decode path for transcribe helpers."""
        was_training = self.training
        self.eval()
        try:
            if chunked_kv_cache:
                if not use_kv_cache:
                    raise ValueError("chunked_kv_cache requires use_kv_cache=True")
                raw_chunk_size = (
                    kv_cache_chunk_spectrogram_length
                    or max_kv_cache_spectrogram_length
                    or 2048
                )
                subsampling_history_length = subsampling_history_spectrogram_length or raw_chunk_size
                decoder_history_length = decoder_history_spectrogram_length or raw_chunk_size
                if length is None:
                    length = torch.full((audio_signal.size(0),), audio_signal.size(-1), device=audio_signal.device)
                predictions = self._decode_with_chunked_kv_cache(
                    audio_signal=audio_signal,
                    length=length,
                    raw_chunk_size=raw_chunk_size,
                    subsampling_history_length=subsampling_history_length,
                    decoder_history_length=decoder_history_length,
                    max_output_frames=max_frames,
                    sample=sample,
                    temperature=temperature,
                    sample_silence_only=sample_silence_only,
                    max_kv_cache_length=max_kv_cache_length,
                    final_flush_spectrogram_length=final_flush_spectrogram_length,
                )
                out_lengths = torch.full(
                    (audio_signal.size(0),),
                    predictions.size(1),
                    dtype=torch.long,
                    device=audio_signal.device,
                )
                return {"predictions": predictions, "length": out_lengths}

            x, out_lengths, key_padding_mask = self._prepare_decode_features(
                audio_signal=audio_signal,
                length=length,
                max_frames=max_frames,
            )
            if use_kv_cache:
                predictions = self._decode_with_kv_cache(
                    x=x,
                    max_cache_length=self._effective_max_kv_cache_length(
                        max_kv_cache_length=max_kv_cache_length,
                        max_kv_cache_spectrogram_length=max_kv_cache_spectrogram_length,
                    ),
                    sample=sample,
                    temperature=temperature,
                    sample_silence_only=sample_silence_only,
                )
            else:
                predictions = self._decode_with_full_context(
                    x=x,
                    key_padding_mask=key_padding_mask,
                    sample=sample,
                    temperature=temperature,
                    sample_silence_only=sample_silence_only,
                )
            return {"predictions": predictions, "length": out_lengths}
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def greedy_decode(
        self,
        audio_signal: torch.Tensor,
        length: Optional[torch.Tensor] = None,
        max_frames: Optional[int] = None,
        use_kv_cache: bool = False,
        max_kv_cache_length: Optional[int] = None,
        max_kv_cache_spectrogram_length: Optional[int] = None,
        chunked_kv_cache: bool = False,
        kv_cache_chunk_spectrogram_length: Optional[int] = None,
        subsampling_history_spectrogram_length: Optional[int] = None,
        decoder_history_spectrogram_length: Optional[int] = None,
        final_flush_spectrogram_length: Optional[int] = None,
    ) -> dict:
        """Autoregressively decode by greedy two-head prediction at each output frame."""
        return self._autoregressive_decode(
            audio_signal=audio_signal,
            length=length,
            max_frames=max_frames,
            sample=False,
            use_kv_cache=use_kv_cache,
            max_kv_cache_length=max_kv_cache_length,
            max_kv_cache_spectrogram_length=max_kv_cache_spectrogram_length,
            chunked_kv_cache=chunked_kv_cache,
            kv_cache_chunk_spectrogram_length=kv_cache_chunk_spectrogram_length,
            subsampling_history_spectrogram_length=subsampling_history_spectrogram_length,
            decoder_history_spectrogram_length=decoder_history_spectrogram_length,
            final_flush_spectrogram_length=final_flush_spectrogram_length,
        )

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
        chunked_kv_cache: bool = False,
        kv_cache_chunk_spectrogram_length: Optional[int] = None,
        subsampling_history_spectrogram_length: Optional[int] = None,
        decoder_history_spectrogram_length: Optional[int] = None,
        final_flush_spectrogram_length: Optional[int] = None,
        sample_silence_only: bool = False,
    ) -> dict:
        """Autoregressively decode while sampling frame ids."""
        return self._autoregressive_decode(
            audio_signal=audio_signal,
            length=length,
            max_frames=max_frames,
            sample=True,
            temperature=temperature,
            use_kv_cache=use_kv_cache,
            max_kv_cache_length=max_kv_cache_length,
            max_kv_cache_spectrogram_length=max_kv_cache_spectrogram_length,
            chunked_kv_cache=chunked_kv_cache,
            kv_cache_chunk_spectrogram_length=kv_cache_chunk_spectrogram_length,
            subsampling_history_spectrogram_length=subsampling_history_spectrogram_length,
            decoder_history_spectrogram_length=decoder_history_spectrogram_length,
            final_flush_spectrogram_length=final_flush_spectrogram_length,
            sample_silence_only=sample_silence_only,
        )

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
        chunked_kv_cache: bool = False,
        kv_cache_chunk_spectrogram_length: Optional[int] = None,
        subsampling_history_spectrogram_length: Optional[int] = None,
        decoder_history_spectrogram_length: Optional[int] = None,
        final_flush_spectrogram_length: Optional[int] = None,
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
                    chunked_kv_cache=chunked_kv_cache,
                    kv_cache_chunk_spectrogram_length=kv_cache_chunk_spectrogram_length,
                    subsampling_history_spectrogram_length=subsampling_history_spectrogram_length,
                    decoder_history_spectrogram_length=decoder_history_spectrogram_length,
                    final_flush_spectrogram_length=final_flush_spectrogram_length,
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

        if decode_mode in {"sample", "sample_silence_greedy_text"}:
            decoded = self.sample_decode(
                audio_signal=audio_signal,
                length=length,
                max_frames=max_output_frames,
                temperature=temperature,
                use_kv_cache=use_kv_cache,
                max_kv_cache_length=max_kv_cache_length,
                max_kv_cache_spectrogram_length=max_kv_cache_spectrogram_length,
                chunked_kv_cache=chunked_kv_cache,
                kv_cache_chunk_spectrogram_length=kv_cache_chunk_spectrogram_length,
                subsampling_history_spectrogram_length=subsampling_history_spectrogram_length,
                decoder_history_spectrogram_length=decoder_history_spectrogram_length,
                final_flush_spectrogram_length=final_flush_spectrogram_length,
                sample_silence_only=decode_mode == "sample_silence_greedy_text",
            )
        elif decode_mode == "greedy":
            decoded = self.greedy_decode(
                audio_signal=audio_signal,
                length=length,
                max_frames=max_output_frames,
                use_kv_cache=use_kv_cache,
                max_kv_cache_length=max_kv_cache_length,
                max_kv_cache_spectrogram_length=max_kv_cache_spectrogram_length,
                chunked_kv_cache=chunked_kv_cache,
                kv_cache_chunk_spectrogram_length=kv_cache_chunk_spectrogram_length,
                subsampling_history_spectrogram_length=subsampling_history_spectrogram_length,
                decoder_history_spectrogram_length=decoder_history_spectrogram_length,
                final_flush_spectrogram_length=final_flush_spectrogram_length,
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
