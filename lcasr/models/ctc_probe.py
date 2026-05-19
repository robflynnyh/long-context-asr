import torch
import torch.nn as nn
import torch.nn.functional as F

from lcasr.components.decoder import ASRLinearSCDecoder
from lcasr.models.base import LayerNorm, RMSNorm


class BiLSTMCTCProbeHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        norm: bool = False,
        norm_fn=LayerNorm,
        hidden_size: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = vocab_size + 1
        self.norm = norm_fn(d_model) if norm else nn.Identity()
        dropout = dropout if num_layers > 1 else 0.0
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.ff = nn.Linear(hidden_size * 2, self.num_classes)
        self.reprojection = nn.Linear(self.num_classes, d_model)

    def forward(self, x, logits=False):
        x_norm = self.norm(x)
        with torch.cuda.amp.autocast(enabled=False):
            x, _ = self.bilstm(x_norm.float())
            x = self.ff(x)
        return x if logits else F.log_softmax(x, dim=-1)

    def project_back(self, x):
        return self.reprojection(x)

    def integrate_projections(self, x, proj1):
        return x + proj1


class FrozenBackboneCTCProbe(nn.Module):
    def __init__(self, acoustic_model: nn.Module, decoder: nn.Module):
        super().__init__()
        self.acoustic_model = acoustic_model
        self.decoder = decoder

    @property
    def subsampling(self):
        return self.acoustic_model.subsampling

    def print_total_params(self, only_trainable=False):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad) if only_trainable else sum(p.numel() for p in self.parameters())
        pstr = "Total trainable params: " if only_trainable else "Total params: "
        print(f"{pstr}: ", total / 1e6, "M")
        return total

    def forward(self, *args, **kwargs):
        return_logits = kwargs.get("return_logits", False)
        kwargs["skip_vocab_projection"] = True
        output = self.acoustic_model(*args, **kwargs)
        hidden_states = output["hidden_states"]
        if getattr(self.acoustic_model, "legasee_double_norm", False):
            hidden_states = self.decoder.norm(hidden_states)
        final_posts = self.decoder(x=hidden_states, logits=return_logits)
        return {"final_posteriors": final_posts, "length": output["length"]}

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(normalize_probe_state_dict(self, state_dict), strict=strict)


def _norm_fn(config):
    return RMSNorm if config.model.get("default_norm", "layer_norm") == "rms_norm" else LayerNorm


def wrap_model_with_ctc_probe(config, acoustic_model: nn.Module, vocab_size: int):
    if not config.get("probe", {}):
        return acoustic_model

    probe_head = config.probe.get("head", "linear")
    if probe_head == "linear":
        head = ASRLinearSCDecoder(
            d_model=config.model.d_model,
            vocab_size=vocab_size,
            norm=config.model.get("decoder_norm", False),
            norm_fn=_norm_fn(config),
        )
    elif probe_head == "bilstm":
        head = BiLSTMCTCProbeHead(
            d_model=config.model.d_model,
            vocab_size=vocab_size,
            norm=config.model.get("decoder_norm", False),
            norm_fn=_norm_fn(config),
            hidden_size=config.probe.get("bilstm_hidden_size", 1024),
            num_layers=config.probe.get("bilstm_num_layers", 2),
            dropout=config.probe.get("bilstm_dropout", 0.2),
        )
    else:
        raise NotImplementedError(f"unknown CTC probe head: {probe_head}")
    return FrozenBackboneCTCProbe(acoustic_model=acoustic_model, decoder=head)


def acoustic_model_from_probe(model: nn.Module):
    return model.acoustic_model if isinstance(model, FrozenBackboneCTCProbe) else model


def normalize_probe_state_dict(model: nn.Module, state_dict):
    if not isinstance(model, FrozenBackboneCTCProbe):
        return state_dict
    if any(key.startswith("acoustic_model.") for key in state_dict):
        return state_dict
    has_decoder = any(key.startswith("decoder.") for key in state_dict)
    has_final_decoder = any(key.startswith("final_decoder.") for key in state_dict)
    if not has_decoder and not has_final_decoder:
        return state_dict

    converted = {}
    for key, value in state_dict.items():
        if key.startswith("final_decoder."):
            converted[f"decoder.{key[len('final_decoder.'):]}"] = value
        elif key.startswith("decoder."):
            if has_final_decoder:
                converted[f"acoustic_model.{key}"] = value
            else:
                converted[key] = value
                converted[f"acoustic_model.{key}"] = value
        else:
            converted[f"acoustic_model.{key}"] = value
    return converted
