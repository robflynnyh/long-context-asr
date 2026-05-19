import torch.nn as nn, torch.nn.functional as F, torch
try: from apex.normalization import FusedRMSNorm as DEFAULT_NORM
except: from lcasr.components.normalisation import RMSNorm as DEFAULT_NORM
from einops import rearrange

class ASRLinearSCDecoder(nn.Module):
    def __init__(
            self, 
            d_model, 
            vocab_size, 
            norm=False, 
            norm_fn=DEFAULT_NORM,
            **kwargs
        ):
        super().__init__()
        # Add 1 for blank char
        self.num_classes = vocab_size + 1
        self.ff = nn.Linear(d_model, self.num_classes)
        self.reprojection = nn.Linear(self.num_classes, d_model)
        self.norm = norm_fn(d_model) if norm else nn.Identity()

    def forward(self, x, logits=False):
        x_norm = self.norm(x)
        x = self.ff(x_norm)
        x = F.log_softmax(x, dim=-1) if not logits else x
        return x        

    def project_back(self, x):
        return self.reprojection(x)

    def integrate_projections(self, x, proj1):
        return x + proj1


class BiLSTMCTCDecoder(nn.Module):
    def __init__(
            self,
            d_model,
            vocab_size,
            norm=False,
            norm_fn=DEFAULT_NORM,
            bilstm_hidden_size=1024,
            bilstm_num_layers=2,
            bilstm_dropout=0.2,
            **kwargs,
        ):
        super().__init__()
        self.num_classes = vocab_size + 1
        self.norm = norm_fn(d_model) if norm else nn.Identity()
        dropout = bilstm_dropout if bilstm_num_layers > 1 else 0.0
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=bilstm_hidden_size,
            num_layers=bilstm_num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.ff = nn.Linear(bilstm_hidden_size * 2, self.num_classes)
        self.reprojection = nn.Linear(self.num_classes, d_model)

    def forward(self, x, logits=False):
        x_norm = self.norm(x)
        with torch.cuda.amp.autocast(enabled=False):
            x, _ = self.bilstm(x_norm.float())
            x = self.ff(x)
        x = F.log_softmax(x, dim=-1) if not logits else x
        return x

    def project_back(self, x):
        return self.reprojection(x)

    def integrate_projections(self, x, proj1):
        return x + proj1


def _linear_final_decoder(**kwargs):
    return None


FINAL_CTC_DECODERS = {
    'linear': _linear_final_decoder,
    'bilstm': BiLSTMCTCDecoder,
}

FINAL_DECODER_ARG_ALIASES = {
    'final_decoder_bilstm_hidden_size': 'bilstm_hidden_size',
    'final_decoder_bilstm_num_layers': 'bilstm_num_layers',
    'final_decoder_bilstm_dropout': 'bilstm_dropout',
}


def build_final_ctc_decoder(decoder_type='linear', **kwargs):
    try:
        decoder_cls = FINAL_CTC_DECODERS[decoder_type]
    except KeyError as exc:
        choices = ', '.join(sorted(FINAL_CTC_DECODERS))
        raise ValueError(f'Unknown final_decoder_type {decoder_type}; expected one of {choices}') from exc

    decoder_kwargs = dict(kwargs)
    for source, target in FINAL_DECODER_ARG_ALIASES.items():
        if source in decoder_kwargs:
            decoder_kwargs[target] = decoder_kwargs[source]
    return decoder_cls(**decoder_kwargs)
