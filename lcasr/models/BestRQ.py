import torch, torch.nn as nn, torch.nn.functional as F

from einops import rearrange, repeat
from typing import Optional

from lcasr.utils.helpers import exists

try: from apex.normalization import FusedLayerNorm as LayerNorm
except:
    from torch.nn import LayerNorm as LayerNorm

from lcasr.models.base import BaseModel
from lcasr.models.sconformer_xl import SCConformerXL
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pad_feats(feats, divis_by): # pad from: https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriSpeech/self-supervised-learning/BEST-RQ/train.py
    """BEST-RQ quantizer stackes frames together. Hence, we need to pad the
    incoming features such that the time dimension is divisible by divis_by.

    Arguments
    ---------
    feats: torch.Tensor
        The feature tensor.
    divis_by: int
        The stacking factor. The time dimension of feats will become divisible
        by this value.

    Returns
    -------
    Padded features
    """

    B, T, C = feats.shape

    #### pad features to enable a reduction by pad_to_divisible_by for the
    # quantiser of BEST-RQ
    current_dim_size = T
    dim_to_pad = 1  # Pad along the second dimension (i.e. time)

    # Calculate the amount of padding needed to make the tensor divisible
    # by divis_by
    current_dim_size = feats.shape[dim_to_pad]
    # Ensure positive padding
    padding_needed = (divis_by - (current_dim_size % divis_by)) % divis_by

    # Define the padding
    # Initialize padding for all dimensions, have a look at the documentation of
    # torch.nn.functional.pad because the padding argument is quite special.
    padding = [0, 0, 0, 0, 0, 0]
    padding[dim_to_pad * 2 + 1] = (
        padding_needed  # Set padding for the chosen dimension
    )

    # add in padding to features and mask
    return torch.nn.functional.pad(feats, padding)

class BestRQ(BaseModel):
    def __init__(
        self,
        model: SCConformerXL,
        mask_percentage: float = 0.1,
        frames_to_mask: int = 5,
        mask_mode: str = "legacy_groups",
        mask_prob: Optional[float] = None,
        mask_length: Optional[int] = None,
        downsampling_factor: int = 8,
        codebook_size: int = 8192,
        codebook_dim: int = 16,
        **kwargs
    ):
        super().__init__()
        self.downsampling_factor = downsampling_factor
        self.mask_percentage = mask_percentage
        self.frames_to_mask = frames_to_mask
        self.mask_mode = mask_mode
        self.mask_prob = mask_percentage if mask_prob is None else mask_prob
        self.mask_length = frames_to_mask if mask_length is None else mask_length


        self.out_projection = nn.Sequential(
            nn.LayerNorm(model.d_model),
            nn.Linear(model.d_model, codebook_size)
        )


        from vector_quantize_pytorch import RandomProjectionQuantizer
        self.quantizer = RandomProjectionQuantizer(
            dim = model.feat_in * downsampling_factor,  # input dimension -- 8 * 80 i.e 640
            num_codebooks = 1,
            codebook_dim = codebook_dim,
            codebook_size = codebook_size
        )
        self.model = model

    def _select_legacy_group_mask(self, B:int, T:int, device=None) -> torch.Tensor:
        frames_to_mask = self.frames_to_mask
        masking_percentage = self.mask_percentage
        n_masks = T // frames_to_mask
        has_remainder = int((T % frames_to_mask) != 0)
        n_masks += has_remainder

        device_kwargs = {} if device is None else {'device': device}
        probs = torch.rand(B, n_masks, **device_kwargs)
        mask = probs < masking_percentage
        mask = repeat(mask, 'b t -> b (t f)', f=frames_to_mask)[:, :T]
        return mask

    def _select_speechbrain_mask(
            self,
            B:int,
            T:int,
            valid_stacked: Optional[torch.Tensor] = None,
            device = None,
        ) -> torch.Tensor:
        device_kwargs = {} if device is None else {'device': device}
        mask = torch.zeros(B, T, dtype=torch.bool, **device_kwargs)
        if B == 0 or T == 0:
            return mask

        if valid_stacked is None:
            sample_lens = torch.full((B,), T, dtype=torch.long, **device_kwargs)
        else:
            sample_lens = (valid_stacked.to(device) if device is not None else valid_stacked).long().sum(dim=1)

        mask_length = int(self.mask_length)
        if mask_length <= 0:
            raise ValueError(f"mask_length must be positive, got {mask_length}")

        selected_offsets = torch.arange(mask_length, **device_kwargs)
        for batch_idx, sample_len_tensor in enumerate(sample_lens):
            sample_len = int(sample_len_tensor.item())
            if sample_len <= 0:
                continue

            num_blocks = sample_len // mask_length
            if num_blocks <= 0:
                continue

            num_mask = int(float(self.mask_prob) * sample_len + torch.rand((), **device_kwargs).item())
            num_mask = max(1, min(num_mask, num_blocks))

            selected_blocks = torch.randperm(num_blocks, **device_kwargs)[:num_mask] * mask_length
            selected_indices = (selected_blocks[:, None] + selected_offsets[None, :]).reshape(-1)
            selected_indices = selected_indices[selected_indices < sample_len]
            mask[batch_idx, selected_indices] = True
        return mask

    def select_mask(
            self,
            B:int,
            T:int,
            valid_stacked: Optional[torch.Tensor] = None,
            device = None,
        ) -> torch.Tensor:
        if self.mask_mode in {"legacy", "legacy_groups", "group", "groups"}:
            return self._select_legacy_group_mask(B=B, T=T, device=device)
        if self.mask_mode in {"speechbrain", "paper", "paper_start", "paper_style"}:
            return self._select_speechbrain_mask(
                B=B,
                T=T,
                valid_stacked=valid_stacked,
                device=device,
            )
        raise ValueError(f"Unknown BEST-RQ mask_mode: {self.mask_mode}")


    def calc_loss(self, x, targets) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(x, target=targets)

    def forward(
            self,
            audio_signal,
            length = None,
        ):
        '''
        audio_signal: (batch_size, feat, time) - mel spectrogram see lcasr.utils.audio_tools for how it is computed
        length: (batch_size,)
        '''
        if length is None:
            length = torch.full(
                (audio_signal.shape[0],),
                audio_signal.shape[-1],
                dtype=torch.long,
                device=audio_signal.device,
            )

        audio_signal = rearrange(audio_signal, 'b c t -> b t c')
        device = audio_signal.device
        length = length.to(device)
        audio_signal = pad_feats(audio_signal, self.downsampling_factor)

        B, T, C = audio_signal.shape
        ds = self.downsampling_factor
        T_stacked = T // ds

        stacked_signal = audio_signal.reshape(B, T_stacked, C * self.downsampling_factor)

        # --- validity mask from lengths (assuming length is in mel frames) ---
        stacked_lengths = torch.div(length, ds, rounding_mode="floor").clamp(max=T_stacked)
        valid_stacked = torch.arange(T_stacked, device=device)[None, :] < stacked_lengths[:, None]


        stacked_mask = self.select_mask(
            B,
            T // self.downsampling_factor,
            valid_stacked=valid_stacked,
            device=device,
        ).to(device)
        stacked_mask = stacked_mask & valid_stacked

        num_valid_stacked_frames = int(valid_stacked.sum().item())
        num_masked_stacked_frames = int(stacked_mask.sum().item())
        mask_diagnostics = {
            'mask_mode': self.mask_mode,
            'valid_stacked_frames': num_valid_stacked_frames,
            'masked_stacked_frames': num_masked_stacked_frames,
            'actual_mask_ratio': (
                num_masked_stacked_frames / num_valid_stacked_frames
                if num_valid_stacked_frames > 0 else 0.0
            ),
            'skipped_empty_mask': 0,
        }

        if not valid_stacked.any():
            logging.warning("no valid stacked BEST-RQ frames, skipping loss")
            mask_diagnostics['skipped_empty_mask'] = 1
            return {'loss': None, 'num_masked_frames': 0, **mask_diagnostics}

        if not stacked_mask.any():
            logging.warning("no masked BEST-RQ frames selected, skipping loss")
            mask_diagnostics['skipped_empty_mask'] = 1
            return {'loss': None, 'num_masked_frames': 0, **mask_diagnostics}

        target_frames = stacked_signal[stacked_mask]  # (num_masked_frames, C * downsampling_factor)
        if target_frames.shape[0] == 0:
            logging.warning("no masked BEST-RQ frames selected, skipping loss")
            mask_diagnostics['skipped_empty_mask'] = 1
            return {'loss': None, 'num_masked_frames': 0, **mask_diagnostics}

        targets = self.quantizer(target_frames[None]).reshape(-1).long() # (num_masked_frames,)

        mask = repeat(stacked_mask, 'b t -> b (t f)', f=self.downsampling_factor)
        assert mask.shape == (B, T), f"Something went wrong, got mask shape {mask.shape}, expected {(B,T)}"

        mask_num = int(mask.sum().item())
        if mask_num > 0:
            audio_signal[mask] = torch.normal(
                mean=0.0,
                std=0.1,
                size=(mask_num, C),
                device=device,
            )
        audio_signal = rearrange(audio_signal, 'b t c -> b c t')
        out = self.model(
            audio_signal = audio_signal,
            length = length,
            skip_vocab_projection = True,
        )
        hidden_stated = out["hidden_states"]


        x = self.out_projection(hidden_stated)

        x_tgt = x[stacked_mask]  # (num_masked_frames, codebook_size)

        loss = self.calc_loss(x_tgt, targets)


        return {'loss': loss, 'num_masked_frames': int(targets.numel()), **mask_diagnostics}



if __name__ == '__main__':
    # run test
    model = SCConformerXL(vocab_size=4096, head_dim=256, n_heads=3)
    bestrq = BestRQ(model=model)

    audio = torch.randn(2, 80, 1000)
    lengths = torch.tensor([1000, 500])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    bestrq = bestrq.to(device)
    audio = audio.to(device)
    lengths = lengths.to(device)
    out = bestrq(audio, length=lengths)
    logger.info(out['loss'])
