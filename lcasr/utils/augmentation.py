'''
This file contains augmentation functions i.e specaugment
'''

import torch, torchaudio.functional as F
from torch import Tensor


class SpecAugment(torch.nn.Module): # taken from https://pytorch.org/audio/main/_modules/torchaudio/transforms/_transforms.html#FrequencyMasking
    r"""Apply time and frequency masking to a spectrogram.
    Args:
        n_time_masks (int): Number of time masks. If its value is zero, no time masking will be applied.
        time_mask_param (int): Maximum possible length of the time mask.
        n_freq_masks (int): Number of frequency masks. If its value is zero, no frequency masking will be applied.
        freq_mask_param (int): Maximum possible length of the frequency mask.
        iid_masks (bool, optional): Applies iid masks to each of the examples in the batch dimension.
            This option is applicable only when the input tensor is 4D. (Default: ``True``)
        p (float, optional): maximum proportion of time steps that can be masked.
            Must be within range [0.0, 1.0]. (Default: 1.0)
        zero_masking (bool, optional): If ``True``, use 0 as the mask value,
            else use mean of the input tensor. (Default: ``False``)
    """
    __constants__ = [
        "n_time_masks",
        "time_mask_param",
        "n_freq_masks",
        "freq_mask_param",
        "iid_masks",
        "p",
        "zero_masking",
    ]

    def __init__(
        self,
        n_time_masks: int,
        time_mask_param: int,
        n_freq_masks: int,
        freq_mask_param: int,
        iid_masks: bool = True,
        p: float = 1.0,
        zero_masking: bool = False,
    ) -> None:
        super(SpecAugment, self).__init__()
        self.n_time_masks = n_time_masks
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.freq_mask_param = freq_mask_param
        self.iid_masks = iid_masks
        self.p = p
        self.zero_masking = zero_masking

    def forward(self, specgram: Tensor, audio_lengths: Tensor = None) -> Tensor:
        r"""
        Args:
            specgram (Tensor): Tensor of shape `(..., freq, time)`.
            audio_lengths (Tensor): Tensor of shape `(batch)` containing sizes of each spectrogram in the batch. (Used when zero_masking is False to calculate mean of the spectrogram without masked values.)
        Returns:
            Tensor: Masked spectrogram of shape `(..., freq, time)`.
        """
        f, t = specgram.shape[-2:]
        if self.zero_masking:
            mask_value = 0.0
        else:
            mask_value = specgram.mean() if audio_lengths is None else specgram[(torch.arange(t, device=specgram.device)[None, :] < audio_lengths[..., None]).unsqueeze(-2).repeat(*[1] * (specgram.dim() - 2), f, 1)].mean() # mean of the spectrogram without masked values

        time_dim = specgram.dim() - 1
        freq_dim = time_dim - 1
        

        if specgram.dim() > 2 and self.iid_masks is True:
            missing_channel = specgram.dim() == 3
            if missing_channel:
                specgram = specgram.unsqueeze(1) # (batch, 1, freq, time) 1 is channel
                time_dim, freq_dim = time_dim + 1, freq_dim + 1
            for _ in range(self.n_time_masks):
                specgram = F.mask_along_axis_iid(specgram, self.time_mask_param, mask_value, time_dim, p=self.p)
            for _ in range(self.n_freq_masks):
                specgram = F.mask_along_axis_iid(specgram, self.freq_mask_param, mask_value, freq_dim, p=self.p)
            if missing_channel:
                specgram = specgram.squeeze(1)
        else:
            for _ in range(self.n_time_masks):
                specgram = F.mask_along_axis(specgram, self.time_mask_param, mask_value, time_dim, p=self.p)
            for _ in range(self.n_freq_masks):
                specgram = F.mask_along_axis(specgram, self.freq_mask_param, mask_value, freq_dim, p=self.p)

        return specgram
