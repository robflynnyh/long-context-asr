'''
This file contains augmentation functions i.e specaugment
'''

import torch, torchaudio.functional as F, torch.nn as nn
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
        n_freq_masks: int,
        freq_mask_param: int,
        iid_masks: bool = True,
        time_mask_param: int = -1, # time mask width if -1 then it is calculated from min_p
        min_p: float = -1, # minimun amount of spectogram to be masked, only use when time_mask_param is -1
        max_p: float = 1.0,
        zero_masking: bool = False, # set to true if spectogram is normalized per input i.e then it is equal to mean of the spectogram
        **kwargs,

    ) -> None:
        super(SpecAugment, self).__init__()
        assert min_p != -1 or time_mask_param != -1, "Either min_p or n_time_masks must be set o:"
        assert min_p == -1 or (min_p >= 0 and min_p <= 1), "min_p must be within range [0.0, 1.0]"
        assert max_p >= 0 and max_p <= 1, "max_p must be within range [0.0, 1.0]"

        self.n_time_masks = n_time_masks
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.freq_mask_param = freq_mask_param
        self.iid_masks = iid_masks
        self.max_p = max_p
        self.zero_masking = zero_masking
        self.min_p = min_p

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
        
        time_mask_width, num_time_masks = self.time_mask_param, self.n_time_masks
        if self.min_p != -1:
            total_time_mask_coverage = int(t * self.min_p) # calculate time_mask_width from min_p
            time_mask_width = int(total_time_mask_coverage / num_time_masks) if num_time_masks != 0 else 0 # calculate number of time mask width so that total time mask coverage is equal to min_p   

        if specgram.dim() > 2 and self.iid_masks is True:
            missing_channel = specgram.dim() == 3
            if missing_channel:
                specgram = specgram.unsqueeze(1) # (batch, 1, freq, time) 1 is channel
                time_dim, freq_dim = time_dim + 1, freq_dim + 1
            for _ in range(num_time_masks):
                specgram = F.mask_along_axis_iid(specgram, time_mask_width, mask_value, time_dim, p=self.max_p)
            for _ in range(self.n_freq_masks):
                specgram = F.mask_along_axis_iid(specgram, self.freq_mask_param, mask_value, freq_dim, p=self.max_p)
            if missing_channel:
                specgram = specgram.squeeze(1)
        else:
            for _ in range(num_time_masks):
                specgram = F.mask_along_axis(specgram, time_mask_width, mask_value, time_dim, p=self.max_p)
            for _ in range(self.n_freq_masks):
                specgram = F.mask_along_axis(specgram, self.freq_mask_param, mask_value, freq_dim, p=self.max_p)

        return specgram



# class SpectrogramAugmentation(nn.Module): nvidia's implementation
#     """
#     Performs time and freq cuts in one of two ways.
#     SpecAugment zeroes out vertical and horizontal sections as described in
#     SpecAugment (https://arxiv.org/abs/1904.08779). Arguments for use with
#     SpecAugment are `freq_masks`, `time_masks`, `freq_width`, and `time_width`.
#     SpecCutout zeroes out rectangulars as described in Cutout
#     (https://arxiv.org/abs/1708.04552). Arguments for use with Cutout are
#     `rect_masks`, `rect_freq`, and `rect_time`.

#     Args:
#         freq_masks (int): how many frequency segments should be cut.
#             Defaults to 0.
#         time_masks (int): how many time segments should be cut
#             Defaults to 0.
#         freq_width (int): maximum number of frequencies to be cut in one
#             segment.
#             Defaults to 10.
#         time_width (int): maximum number of time steps to be cut in one
#             segment
#             Defaults to 10.
#         rect_masks (int): how many rectangular masks should be cut
#             Defaults to 0.
#         rect_freq (int): maximum size of cut rectangles along the frequency
#             dimension
#             Defaults to 5.
#         rect_time (int): maximum size of cut rectangles along the time
#             dimension
#             Defaults to 25.
#     """

#     def __init__(
#         self,
#         freq_masks=0,
#         time_masks=0,
#         freq_width=10,
#         time_width=10,
#         rect_masks=0,
#         rect_time=5,
#         rect_freq=20,
#         rng=None,
#         mask_value=0.0,
#         use_numba_spec_augment: bool = True,
#     ):
#         super().__init__()

#         if rect_masks > 0:
#             self.spec_cutout = SpecCutout(rect_masks=rect_masks, rect_time=rect_time, rect_freq=rect_freq, rng=rng,)
#             # self.spec_cutout.to(self._device)
#         else:
#             self.spec_cutout = lambda input_spec: input_spec
#         if freq_masks + time_masks > 0:
#             self.spec_augment = SpecAugment(
#                 freq_masks=freq_masks,
#                 time_masks=time_masks,
#                 freq_width=freq_width,
#                 time_width=time_width,
#                 rng=rng,
#                 mask_value=mask_value,
#             )
#         else:
#             self.spec_augment = lambda input_spec, length: input_spec

#         # Check if numba is supported, and use a Numba kernel if it is
#         if use_numba_spec_augment and numba_utils.numba_cuda_is_supported(__NUMBA_MINIMUM_VERSION__):
#             logging.info('Numba CUDA SpecAugment kernel is being used')
#             self.spec_augment_numba = SpecAugmentNumba(
#                 freq_masks=freq_masks,
#                 time_masks=time_masks,
#                 freq_width=freq_width,
#                 time_width=time_width,
#                 rng=rng,
#                 mask_value=mask_value,
#             )
#         else:
#             self.spec_augment_numba = None

#     def forward(self, input_spec, length):
#         augmented_spec = self.spec_cutout(input_spec=input_spec)

#         # To run the Numba kernel, correct numba version is required as well as
#         # tensor must be on GPU and length must be provided
#         if self.spec_augment_numba is not None and spec_augment_launch_heuristics(augmented_spec, length):
#             augmented_spec = self.spec_augment_numba(input_spec=augmented_spec, length=length)
#         else:
#             augmented_spec = self.spec_augment(input_spec=augmented_spec, length=length)
#         return augmented_spec