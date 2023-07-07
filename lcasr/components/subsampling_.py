# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
from einops import rearrange
import torch.nn as nn

class StackingSubsampling(torch.nn.Module):
    """Stacking subsampling which simply stacks consecutive frames to reduce the sampling rate
    Args:
        subsampling_factor (int): The subsampling factor
        feat_in (int): size of the input features
        feat_out (int): size of the output features
    """

    def __init__(self, subsampling_factor, feat_in, feat_out):
        super(StackingSubsampling, self).__init__()
        self.subsampling_factor = subsampling_factor
        self.proj_out = torch.nn.Linear(subsampling_factor * feat_in, feat_out)

    def forward(self, x, lengths):
        b, t, h = x.size()
        pad_size = (self.subsampling_factor - (t % self.subsampling_factor)) % self.subsampling_factor
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
        _, t, _ = x.size()
        x = torch.reshape(x, (b, t // self.subsampling_factor, h * self.subsampling_factor))
        x = self.proj_out(x)
        lengths = torch.div(lengths + pad_size, self.subsampling_factor, rounding_mode='floor')
        return x, lengths

class DepthwiseSeparable(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DepthwiseSeparable, self).__init__()
        exp_f = 1
        intermid_c = int(out_channels*exp_f)
        self.pointwise_in = torch.nn.Conv1d(in_channels, intermid_c, kernel_size=1, stride=1, padding=0)
        self.depthwise = torch.nn.Conv1d(intermid_c, intermid_c, kernel_size, stride=stride, padding=padding, groups=intermid_c)
        self.pointwise = torch.nn.Conv1d(intermid_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU() # hmm

    def forward(self, x):
        x = self.pointwise_in(x)
        x = self.act(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ConvSubsampling(torch.nn.Module):
    """Convolutional subsampling which supports VGGNet and striding approach introduced in:
    VGGNet Subsampling: Transformer-transducer: end-to-end speech recognition with self-attention (https://arxiv.org/pdf/1910.12977.pdf)
    Striding Subsampling: "Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition" by Linhao Dong et al. (https://ieeexplore.ieee.org/document/8462506)
    Args:
        subsampling (str): The subsampling technique from {"vggnet", "striding"}
        subsampling_factor (int): The subsampling factor which should be a power of 2
        feat_in (int): size of the input features
        feat_out (int): size of the output features
        conv_channels (int): Number of channels for the convolution layers.
        activation (Module): activation function, default is nn.ReLU()
    """

    def __init__(self, feat_in, feat_out, conv_channels, activation=nn.ReLU(), stride=2, kernel_size=3):
        super(ConvSubsampling, self).__init__()
        subsampling, subsampling_factor = 'striding', 4
        self._subsampling = subsampling

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))

        in_channels = 1
        layers = []
        self._ceil_mode = False

        if subsampling == 'vggnet':
            self._padding = 0
            self._stride = stride
            self._kernel_size = kernel_size
            self._ceil_mode = True

            for i in range(self._sampling_num):
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1
                    )
                )
                layers.append(activation)
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1
                    )
                )
                layers.append(activation)
                layers.append(
                    torch.nn.MaxPool2d(
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._padding,
                        ceil_mode=self._ceil_mode,
                    )
                )
                in_channels = conv_channels
        elif subsampling == 'striding':
            self._padding = 1
            self._stride = stride
            self._kernel_size = kernel_size
            self._ceil_mode = False

            for i in range(self._sampling_num):
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._padding,
                    )
                )
                layers.append(activation)
                in_channels = conv_channels
        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        in_length = torch.tensor(feat_in, dtype=torch.float)
        out_length = calc_length(
            in_length,
            padding=self._padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x, lengths):
        lengths = calc_length(
            lengths,
            padding=self._padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        
        x = x.unsqueeze(1)
        if self._subsampling == 'striding':
            # added in order to prevent slowdown in torch.nn.Conv2d with bfloat16 / CUDNN v8 API
            # to be removed once the above is fixed in cudnn
            
            #with torch.cuda.amp.autocast(dtype=torch.float32):
            x = self.conv(x)
        else:
            x = self.conv(x)

        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        return x, lengths

class ConvSubsampling_(torch.nn.Module):
    """Convolutional subsampling which supports VGGNet and striding approach introduced in:
    VGGNet Subsampling: Transformer-transducer: end-to-end speech recognition with self-attention (https://arxiv.org/pdf/1910.12977.pdf)
    Striding Subsampling: "Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition" by Linhao Dong et al. (https://ieeexplore.ieee.org/document/8462506)
    Args:
        subsampling (str): The subsampling technique from {"vggnet", "striding"}
        subsampling_factor (int): The subsampling factor which should be a power of 2
        feat_in (int): size of the input features
        feat_out (int): size of the output features
        conv_channels (int): Number of channels for the convolution layers.
        activation (Module): activation function, default is nn.ReLU()
    """

    def __init__(self, feat_in, feat_out, conv_channels, activation=nn.ReLU(), stride=2, kernel_size=3):
        super(ConvSubsampling_, self).__init__()
        self._sampling_num = int(math.log(4, 2))

        in_channels = 1
        
        self._ceil_mode = False

       
        self._padding = 1
        self._stride = stride
        self._kernel_size = kernel_size
        self._ceil_mode = False
    
        self.convset1 = []
        self.convset2 = []
        
        self.convset1.append(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv_channels,
                kernel_size=self._kernel_size,
                stride=self._stride,
                padding=self._padding,
            )
        )
        self.convset1.append(activation)
        in_channels = conv_channels
        self.convset2.append(
            DepthwiseSeparable(
                in_channels = feat_out * (feat_in//2), 
                out_channels = conv_channels,
                kernel_size = self._kernel_size,
                stride = self._stride,
                padding = self._padding
            )
        )
        self.convset2.append(activation)

        in_length = torch.tensor(feat_in, dtype=torch.float)
        out_length = calc_length(
            in_length,
            padding=self._padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
        self.convset1 = torch.nn.Sequential(*self.convset1)
        self.convset2 = torch.nn.Sequential(*self.convset2)

    def forward(self, x, lengths):
        lengths = calc_length(
            lengths,
            padding=self._padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        x = x.unsqueeze(1)
        #with torch.cuda.amp.autocast(dtype=torch.float32):
        x = self.convset1(x)
        x = self.convset2(rearrange(x, 'b c t f -> b (c f) t')).transpose(1, 2)

        return x, lengths


def calc_length(lengths, padding, kernel_size, stride, ceil_mode, repeat_num=1):
    """ Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = (padding * 2) - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)
