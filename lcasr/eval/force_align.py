# uses code from: https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html
# https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html

import numpy as np
import torch
from torch import nn
from torch.nn import Tensor
from lcasr.utils.audio_tools import total_frames, total_seconds
from typing import List, Dict, Tuple, Union
from tqdm import tqdm
from lcasr.models.sconformer_xl import SCConformerXL
from .buffered_transcription import fetch_logits

def force_align(
        model:Union[SCConformerXL, nn.Module],
        audio:Tensor,
        transcript:str
    ):
    pass