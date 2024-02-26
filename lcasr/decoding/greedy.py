import torch, torch.nn as nn
from typing import List

class GreedyCTCDecoder(torch.nn.Module): # Modifcation of: https://pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html
    def __init__(self, tokenizer=None, blank_id=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.blank = blank_id

    def forward(self, emission: torch.Tensor, decode=True) -> str:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        decode = decode and self.tokenizer is not None
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1).tolist()
        indices = [i for i in indices if i != self.blank]
        return self.tokenizer.decode(indices) if decode else indices


