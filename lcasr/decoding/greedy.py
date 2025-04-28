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
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]` or `[num_batch, num_seq, num_label]`
          note that batches are not processed in parallel (other than the argmax) though it shouldn't be a problem for reasonable batch sizes

        Returns:
          List[str]: The resulting transcript
        """
        decode = decode and self.tokenizer is not None
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        if indices.ndim > 1:
            # batch of sequences
            indices = [torch.unique_consecutive(i, dim=-1).tolist() for i in indices]
            indices = [[i for i in index if i != self.blank] for index in indices]
            if decode:
                indices = [self.tokenizer.decode(index) for index in indices]
            return indices
        else:
          indices = torch.unique_consecutive(indices, dim=-1).tolist()
          indices = [i for i in indices if i != self.blank]
          return self.tokenizer.decode(indices) if decode else indices


