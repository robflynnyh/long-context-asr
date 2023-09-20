# Code for the paper: How Much Context Does My Attention-Based ASR System Need?
Repository to be updated with instructions and links to all model checkpoints a.s.a.p
## Installation
For lanaguage model decoding the following repo must also be installed: https://github.com/robflynnyh/language_modelling
instructions on how to properly install this repo and the required libraries will be provided a.s.a.p
- currently we are using [flash-attention 1](https://github.com/Dao-AILab/flash-attention/tree/6d48e14a6c2f551db96f0badc658a6279a929df3)
- [Apex](https://github.com/NVIDIA/apex/tree/master) is used for fused rms/layer norm (and fused Adam is not using madgrad)

## Checkpoints
Below are model checkpoints for Acoustic models discussed in the paper. The <b>greedy</b> WERs (no LM) are also provided using overlapping inferernce (87.5% overlap). For checkpoints with multiple repeats the average WERs are provided.
| Context | Epochs | Seq Warmup | Tedlium (WER) | Earnings-22 (WER) | Download |
| --------|-------:|:----------:|--------------:|------------------:|----------|
|  80s    |    2   |  Yes       |       6.1     |      17.1         | [here](https://huggingface.co/rjflynn2/lcasr-80s-epoch-2/) |
|  1 hour |    1   |  Yes       |       6.4     |      18.8         | [here](https://huggingface.co/rjflynn2/lcasr-1hour) |
|  320s   |    1   | Yes        |       6.5     |      18.6         | [here](https://huggingface.co/rjflynn2/lcasr-320s) |

Remaining checkpoints, including language model checkpoints for decoding, and instructions on how to use and train models coming soon..!


A <b>messy</b> dump of experimental results including WERs for each repeat and to a higher precision presented paper, can be found [here](https://github.com/robflynnyh/long-context-asr/blob/main/artifacts/experiment_dump.pdf)
