# Code for the paper: How Much Context Does My Attention-Based ASR System Need?
Repository to be updated with instructions and links to all model checkpoints a.s.a.p

## Installation
For lanaguage model decoding the following repo must also be installed: https://github.com/robflynnyh/language_modelling
instructions on how to properly install this repo and the required libraries will be provided a.s.a.p
- Requires Pytorch 2.0 or greater
- currently we are using [flash-attention 1](https://github.com/Dao-AILab/flash-attention/tree/6d48e14a6c2f551db96f0badc658a6279a929df3) (update to v2 planned in future)
- [Apex](https://github.com/NVIDIA/apex/tree/master) is used for fused rms/layer norm (and fused Adam if not using madgrad)
TODO: setup code to work without flash-attention and fused layers installed for easier usage

## Data
- For training models you must request access to receive the spotify training data which can be done via the following: [link](https://podcastsdataset.byspotify.com/)
- Evaluation dev/test splits for Earnings-22 and Tedlium can be found in [/data](https://github.com/robflynnyh/long-context-asr/tree/main/data)
- Alternatively the full datasets can be found via the following links: [Earnings-22](https://github.com/revdotcom/speech-datasets/tree/main/earnings22) [Tedlium](https://www.openslr.org/51/) 

## Checkpoints
### Acoustic Model
Below are model checkpoints for Acoustic models discussed in the paper. The <b>greedy</b> WERs (no LM) are also provided using overlapping inferernce (87.5% overlap). For checkpoints with multiple repeats the average WERs are provided.
| Context | Epochs | Seq Warmup | Tedlium (WER) | Earnings-22 (WER) | Download |
| --------|-------:|:----------:|--------------:|------------------:|----------|
|  80s    |    2   |  Yes       |       6.1     |      17.1         | [here](https://huggingface.co/rjflynn2/lcasr-80s-epoch-2/) |
|  1 hour |    1   |  Yes       |       6.4     |      18.8         | [here](https://huggingface.co/rjflynn2/lcasr-1hour) |
|  320s   |    1   | Yes        |       6.5     |      18.6         | [here](https://huggingface.co/rjflynn2/lcasr-320s) |
|  160s   |    1   | Yes        |       6.5     |      18.7         | [here](https://huggingface.co/rjflynn2/lcasr-160s) |
|  80s    |    1   | Yes        |       6.5     |      18.7         | [here](https://huggingface.co/rjflynn2/lcasr-80s) |
|  40s    |    1   | Yes        |       6.7     |      19.2         | [here](https://huggingface.co/rjflynn2/lcasr-40s-seq_warmup) |
|  40s    |    1   | No         |       6.5     |      19.4         | [here](https://huggingface.co/rjflynn2/lcasr-40s) |
|  20s    |    1   | No         |       6.6     |      19.4         | [here](https://huggingface.co/rjflynn2/lcasr-20s)  |
|  10s    |    1   | No         |       6.8     |      20.5         | [here](https://huggingface.co/rjflynn2/lcasr-10s)  |
|  5s    |    1   | No         |       7.4     |      21.9         | [here](https://huggingface.co/rjflynn2/lcasr-5s)  |

### Language Model
Language Model checkpoint added soon!

## All Results

A <b>messy</b> dump of experimental results including WERs for each repeat and to a higher precision presented paper, can be found [here](https://github.com/robflynnyh/long-context-asr/blob/main/artifacts/experiment_dump.pdf)
