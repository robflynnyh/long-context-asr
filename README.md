# Code for the paper: How Much Context Does My Attention-Based ASR System Need? (Interspeech 2024)
![figure 1.0 from paper](https://github.com/robflynnyh/long-context-asr/blob/main/eval/results/IS_paper/weracross_data.png)
- Current Pre-Print accepted at Interspeech 2024 available on [arXiv](https://arxiv.org/abs/2310.15672) 
- Code for the old 2023 preprint is in the 2023-preprint branch
- Repository is continually being updated with more intstructions, eventually I hope to have a colab that can be used to run some of the pretrained models
- As repo is w.i.p if you cannot figure out how to use anything please feel free to contact me by creating an issue!

## Installation
- Requires Pytorch 2.0 or greater
- Install Flash Attention 2.0 https://github.com/Dao-AILab/flash-attention for best performance
- Install fused_dense_lib for fused MLP layers from https://github.com/Dao-AILab/flash-attention/tree/main/csrc/fused_dense_lib
- [Apex](https://github.com/NVIDIA/apex/tree/master) is used for fused rms/layer norm (and fused Adam if not using madgrad)
TODO: setup code to work without fused layers installed for easier usage

## Data
- <del>For training models you must request access to receive the spotify training data which can be done via the following:</del> [link](https://podcastsdataset.byspotify.com/) (unfortunatly spotify are no longer maintaining this dataset)
- For training models this code will work with any set of data where you have unsegmented (not segmented into utterances) precomputed spectrograms and corresponding transcriptions (with word level alignment). Word level alignement is needed to be able to chunk files into arbitrary sequence lengths.
- Evaluation dev/test splits for Earnings-22 and Tedlium can be found in [/data](https://github.com/robflynnyh/long-context-asr/tree/main/data)
- Alternatively the full datasets can be found via the following links: [Earnings-22](https://github.com/revdotcom/speech-datasets/tree/main/earnings22) [Tedlium](https://www.openslr.org/51/) 

## Checkpoints
Config files for all pretrained models are provided within the checkpoint file
### Acoustic Model
All checkpoints from the paper are currently hosted on [huggingface](https://huggingface.co/rjflynn2) below is a table that provides links for each model type aswell as there configuration in performance.
checkpoints for each sequence length and repeat are contained inside folders in each repository that is linked for example a model with 10s of context and repeat 1 (out of 3) would be in the folder: n_seq_sched_1024_rp_1. 1024 represents the spectrogram length, 1024/100 =10.24 seconds, rp_1 = repeat 1. 
All models from this table/paper use [this](https://github.com/robflynnyh/long-context-asr/blob/main/lcasr/models/sconformer_xl.py) model class.
If anything is unclear let me know!

| Download | D_model | Layers | Params (M) | Attn head dim | Epochs | Pos Enc | SpecAugment | Subsampling | Tedlium (WER) | Earnings-22 (WER) | 
|----------|---------|--------|------------|---------------|--------|---------|-------------|-------------|---------------|-------------------|
| [here](https://huggingface.co/rjflynn2/lcasr-9L-768D-6H-RB-1p5M) | 768 | 9 | ~120 | 128 | 1 | Rotary (\theta=1.5M) | No | 8X Depthwise 256D | 6.8/6.0/5.9 | 26.6/23.1/22.7 |
| [here](https://huggingface.co/rjflynn2/lcasr-6L-768D-6H-RB-1p5M) | 768 | 6 | 90 | 128 | 1 | Rotary (\theta=1.5M) | No | 8X Depthwise 256D | 6.8/6.4/6.2 | 27.7/24.6/24.4 | 
| [here](https://huggingface.co/rjflynn2/lcasr-6L-768D-12H-RB-1p5M)| 768 | 6 | 90 | 64 | 1 | Rotary (\theta=1.5M) | No | 8X Depthwise 256D | ... | 27.5/24.8/24.4 | 
| [here](https://huggingface.co/rjflynn2/lcasr-6L-768D-24H-RB-1p5M)| 768 | 6 | 90 | 32 | 1 | Rotary (\theta=1.5M) | No | 8X Depthwise 256D | 6.8/6.4/6.4 |  26.7/24.6/24.8| 
| [see 2023-preprint branch](https://github.com/robflynnyh/long-context-asr/tree/2023-preprint) | 768 | 6 | 90 | 128 | 1 | Rotary (\theta=10K)  | No | 8X Depthwise 256D | ... | 27.2/24.9/25.0* | 
| [here](https://huggingface.co/rjflynn2/lcasr-6L-768D-6H-SinePos) | 768 | 6 | 90 | 128 | 1 | Sine | No | 8X Depthwise 256D | 7.2/6.7/6.6 | 27.8/25.3/25.3 | 
| [here](https://huggingface.co/rjflynn2/lcasr-6L-768D-6H-NoPos) | 768 | 6 | 90 | 128 | 1 | None | No | 8X Depthwise 256D | 7.7/6.8/6.6 | 27.5/25.3/25.2 | 
| [here](https://huggingface.co/rjflynn2/lcasr-3L-2048D-16H-RB-1p5M) | 2048 | 3 | 315 | 128 | 1 | Rotary (\theta=1.5M) | No | 8X Depthwise 256D | ... | 28.7/26.1/26.1 | 
| [here](https://huggingface.co/rjflynn2/lcasr-3L-768D-6H-RB-1p5M) | 768 | 3 | ~50 | 128 | 1 | Rotary (\theta=1.5M) | No | 8X Depthwise 256D | 8.2/7.8/7.4 | 32.3/29.6/30.2 | 
| [here](https://huggingface.co/rjflynn2/lcasr-12L-256D-8H-RB-1p5M) | 256 | 12 | ~20 | 32 | 1 | Rotary (\theta=1.5M) | No | 8X Depthwise 256D | 7.6/6.9/6.9 | 28.6/26.3/26.4 | 
| [here](https://huggingface.co/rjflynn2/lcasr-6L-256D-8H-RB-1p5M) | 256 | 6 | ~10 | 32 | 1 | Rotary (\theta=1.5M) | No | 8X Depthwise 256D | 8.8/8.0/8.2 | 32.2/29.8/29.9 |

*@1hour
