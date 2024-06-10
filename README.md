# Code for the paper: How Much Context Does My Attention-Based ASR System Need? (Interspeech 2024)
- Pre-Print Available on [arXiv](https://arxiv.org/abs/2310.15672) (currently an old pre-print! from prior ICASSP submission)
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
All checkpoints from the paper are currently hosted on huggingface.co/rjflynn2/ I will be adding a table to this README with specific links to each checkpoint and the checkpoints performance ASAP.  

