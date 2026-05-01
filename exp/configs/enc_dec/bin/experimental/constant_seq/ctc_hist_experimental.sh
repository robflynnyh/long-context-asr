#!/bin/bash
#SBATCH --time=90:00:00
#SBATCH --mem=110GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1   
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=16


module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main/

cd /users/acp21rjf/long-context-asr/exp
python train_enc_dec.py -config ./configs/enc_dec/bin/experimental/constant_seq/ctc_hist_experimental.yaml -num_workers 0
