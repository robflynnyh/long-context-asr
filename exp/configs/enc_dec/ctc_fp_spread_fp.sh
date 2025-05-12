#!/bin/bash
#SBATCH --time=90:00:00
#SBATCH --mem=100GB
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1   
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=16


module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main/

cd /users/acp21rjf/long-context-asr/exp
python train_enc_dec_spread_test.py -config ./configs/enc_dec/ctc_fp_spread_test.yaml -num_workers 0
