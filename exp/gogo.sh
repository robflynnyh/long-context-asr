#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --mem=190GB
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=16

module unload CUDA/11.7.0
module unload cuDNN/8.4.1.50-CUDA-11.7.0
module load Anaconda3/2022.10 binutils/2.31.1-GCCcore-8.2.0 CUDA/11.8.0 cuDNN/8.6.0.163-CUDA-11.8.0 GCCcore/8.2.0
source activate /mnt/parscratch/users/acp21rjf/env/h100/

python train_meta.py -config ./configs/meta_test.yaml