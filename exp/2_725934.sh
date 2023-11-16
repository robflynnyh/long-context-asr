#!/bin/bash

#SBATCH --time=90:00:00
#SBATCH --mem=160GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8

module load Anaconda3/2022.10 binutils/2.35-GCCcore-10.2.0 CUDA/11.8.0 cuDNN/8.6.0.163-CUDA-11.8.0 GCCcore/10.2.0
source activate a100

python train_meta.py -config ./configs/meta_test_nonorm.yaml