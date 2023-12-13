#!/bin/bash
#SBATCH --time=31:00:00
#SBATCH --mem=150GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8

module load Anaconda3/2022.10
source activate a100

python train_mamba.py -config ./configs/mamba_test.yaml
