#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --mem=180GB
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8

module load Anaconda3/2022.10
source activate a100

python train.py -config ./configs/no_ds_no_conv.yaml
