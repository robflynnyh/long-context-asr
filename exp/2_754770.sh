#!/bin/bash

#SBATCH --time=90:00:00
#SBATCH --mem=150GB
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8

module load Anaconda3/2022.10
source activate a100


python train.py -config ./.tmp/2_754770.yaml -num_workers 0