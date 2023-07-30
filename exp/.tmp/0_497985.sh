#!/bin/bash

#SBATCH --time=96:00:00
#SBATCH --mem=200GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=16

module load Anaconda3/2022.10 binutils/2.31.1-GCCcore-8.2.0 cuDNN/8.4.1.50-CUDA-11.7.0 GCCcore/8.2.0
source activate a100

python train_xl.py -config ./.tmp/0_497985.yaml -num_workers 0 -rm_sched -reset_step