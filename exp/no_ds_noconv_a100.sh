#!/bin/bash
#SBATCH --time=90:00:00
#SBATCH --mem=130GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=4

module load Anaconda3/2022.10
source activate a100

python train.py -config ./configs/no_ds_no_conv_a100.yaml
