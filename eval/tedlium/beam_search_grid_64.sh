#!/bin/bash
#SBATCH --time=00:45:00
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=14

module load Anaconda3/2022.10 binutils/2.31.1-GCCcore-8.2.0 cuDNN/8.4.1.50-CUDA-11.7.0 GCCcore/8.2.0
source activate a100

wandb agent wobrob101/long-context-asr-eval_tedlium/336cg2dk
