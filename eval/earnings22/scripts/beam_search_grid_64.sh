#!/bin/bash
#SBATCH --time=03:45:00
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=14

module load Anaconda3/2022.10 binutils/2.31.1-GCCcore-8.2.0 cuDNN/8.4.1.50-CUDA-11.7.0 GCCcore/8.2.0
source activate a100

# module unload CUDA/11.7.0
# module unload cuDNN/8.4.1.50-CUDA-11.7.0
# module load Anaconda3/2022.10 binutils/2.31.1-GCCcore-8.2.0 CUDA/11.8.0 cuDNN/8.6.0.163-CUDA-11.8.0 GCCcore/8.2.0
# source activate /mnt/parscratch/users/acp21rjf/env/h100/

wandb agent wobrob101/long-context-asr-eval_earnings22/lvru12m8
