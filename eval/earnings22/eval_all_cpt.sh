#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --mem=60GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=16

module load Anaconda3/2022.10 binutils/2.31.1-GCCcore-8.2.0 cuDNN/8.4.1.50-CUDA-11.7.0 GCCcore/8.2.0
source activate a100

python eval_for_each_cpt.py -cf /mnt/parscratch/users/acp21rjf/spotify/checkpoints/1e3_4096_9000_wu/ -seq 4096 -overlap 3584 -log ./logs/4096_logs.log
python eval_for_each_cpt.py -cf /mnt/parscratch/users/acp21rjf/spotify/checkpoints/512_4e3/ -seq 512 -overlap 448 -log ./logs/512_logs.log
python eval_for_each_cpt.py -cf /mnt/parscratch/users/acp21rjf/spotify/checkpoints/3e3_1024/ -seq 1024 -overlap 896 -log ./logs/1024_logs.log
python eval_for_each_cpt.py -cf /mnt/parscratch/users/acp21rjf/spotify/checkpoints/2048_3e3/ -seq 2048 -overlap 1792 -log ./logs/2048_logs.log

