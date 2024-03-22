#!/bin/bash
#SBATCH --time=90:00:00
#SBATCH --mem=80GB
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1   
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=4


module load Anaconda3/2022.10
source activate a100


python per_ep_eval.py -seq 360000000 -o 0 -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_360000_rp_1/step_105360.pt
python per_ep_eval.py -seq 360000000 -o 0 -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_360000_rp_2/step_105360.pt
python per_ep_eval.py -seq 360000000 -o 0 -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_360000_rp_3/step_105360.pt