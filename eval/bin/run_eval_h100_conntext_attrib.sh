#!/bin/bash
#SBATCH --time=90:00:00
#SBATCH --mem=80GB
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1   
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=4


module load Anaconda3/2022.10
source activate a100


for i in {1..3}; do
    python run_context_attribution.py -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_360000_rp_"$i"/step_105360.pt -d tedlium
done

for i in {1..3}; do
    python run_context_attribution.py -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_360000_rp_"$i"/step_105360.pt -d earnings22
done

for i in {1..3}; do
    python run_context_attribution.py -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_360000_rp_"$i"/step_105360.pt -d rev16
done

for i in {1..3}; do
    python run_context_attribution.py -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_360000_rp_"$i"/step_105360.pt -d this_american_life
done