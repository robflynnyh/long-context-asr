#!/bin/bash
#SBATCH --time=90:00:00
#SBATCH --mem=1007GB
#SBATCH --cpus-per-task=8


module load Anaconda3/2022.10
source activate a100

echo "Running eval_manager.py with config $CONFIG"

python get_attention_weights.py -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_360000_rp_1/step_105360.pt -save /mnt/parscratch/users/acp21rjf/attn_weights_earnings22/