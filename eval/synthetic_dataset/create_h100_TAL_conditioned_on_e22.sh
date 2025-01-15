#!/bin/bash
#SBATCH --time=90:00:00
#SBATCH --mem=80GB
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1   
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=4


module load Anaconda3/2022.10
source activate a100


#python create.py --save_path "/mnt/parscratch/users/acp21rjf/synthetic_earnings22_test" -c "/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_360000_rp_1/step_105360.pt"

python create.py --condition_on "earnings22" -d "this_american_life" --save_path "/mnt/parscratch/users/acp21rjf/synthetic_TAL_test_conditioned_on_e22" -c "/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_360000_rp_1/step_105360.pt"


# example use: sbatch --export=CONFIG='./eval_configs_for_journal/...'  ./run_eval_h100.sh