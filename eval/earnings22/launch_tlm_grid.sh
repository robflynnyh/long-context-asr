#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=60GB
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1   
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8


module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main/


python run_tlm_grid_search.py --logits_path_dev /mnt/parscratch/users/acp21rjf/spotify/logits/earnings/n_seq_sched_${SEQ_LEN}_rp_${REPEAT}_dev.pt --logits_path_test /mnt/parscratch/users/acp21rjf/spotify/logits/earnings/n_seq_sched_${SEQ_LEN}_rp_${REPEAT}_test.pt --use_gpu --log_path ./logs/beam_search_thesis_constant_lm_params_no_init_cache/n_seq_sched_${SEQ_LEN}_rp_${REPEAT}_tlm_beam_grid_search.log --no_init_cache
