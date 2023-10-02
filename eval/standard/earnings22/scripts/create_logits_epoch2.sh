#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH --mem=80GB
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8

# module load Anaconda3/2022.10 binutils/2.31.1-GCCcore-8.2.0 cuDNN/8.4.1.50-CUDA-11.7.0 GCCcore/8.2.0
# source activate a100

module unload CUDA/11.7.0
module unload cuDNN/8.4.1.50-CUDA-11.7.0
module load Anaconda3/2022.10 binutils/2.31.1-GCCcore-8.2.0 CUDA/11.8.0 cuDNN/8.6.0.163-CUDA-11.8.0 GCCcore/8.2.0
source activate /mnt/parscratch/users/acp21rjf/env/h100/


# just tune using repeat 1 on dev set (too slow to do all repeats)
# echo '--- 512 dev ---'
# python create_logits.py -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_rp/n_512_rp_1/step_105360.pt  -s "/mnt/parscratch/users/acp21rjf/spotify/logits/n_512_rp_1_dev.pt" --split "dev" --overlap 448 -seq 512

# echo '--- 2048 dev ---'
# python create_logits.py -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_rp/n_2048_rp_1/step_105360.pt  -s "/mnt/parscratch/users/acp21rjf/spotify/logits/n_2048_rp_1_dev.pt" --split "dev" --overlap 1792 -seq 2048

# echo '--- 8192 dev ---'
# python create_logits.py -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_3/n_seq_sched_8192_rp_1/step_105360.pt  -s "/mnt/parscratch/users/acp21rjf/spotify/logits/n_seq_sched_8192_rp_1_dev.pt" --split "dev" -seq 8192 --overlap 7168

# echo '--- time for the test set ---'

SPLITS=(dev test)

for SPLIT in "${SPLITS[@]}"
do
    python create_logits.py -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_multi_epoch/n_seq_sched_8192_rp_1/step_105360.pt  -s "/mnt/parscratch/users/acp21rjf/spotify/logits/earnings/epoch_2_n_seq_sched_8192_rp_1_${SPLIT}.pt" --split "${SPLIT}" --overlap 7168 -seq 8192
done
