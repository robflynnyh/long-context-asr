#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH --mem=80GB
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=12

# module load Anaconda3/2022.10 binutils/2.31.1-GCCcore-8.2.0 cuDNN/8.4.1.50-CUDA-11.7.0 GCCcore/8.2.0
# source activate a100

module unload CUDA/11.7.0
module unload cuDNN/8.4.1.50-CUDA-11.7.0
module load Anaconda3/2022.10 binutils/2.31.1-GCCcore-8.2.0 CUDA/11.8.0 cuDNN/8.6.0.163-CUDA-11.8.0 GCCcore/8.2.0
source activate /mnt/parscratch/users/acp21rjf/env/h100/

SPLITS=(test dev)

for SPLIT in ${SPLITS[@]}
do
    python tlm_beam.py -gpu -p 2.96 -beta 1.95 -alpha 0.42  -max_len 1024 -logits /mnt/parscratch/users/acp21rjf/spotify/logits/earnings/epoch_2_n_seq_sched_8192_rp_1_${SPLIT}.pt -log ./logs2/EPOCH2_n_seq_sched_8192_rp_1_${SPLIT}_tlm_beam_1024.log
done


exit 0


    # !/bin/bash
    # SBATCH --time=40:00:00
    # SBATCH --mem=80GB
    # SBATCH --partition=gpu
    # SBATCH --gres=gpu:1
    # SBATCH --qos=gpu
    # SBATCH --cpus-per-task=4


#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH --mem=80GB
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8

# module load Anaconda3/2022.10 binutils/2.31.1-GCCcore-8.2.0 cuDNN/8.4.1.50-CUDA-11.7.0 GCCcore/8.2.0
# source activate a100

# module unload CUDA/11.7.0
# module unload cuDNN/8.4.1.50-CUDA-11.7.0
# module load Anaconda3/2022.10 binutils/2.31.1-GCCcore-8.2.0 CUDA/11.8.0 cuDNN/8.6.0.163-CUDA-11.8.0 GCCcore/8.2.0
# source activate /mnt/parscratch/users/acp21rjf/env/h100/