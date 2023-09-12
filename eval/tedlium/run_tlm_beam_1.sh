#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --mem=80GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8


module load Anaconda3/2022.10 binutils/2.31.1-GCCcore-8.2.0 cuDNN/8.4.1.50-CUDA-11.7.0 GCCcore/8.2.0
source activate a100

# just tune using repeat 1 on dev set (too slow to do all repeats)
# echo '--- 512 dev ---'
# python create_logits.py -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_rp/n_512_rp_1/step_105360.pt  -s "/mnt/parscratch/users/acp21rjf/spotify/logits/n_512_rp_1_dev.pt" --split "dev" --overlap 448 -seq 512

# echo '--- 2048 dev ---'
# python create_logits.py -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_rp/n_2048_rp_1/step_105360.pt  -s "/mnt/parscratch/users/acp21rjf/spotify/logits/n_2048_rp_1_dev.pt" --split "dev" --overlap 1792 -seq 2048



SEQ_LENS=(512)
CONTEXTS=(256 512 1024 64 128)
SPLITS=(dev)
REPEATS=(1)

#n_seq_sched_8192_rp_3
for SEQ_LEN in ${SEQ_LENS[@]}
do
    for SPLIT in ${SPLITS[@]}
    do
        for REPEAT in ${REPEATS[@]}
        do  
            for CONTEXT in ${CONTEXTS[@]}
            do
                echo "SEQ_LEN: ${SEQ_LEN}, OVERLAP: ${OVERLAP}, SPLIT: ${SPLIT}, REPEAT: ${REPEAT} CONTEXT: ${CONTEXT}"
                if [ ${SEQ_LEN} -eq 8192 ]
                    then
                        python tlm_beam.py -gpu -alpha 0.35 -beta 2.19 -p 3.03 -max_len ${CONTEXT} -logits /mnt/parscratch/users/acp21rjf/spotify/logits/tedlium/n_seq_sched_${SEQ_LEN}_rp_${REPEAT}_${SPLIT}.pt -log ./logs/n_seq_sched_${SEQ_LEN}_rp_${REPEAT}_${SPLIT}_tlm_beam_${CONTEXT}.log
                    else
                        python tlm_beam.py -gpu -alpha 0.35 -beta 2.19 -p 3.03 -max_len ${CONTEXT} -logits /mnt/parscratch/users/acp21rjf/spotify/logits/tedlium/n_${SEQ_LEN}_rp_${REPEAT}_${SPLIT}.pt -log ./logs/n_${SEQ_LEN}_rp_${REPEAT}_${SPLIT}_tlm_beam_${CONTEXT}.log
                fi
            done
        done
     
    done
done

exit 0

# SEQ_LENS=(8192)
# CONTEXTS=(64 128 512 1024)
# SPLITS=(dev)
# REPEATS=(2 3)

# #n_seq_sched_8192_rp_3
# for SEQ_LEN in ${SEQ_LENS[@]}
# do
#     for SPLIT in ${SPLITS[@]}
#     do
#         for REPEAT in ${REPEATS[@]}
#         do  
#             for CONTEXT in ${CONTEXTS[@]}
#             do
#                 echo "SEQ_LEN: ${SEQ_LEN}, OVERLAP: ${OVERLAP}, SPLIT: ${SPLIT}, REPEAT: ${REPEAT} CONTEXT: ${CONTEXT}"
#                 if [ ${SEQ_LEN} -eq 8192 ]
#                     then
#                         python tlm_beam.py -p 2.96 -beta 1.95 -alpha 0.42 -max_len ${CONTEXT} -logits /mnt/parscratch/users/acp21rjf/spotify/logits/tedlium/n_seq_sched_${SEQ_LEN}_rp_${REPEAT}_${SPLIT}.pt -log ./logs/n_seq_sched_${SEQ_LEN}_rp_${REPEAT}_${SPLIT}_tlm_beam_${CONTEXT}.log
#                     else
#                         python tlm_beam.py -p 2.96 -beta 1.95 -alpha 0.42 -max_len ${CONTEXT} -logits /mnt/parscratch/users/acp21rjf/spotify/logits/tedlium/n_${SEQ_LEN}_rp_${REPEAT}_${SPLIT}.pt -log ./logs/n_${SEQ_LEN}_rp_${REPEAT}_${SPLIT}_tlm_beam_${CONTEXT}.log
#                 fi
#             done
#         done
     
#     done
# done


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