#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH --mem=80GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=4

module load Anaconda3/2022.10 binutils/2.31.1-GCCcore-8.2.0 cuDNN/8.4.1.50-CUDA-11.7.0 GCCcore/8.2.0
source activate a100

SEQ_LENS=(8192 2048 512)
OVERLAP_Ps=(0.0 0.25 0.50 0.75 0.875 0.9375)
SPLITS=(dev test)
REPEATS=(1 2 3)

#n_seq_sched_8192_rp_3
for SEQ_LEN in ${SEQ_LENS[@]}
do
    for SPLIT in ${SPLITS[@]}
    do
        for OVERLAP_P in ${OVERLAP_Ps[@]}
        do
            for REPEAT in ${REPEATS[@]}
            do  

                X=$(bc <<< "scale=10; ${OVERLAP_P}*${SEQ_LEN}")
                OVERLAP=${X%.*}
                echo "SEQ_LEN: ${SEQ_LEN}, SPLIT: ${SPLIT}, REPEAT: ${REPEAT} OVERLAP: ${OVERLAP}"
                if [ ${SEQ_LEN} -eq 8192 ]
                    then
                        python run.py -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_3/n_seq_sched_${SEQ_LEN}_rp_${REPEAT}/step_105360.pt  -log "./logs/overlap_eval_${SEQ_LEN}_${SPLIT}.log" --split "${SPLIT}" --overlap ${OVERLAP}
                    else
                        python run.py -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_rp/n_${SEQ_LEN}_rp_${REPEAT}/step_105360.pt  -log "./logs/overlap_eval_${SEQ_LEN}_${SPLIT}.log" --split "${SPLIT}" --overlap ${OVERLAP}
                fi
            done
        done
    done
done

