#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH --mem=80GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=4

module load Anaconda3/2022.10 binutils/2.31.1-GCCcore-8.2.0 cuDNN/8.4.1.50-CUDA-11.7.0 GCCcore/8.2.0
source activate a100

SEQ_LENS=(65536)
OVERLAP_Ps=(0.875)
SPLITS=(dev test)
CHECKPOINTS=($(ls /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rotarybase/ | grep freq_spec))


#n_seq_sched_8192_rp_3
for SEQ_LEN in ${SEQ_LENS[@]}
do
    for SPLIT in ${SPLITS[@]}
    do
        for OVERLAP_P in ${OVERLAP_Ps[@]}
        do
            for CHECKPOINT in ${CHECKPOINTS[@]}
            do
                X=$(bc <<< "scale=10; ${OVERLAP_P}*${SEQ_LEN}")
                OVERLAP=${X%.*}
                echo "SEQ_LEN: ${SEQ_LEN}, SPLIT: ${SPLIT}, REPEAT: ${REPEAT} OVERLAP: ${OVERLAP}"
                python run.py -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rotarybase/${CHECKPOINT}/step_105360.pt  -log "./logs/${CHECKPOINT}_${SPLIT}.log" --split "${SPLIT}" -overlap ${OVERLAP} -seq ${SEQ_LEN} 
          
            done
        done
    done
done

