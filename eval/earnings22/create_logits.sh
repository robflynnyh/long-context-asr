#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --mem=52GB
#SBATCH --cpus-per-task=16

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main


SEQ_LENS=(1024 2048 4096 8192 16384 32768 65536 131072 262144 360000)
OVERLAP_Ps=(0.875)
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
                echo "SEQ_LEN: ${SEQ_LEN}, OVERLAP: ${OVERLAP}, SPLIT: ${SPLIT}, REPEAT: ${REPEAT}"
          
                python create_logits.py -c /mnt/parscratch/users/acp21rjf/lcasr-6L-768D-6H-RB-1p5M/n_seq_sched_${SEQ_LEN}_rp_${REPEAT}/step_105360.pt  -s "/mnt/parscratch/users/acp21rjf/spotify/logits/earnings/n_seq_sched_${SEQ_LEN}_rp_${REPEAT}_${SPLIT}.pt" --split "${SPLIT}" --overlap ${OVERLAP} --seq_len ${SEQ_LEN}

            done
        done
    done
done

