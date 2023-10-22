SEQ_LENS=(360000)
OVERLAP_Ps=(0.875)
SPLITS=(test)
REPEATS=(1 2 3)
#CHECKPOINTS=($(ls /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rotarybase/ | grep freq_spec))


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
                python run.py -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_spec/${SEQ_LEN}_rotbase_1p5M_spec_freq_mask_repeat_${REPEAT}/step_105360.pt  -log "./logs/${SEQ_LEN}_rotbase_1p5M_spec_freq_mask_${SPLIT}.log" --split "${SPLIT}" -overlap ${OVERLAP} -seq ${SEQ_LEN} 
          
            done
        done
    done
done

