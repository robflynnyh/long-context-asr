#!/bin/bash

SEQ_LENS=(360000 32768 2048 4096 8192 16384)
OVERLAP=0
vars=(dev test)

#n_seq_sched_8192_rp_3
for SEQ_LEN in ${SEQ_LENS[@]}
do
    for var in ${vars[@]}
    do
        for num in {2..3}
        do
            python run.py  --single_utt -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_3/n_seq_sched_${SEQ_LEN}_rp_${num}/step_105360.pt  -log "./logs/single_utt_sched_${SEQ_LEN}_${var}.log" --split "${var}" 
        done
    done
done

