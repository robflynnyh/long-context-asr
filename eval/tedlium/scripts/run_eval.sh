#!/bin/bash

SEQ_LEN=512
vars=(dev test)

for var in ${vars[@]}
do
    for num in {2..3}
    do
        python run.py -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_rp/n_${SEQ_LEN}_rp_${num}/step_105360.pt  -log "./logs/${SEQ_LEN}_${var}.log" --split "${var}" -overlap 448
    done
done

SEQ_LEN=1024
vars=(dev test)

for var in ${vars[@]}
do
    for num in {2..3}
    do
        python run.py -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_rp/n_${SEQ_LEN}_rp_${num}/step_105360.pt  -log "./logs/${SEQ_LEN}_${var}.log" --split "${var}" -overlap 896 
    done
done

SEQ_LEN=2048
vars=(dev test)

for var in ${vars[@]}
do
    for num in {1..3}
    do
        python run.py -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_rp/n_${SEQ_LEN}_rp_${num}/step_105360.pt  -log "./logs/${SEQ_LEN}_${var}.log" --split "${var}" -overlap 1792
    done
done


SEQ_LEN=4096
vars=(dev test)

for var in ${vars[@]}
do
    for num in {1..2}
    do
        python run.py -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_rp/n_${SEQ_LEN}_rp_${num}/step_105360.pt  -log "./logs/${SEQ_LEN}_${var}.log" --split "${var}" -overlap 3584
    done
done

