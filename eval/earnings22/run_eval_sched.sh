#!/bin/bash

SEQ_LEN=2048
OVERLAP=1792
vars=(dev test)

#n_seq_sched_8192_rp_3
for var in ${vars[@]}
do
    for num in {1..3}
    do
        python run.py -seq ${SEQ_LEN} -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_3/n_seq_sched_${SEQ_LEN}_rp_${num}/step_105360.pt  -log "./logs/sched_${SEQ_LEN}_${var}.log" --split "${var}" -overlap ${OVERLAP} 
    done
done


SEQ_LEN=4096
OVERLAP=3584
vars=(dev test)

#n_seq_sched_8192_rp_3
for var in ${vars[@]}
do
    for num in {1..3}
    do
        python run.py -seq ${SEQ_LEN} -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_3/n_seq_sched_${SEQ_LEN}_rp_${num}/step_105360.pt  -log "./logs/sched_${SEQ_LEN}_${var}.log" --split "${var}" -overlap ${OVERLAP}
    done
done

SEQ_LEN=8192
OVERLAP=7168
vars=(dev test)

#n_seq_sched_8192_rp_3
for var in ${vars[@]}
do
    for num in {1..3}
    do
        python run.py -seq ${SEQ_LEN} -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_3/n_seq_sched_${SEQ_LEN}_rp_${num}/step_105360.pt  -log "./logs/sched_${SEQ_LEN}_${var}.log" --split "${var}" -overlap ${OVERLAP}
    done
done


SEQ_LEN=16384
OVERLAP=14336
vars=(dev test)

#n_seq_sched_8192_rp_3
for var in ${vars[@]}
do
    for num in {1..3}
    do
        python run.py -seq ${SEQ_LEN} -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_3/n_seq_sched_${SEQ_LEN}_rp_${num}/step_105360.pt  -log "./logs/sched_${SEQ_LEN}_${var}.log" --split "${var}" -overlap ${OVERLAP}
    done
done

SEQ_LEN=32768
OVERLAP=28672
vars=(dev test)

#n_seq_sched_8192_rp_3
for var in ${vars[@]}
do
    for num in {1..3}
    do
        python run.py -seq ${SEQ_LEN} -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_3/n_seq_sched_${SEQ_LEN}_rp_${num}/step_105360.pt  -log "./logs/sched_${SEQ_LEN}_${var}.log" --split "${var}" -overlap ${OVERLAP}
    done
done
