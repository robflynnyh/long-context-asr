
SEQ_LENS=(1024 2048 4096 8192 16384 32768 65536 131072 262144 360000)
REPEATS=(1 2 3)

#n_seq_sched_8192_rp_3
for SEQ_LEN in ${SEQ_LENS[@]}
do
    for REPEAT in ${REPEATS[@]}
    do  
        echo "SEQ_LEN: ${SEQ_LEN}, REPEAT: ${REPEAT}"
        sbatch --export=SEQ_LEN=${SEQ_LEN},REPEAT=${REPEAT} ./launch_tlm_grid.sh

    done
done

