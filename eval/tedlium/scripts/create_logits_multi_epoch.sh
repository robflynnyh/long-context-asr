SPLITS=(dev test)

for SPLIT in "${SPLITS[@]}"
do
    python create_logits.py -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints/multi_epoch_run_16k/step_948240.pt  -s "/mnt/parscratch/users/acp21rjf/spotify/logits/tedlium/multi_epoch_run_16k_${SPLIT}.pt" --split "${SPLIT}" -overlap 14336 -seq 16384
done
