#!/bin/bash
#SBATCH --time=03:30:00
#SBATCH --mem=32GB

module load Anaconda3/2022.10
source activate a100
python -m lcasr.utils.preprocess --ogg_path /mnt/parscratch/users/acp21rjf/spotify/audio/7/S --stage 0
        