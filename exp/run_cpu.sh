#!/bin/bash
#SBATCH --time=90:00:00

module load Anaconda3/2022.10
source activate a100

python save_utterances.py
