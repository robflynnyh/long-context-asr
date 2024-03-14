#!/bin/bash
#SBATCH --time=90:00:00
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=8

module load Anaconda3/2022.10
source activate a100

python eval_manager.py -config eval_config_rb_spotify.yaml