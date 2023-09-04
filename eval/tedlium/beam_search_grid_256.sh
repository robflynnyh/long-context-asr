#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=30GB
#SBATCH --cpus-per-task=8

module load Anaconda3/2022.10 binutils/2.31.1-GCCcore-8.2.0 GCCcore/8.2.0
source activate a100


python tlm_beam_search.py -max_len 256 

