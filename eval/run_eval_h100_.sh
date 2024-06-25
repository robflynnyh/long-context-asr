#!/bin/bash
#SBATCH --time=90:00:00
#SBATCH --mem=80GB
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1   
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=4


module load Anaconda3/2022.10
source activate a100

echo "Running eval_manager.py with config $CONFIG"

python eval_manager_.py -config $CONFIG

# example use: sbatch --export=CONFIG='./eval_configs_for_journal/...'  ./run_eval_h100_.sh