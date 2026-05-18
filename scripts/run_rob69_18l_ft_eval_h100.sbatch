#!/bin/bash
#SBATCH --job-name=rob69-18l-ft-eval
#SBATCH --time=20:00:00
#SBATCH --mem=80GB
#SBATCH --partition=gpu-h100-nvl
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-69/eval-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-69/eval-%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-69}
CONFIG=${CONFIG:-./eval_configs_for_thesis/rob69_18l_long_context_finetune.yaml}

cd "$REPO_ROOT"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main/

python scripts/prepare_rob69_seed_results.py

cd eval
python eval_manager.py -config "$CONFIG"
