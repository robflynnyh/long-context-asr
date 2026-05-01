#!/bin/bash
#SBATCH --job-name=ROB-26-eval
#SBATCH --time=20:00:00
#SBATCH --mem=80GB
#SBATCH --partition=gpu-h100-nvl
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/eval-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/eval-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces/ROB-26}"
CONFIG="${CONFIG:-./eval_configs/enc_dec_rl_tedlium.yaml}"

mkdir -p /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/eval

cd "${REPO_DIR}/eval"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

echo "Running eval_manager.py with config ${CONFIG}"
python eval_manager.py -config "${CONFIG}"
