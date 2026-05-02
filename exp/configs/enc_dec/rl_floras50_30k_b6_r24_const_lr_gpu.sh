#!/bin/bash
#SBATCH --job-name=ROB-26-rl30k
#SBATCH --time=24:00:00
#SBATCH --mem=130GB
#SBATCH --partition=gpu-h100-nvl
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/rl30k-%x-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/rl30k-%x-%j.err

set -euo pipefail

CONFIG="${CONFIG:?Set CONFIG to an RL config path}"
REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces/ROB-26}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
mkdir -p /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26
mkdir -p /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/checkpoints
mkdir -p /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/wandb

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

python exp/train_files/train_enc_dec_rl.py --config "${CONFIG}"
