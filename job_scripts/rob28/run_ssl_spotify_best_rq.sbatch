#!/bin/bash
#SBATCH --job-name=rob28-ssl
#SBATCH --time=36:00:00
#SBATCH --mem=130GB
#SBATCH --partition=gpu-h100-nvl
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-28/logs/ssl-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-28/logs/ssl-%j.err

set -euo pipefail

REPO=/mnt/parscratch/users/acp21rjf/symphony-workspaces/ROB-28
ARTIFACT_ROOT=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-28

mkdir -p "${ARTIFACT_ROOT}/logs" "${ARTIFACT_ROOT}/wandb" "${ARTIFACT_ROOT}/checkpoints/ssl_spotify_ctx2048"
cd "${REPO}"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export WANDB_DIR="${ARTIFACT_ROOT}/wandb"
export WANDB_PROJECT=spotify_long_context_ssl

python exp/train_bestRQ.py \
  -config exp/configs/rob28_ssl_spotify_best_rq_6l_ctx2048.yaml \
  --num_workers 4 \
  --pin_memory \
  --prefetch 2
