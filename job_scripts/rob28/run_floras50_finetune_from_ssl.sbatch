#!/bin/bash
#SBATCH --job-name=rob28-ft
#SBATCH --time=24:00:00
#SBATCH --mem=130GB
#SBATCH --partition=gpu-h100-nvl
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-28/logs/finetune-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-28/logs/finetune-%j.err

set -euo pipefail

REPO=/mnt/parscratch/users/acp21rjf/symphony-workspaces/ROB-28
ARTIFACT_ROOT=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-28
SSL_DIR="${ARTIFACT_ROOT}/checkpoints/ssl_spotify_ctx2048"

mkdir -p "${ARTIFACT_ROOT}/logs" "${ARTIFACT_ROOT}/wandb" "${ARTIFACT_ROOT}/checkpoints/floras50_finetune_ctx2048"
cd "${REPO}"

if ! compgen -G "${SSL_DIR}/step_*.pt" > /dev/null; then
  echo "No SSL checkpoint found in ${SSL_DIR}" >&2
  exit 2
fi

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

export WANDB_DIR="${ARTIFACT_ROOT}/wandb"
export WANDB_PROJECT=floras50_long_context_ssl_finetune

python exp/train.py \
  -config exp/configs/rob28_floras50_finetune_from_ssl_ctx2048.yaml \
  --num_workers 4 \
  --pin_memory \
  --prefetch 2
