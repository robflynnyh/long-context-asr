#!/bin/bash
#SBATCH --job-name=rob100-ssl-smoke
#SBATCH --partition=interactive
#SBATCH --time=00:30:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=2
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-100/smoke-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-100/smoke-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-100}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-100}"
CONFIG_PATH="${ROB100_SMOKE_CONFIG:-exp/configs/ssl/bestrq_6l_2048_spotify_papermask_smoke.yaml}"

cd "$REPO_DIR"
mkdir -p "$ARTIFACT_DIR" "$ARTIFACT_DIR/smoke-checkpoints"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

export PYTHONPATH="$REPO_DIR"
export CUDA_VISIBLE_DEVICES=""

python exp/train_bestRQ.py \
  -config "$CONFIG_PATH" \
  --remove_scheduler \
  --reset_step \
  --no_resume \
  --num_workers 0 \
  --prefetch_factor 1
