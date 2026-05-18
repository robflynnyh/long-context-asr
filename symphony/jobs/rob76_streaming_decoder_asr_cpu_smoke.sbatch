#!/bin/bash
#SBATCH --job-name=rob76-stream-cpu-smoke
#SBATCH --partition=interactive
#SBATCH --time=00:20:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-76/rob76-stream-cpu-smoke-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-76/rob76-stream-cpu-smoke-%j.err

set -Eeuo pipefail

REPO_DIR="${ROB76_REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-76}"
ARTIFACT_DIR="${ROB76_ARTIFACT_DIR:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-76}"
CONFIG_PATH="${ROB76_CONFIG_PATH:-exp/configs/streaming_decoder_asr_100m.yaml}"
MAX_RECORDS="${ROB76_SMOKE_MAX_RECORDS:-1}"
MAX_STEPS="${ROB76_SMOKE_MAX_STEPS:-1}"
BATCH_SIZE="${ROB76_SMOKE_BATCH_SIZE:-1}"
JOB_ID="${SLURM_JOB_ID:-manual}"
CHECKPOINT_DIR="${ROB76_CHECKPOINT_DIR:-$ARTIFACT_DIR/cpu-smoke-checkpoints-${JOB_ID}}"

mkdir -p "$ARTIFACT_DIR" "$CHECKPOINT_DIR"

cd "$REPO_DIR"
if [[ -n "${ROB76_TARGET_COMMIT:-}" ]]; then
  git fetch origin symphony/ROB-76-streaming-decoder-asr
  git checkout --detach "$ROB76_TARGET_COMMIT"
fi
export PYTHONPATH="$PWD"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

echo "job_id=${JOB_ID}"
echo "host=$(hostname)"
echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "config=${CONFIG_PATH}"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
echo "max_records=${MAX_RECORDS}"
echo "max_steps=${MAX_STEPS}"
echo "batch_size=${BATCH_SIZE}"

python exp/train_streaming_decoder_asr.py \
  -config "$CONFIG_PATH" \
  -checkpoint_dir "$CHECKPOINT_DIR" \
  -disable_wandb \
  -max_records "$MAX_RECORDS" \
  -max_steps "$MAX_STEPS" \
  -batch_size "$BATCH_SIZE" \
  -num_workers 0 \
  -prefetch 1
