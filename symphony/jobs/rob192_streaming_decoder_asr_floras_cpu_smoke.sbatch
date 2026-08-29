#!/bin/bash
#SBATCH --job-name=rob192-ft-smoke
#SBATCH --partition=interactive
#SBATCH --time=02:00:00
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/rob192-floras-smoke-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/rob192-floras-smoke-%j.err

set -euo pipefail

REPO_DIR="${ROB192_REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-192}"
cd "$REPO_DIR"

export ROB192_REPO_DIR="$REPO_DIR"
export ROB192_OUT_LOG="${ROB192_OUT_LOG:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/rob192-floras-smoke-${SLURM_JOB_ID}.out}"
export ROB192_ERR_LOG="${ROB192_ERR_LOG:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/rob192-floras-smoke-${SLURM_JOB_ID}.err}"
export ROB192_RUN_ID="${ROB192_RUN_ID:-rob192-rope-streaming-floras50-cpu-smoke-${SLURM_JOB_ID}}"
export ROB192_CALLBACK_DRY_RUN="${ROB192_CALLBACK_DRY_RUN:-1}"
export ROB192_DISABLE_WANDB="${ROB192_DISABLE_WANDB:-1}"
export ROB192_DEBUG_GENERATE_EVERY_RECORDS="${ROB192_DEBUG_GENERATE_EVERY_RECORDS:-0}"
export ROB192_VALIDATE_PATH_LIMIT="${ROB192_VALIDATE_PATH_LIMIT:-50}"
export ROB192_MAX_RECORDS="${ROB192_MAX_RECORDS:-2}"
export ROB192_MAX_STEPS="${ROB192_MAX_STEPS:-1}"

bash symphony/scripts/run_rob192_streaming_decoder_asr_floras_stanage.sh
