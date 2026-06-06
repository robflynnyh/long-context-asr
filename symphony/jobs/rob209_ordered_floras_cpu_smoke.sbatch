#!/bin/bash
#SBATCH --job-name=rob209-cpu-smoke
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-209/rob209-cpu-smoke-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-209/rob209-cpu-smoke-%j.err

set -Eeuo pipefail

export ROB209_RUN_ID="${ROB209_RUN_ID:-rob209-ordered-floras-cpu-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
export ROB209_SOURCE_PAIRS="${ROB209_SOURCE_PAIRS:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/manifests/floras50_safe_norm_drop_oov_short2.json}"
export ROB209_CHECKPOINT_DIR="${ROB209_CHECKPOINT_DIR:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-209/checkpoints/${ROB209_RUN_ID}}"
export ROB209_WANDB_NAME="${ROB209_WANDB_NAME:-rob209_ordered_floras_cpu_smoke}"
export ROB209_BATCH_SIZE="${ROB209_BATCH_SIZE:-1}"
export ROB209_MAX_EPOCHS="${ROB209_MAX_EPOCHS:-1}"
export ROB209_MAX_RECORDS="${ROB209_MAX_RECORDS:-2}"
export ROB209_MAX_STEPS="${ROB209_MAX_STEPS:-1}"
export ROB209_DISABLE_WANDB="${ROB209_DISABLE_WANDB:-1}"
export ROB209_DEBUG_GENERATE_EVERY_RECORDS="${ROB209_DEBUG_GENERATE_EVERY_RECORDS:-0}"
export ROB209_VALIDATE_PATH_LIMIT="${ROB209_VALIDATE_PATH_LIMIT:-2}"
export ROB209_CALLBACK_DRY_RUN="${ROB209_CALLBACK_DRY_RUN:-1}"

bash "${ROB209_REPO_DIR:-/users/acp21rjf/long-context-asr}/symphony/scripts/run_rob209_ordered_floras_stanage.sh"
