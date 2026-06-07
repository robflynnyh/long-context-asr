#!/bin/bash
#SBATCH --job-name=rob209-floras-gpu
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=80:00:00
#SBATCH --mem=130GB
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-209/rob209-floras-gpu-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-209/rob209-floras-gpu-%j.err

set -Eeuo pipefail

export ROB209_REPO_DIR="${ROB209_REPO_DIR:-/users/acp21rjf/long-context-asr}"
export ROB209_REMOTE_BRANCH="${ROB209_REMOTE_BRANCH:-symphony/ROB-209-ordered-floras-bounded-history}"
export ROB209_TARGET_COMMIT="${ROB209_TARGET_COMMIT:-$(git -C "$ROB209_REPO_DIR" ls-remote origin "refs/heads/${ROB209_REMOTE_BRANCH}" | awk '{print $1}')}"
export ROB209_RUN_ID="${ROB209_RUN_ID:-rob209-ordered-floras-bounded-history-$(date -u +%Y%m%dT%H%M%SZ)}"
export ROB209_CHECKPOINT_DIR="${ROB209_CHECKPOINT_DIR:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-209/checkpoints/${ROB209_RUN_ID}}"
export ROB209_WANDB_NAME="${ROB209_WANDB_NAME:-rob209_ordered_floras_bounded_history}"
export ROB209_BATCH_SIZE="${ROB209_BATCH_SIZE:-88}"
export ROB209_MAX_EPOCHS="${ROB209_MAX_EPOCHS:-12}"
export ROB209_LEARNING_RATE="${ROB209_LEARNING_RATE:-5e-5}"
export ROB209_SAVE_EVERY_N_STEPS="${ROB209_SAVE_EVERY_N_STEPS:-2000}"
export ROB209_NUM_WORKERS="${ROB209_NUM_WORKERS:-0}"
export ROB209_PREFETCH="${ROB209_PREFETCH:-1}"

bash "$ROB209_REPO_DIR/symphony/scripts/run_rob209_ordered_floras_stanage.sh"
