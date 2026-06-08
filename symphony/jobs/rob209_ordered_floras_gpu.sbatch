#!/bin/bash
#SBATCH --job-name=rob209-floras-gpu
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=24:00:00
#SBATCH --mem=130GB
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-209/rob209-floras-gpu-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-209/rob209-floras-gpu-%j.err

set -Eeuo pipefail

export ROB209_REPO_DIR="${ROB209_REPO_DIR:-/users/acp21rjf/long-context-asr}"
export ROB209_REMOTE_BRANCH="${ROB209_REMOTE_BRANCH:-symphony/ROB-209-ordered-floras-bounded-history}"

resolve_rob209_target_commit() {
  local target="${ROB209_TARGET_COMMIT:-}"
  if [[ -z "$target" ]]; then
    target="$(git -C "$ROB209_REPO_DIR" rev-parse "origin/${ROB209_REMOTE_BRANCH}" 2>/dev/null || true)"
  fi
  if [[ -z "$target" ]]; then
    target="$(git -C "$ROB209_REPO_DIR" rev-parse HEAD 2>/dev/null || true)"
  fi
  if [[ -z "$target" ]]; then
    target="$(git -C "$ROB209_REPO_DIR" ls-remote origin "refs/heads/${ROB209_REMOTE_BRANCH}" 2>/dev/null | awk '{print $1}' || true)"
  fi
  if [[ -z "$target" ]]; then
    echo "Unable to resolve ROB209_TARGET_COMMIT for ${ROB209_REMOTE_BRANCH}" >&2
    exit 2
  fi
  printf '%s\n' "$target"
}

export ROB209_TARGET_COMMIT="$(resolve_rob209_target_commit)"
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
