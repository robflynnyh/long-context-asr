#!/bin/bash
#SBATCH --job-name=rob70-bestrq-ssl
#SBATCH --partition=gpu-h100-nvl
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --time=36:00:00
#SBATCH --mem=150GB
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-70/gpu-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-70/gpu-%j.err

set -uo pipefail

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-70}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-70}"
CHECKPOINT_DIR="/mnt/parscratch/users/acp21rjf/spotify/bestrq_ssl/6l_2048_1epoch_retry_20260515"
TRAIN_LOG="${ARTIFACT_DIR}/train-${SLURM_JOB_ID:-manual}.log"
WANDB_NAME="rob70_bestrq_6l_2048_1epoch_retry_20260515"
COMMAND="python exp/train_bestRQ.py -config exp/configs/ssl/bestrq_6l_2048_spotify.yaml --remove_scheduler --reset_step --no_resume --num_workers ${ROB70_NUM_WORKERS:-0} --prefetch_factor ${ROB70_PREFETCH_FACTOR:-1}"
LINEAR_ENV_FILE="${LINEAR_ENV_FILE:-}"

if [[ -z "$LINEAR_ENV_FILE" ]]; then
  if [[ -f "$HOME/.config/long-context-asr/linear.env" ]]; then
    LINEAR_ENV_FILE="$HOME/.config/long-context-asr/linear.env"
  elif [[ -f "$HOME/.config/sap-longcontext/linear.env" ]]; then
    LINEAR_ENV_FILE="$HOME/.config/sap-longcontext/linear.env"
  fi
fi

on_exit() {
  status=$?
  set +e
  if [[ "${ROB70_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
    python "$REPO_DIR/symphony/rob70_linear_callback.py" \
      --exit-code "$status" \
      --slurm-job-id "${SLURM_JOB_ID:-manual}" \
      --train-log "$TRAIN_LOG" \
      --checkpoint-dir "$CHECKPOINT_DIR" \
      --wandb-name "$WANDB_NAME" \
      --command "$COMMAND" \
      --dry-run
  else
    python "$REPO_DIR/symphony/rob70_linear_callback.py" \
      --exit-code "$status" \
      --slurm-job-id "${SLURM_JOB_ID:-manual}" \
      --train-log "$TRAIN_LOG" \
      --checkpoint-dir "$CHECKPOINT_DIR" \
      --wandb-name "$WANDB_NAME" \
      --command "$COMMAND"
  fi
  callback_status=$?
  if [[ "$callback_status" -ne 0 ]]; then
    echo "ROB-70 Linear callback failed with exit status ${callback_status}" >&2
  fi
  exit "$status"
}
trap on_exit EXIT

set -e

cd "$REPO_DIR"
mkdir -p "$ARTIFACT_DIR" "$CHECKPOINT_DIR"

if [[ -n "$LINEAR_ENV_FILE" && -f "$LINEAR_ENV_FILE" ]]; then
  set -a
  source "$LINEAR_ENV_FILE"
  set +a
fi

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

export PYTHONPATH="$REPO_DIR"
export WANDB_DIR="/mnt/parscratch/users/acp21rjf/wandb_log"

if [[ "${ROB70_CALLBACK_ONLY:-0}" == "1" ]]; then
  echo "ROB70 callback-only validation"
  exit 0
fi

set -o pipefail
$COMMAND 2>&1 | tee "$TRAIN_LOG"
