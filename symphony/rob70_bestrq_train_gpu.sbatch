#!/bin/bash
#SBATCH --job-name=rob70-bestrq-ssl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=36:00:00
#SBATCH --mem=82GB
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-70/gpu-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-70/gpu-%j.err

set -uo pipefail

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-70}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-70}"
CHECKPOINT_DIR="/mnt/parscratch/users/acp21rjf/spotify/bestrq_ssl/6l_2048_1epoch"
TRAIN_LOG="${ARTIFACT_DIR}/train-${SLURM_JOB_ID:-manual}.log"
COMMAND="python exp/train_bestRQ.py -config exp/configs/ssl/bestrq_6l_2048_spotify.yaml --remove_scheduler --reset_step --num_workers 4 --pin_memory"

on_exit() {
  status=$?
  set +e
  dry_args=()
  if [[ "${ROB70_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
    dry_args+=(--dry-run)
  fi
  python "$REPO_DIR/symphony/rob70_linear_callback.py" \
    --exit-code "$status" \
    --slurm-job-id "${SLURM_JOB_ID:-manual}" \
    --train-log "$TRAIN_LOG" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --wandb-name "rob70_bestrq_6l_2048_1epoch" \
    --command "$COMMAND" \
    "${dry_args[@]}"
  exit "$status"
}
trap on_exit EXIT

set -e

cd "$REPO_DIR"
mkdir -p "$ARTIFACT_DIR" "$CHECKPOINT_DIR"

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

