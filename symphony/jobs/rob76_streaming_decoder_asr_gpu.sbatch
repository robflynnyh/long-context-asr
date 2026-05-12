#!/bin/bash
#SBATCH --job-name=rob76-stream-train
#SBATCH --partition=gpu-h100-nvl
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --time=90:00:00
#SBATCH --mem=130GB
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-76/rob76-stream-train-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-76/rob76-stream-train-%j.err

set -Eeuo pipefail

REPO_DIR="${ROB76_REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-76}"
ARTIFACT_DIR="${ROB76_ARTIFACT_DIR:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-76}"
CONFIG_PATH="${ROB76_CONFIG_PATH:-exp/configs/streaming_decoder_asr_100m.yaml}"
CHECKPOINT_DIR="${ROB76_CHECKPOINT_DIR:-/mnt/parscratch/users/acp21rjf/spotify/streaming_decoder_asr_100m}"
LINEAR_KEY_FILE="${ROB76_LINEAR_KEY_FILE:-$ARTIFACT_DIR/.linear_api_key}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT_LOG="$ARTIFACT_DIR/rob76-stream-train-${JOB_ID}.out"
ERR_LOG="$ARTIFACT_DIR/rob76-stream-train-${JOB_ID}.err"
SUMMARY_FILE="$ARTIFACT_DIR/rob76-stream-train-${JOB_ID}.summary.txt"
CALLBACK_PYTHON="${ROB76_CALLBACK_PYTHON:-python3}"

on_exit() {
  local status=$?
  set +e
  {
    echo "ended_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_code=${status}"
    echo "job_id=${JOB_ID}"
    echo "repo_dir=${REPO_DIR}"
    echo "target_commit=${ROB76_TARGET_COMMIT:-unset}"
    echo "config=${CONFIG_PATH}"
    echo "checkpoint_dir=${CHECKPOINT_DIR}"
  } >> "$SUMMARY_FILE"
  if [[ "${ROB76_ENABLE_CALLBACK:-1}" == "1" ]]; then
    if [[ -z "${LINEAR_API_KEY:-}" && -f "$LINEAR_KEY_FILE" ]]; then
      export LINEAR_API_KEY
      LINEAR_API_KEY="$(cat "$LINEAR_KEY_FILE")"
    fi
    local dry_run_args=()
    if [[ "${ROB76_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
      dry_run_args+=(--dry-run)
    fi
    "$CALLBACK_PYTHON" "$REPO_DIR/symphony/scripts/linear_job_callback.py" \
      --issue-id ROB-76 \
      --state-name Todo \
      --job-id "$JOB_ID" \
      --exit-code "$status" \
      --log-out "$OUT_LOG" \
      --log-err "$ERR_LOG" \
      --output-path "$CHECKPOINT_DIR" \
      --summary-file "$SUMMARY_FILE" \
      --title "ROB-76 streaming decoder GPU training finished" \
      "${dry_run_args[@]}" >> "$SUMMARY_FILE" 2>&1
  fi
  return "$status"
}
trap on_exit EXIT
trap 'trap - TERM INT; exit 143' TERM
trap 'trap - TERM INT; exit 130' INT

mkdir -p "$ARTIFACT_DIR" "$CHECKPOINT_DIR"
{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "job_id=${JOB_ID}"
  echo "host=$(hostname)"
} > "$SUMMARY_FILE"

if [[ "${ROB76_CALLBACK_ONLY:-0}" == "1" ]]; then
  echo "callback-only dry run requested"
  exit 0
fi

: "${ROB76_TARGET_COMMIT:?ROB76_TARGET_COMMIT must pin the code revision}"

cd "$REPO_DIR"
git fetch origin symphony/ROB-76-streaming-decoder-asr
git checkout --detach "$ROB76_TARGET_COMMIT"
export PYTHONPATH="$PWD"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "commit=$(git rev-parse HEAD)"
echo "config=${CONFIG_PATH}"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
python exp/train_streaming_decoder_asr.py \
  -config "$CONFIG_PATH" \
  -num_workers 8 \
  -pin_memory \
  -prefetch 2 \
  -checkpoint_dir "$CHECKPOINT_DIR"
