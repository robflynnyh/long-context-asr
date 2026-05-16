#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${ROB76_REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN_ID="${ROB76_RUN_ID:-rob76-mimas-full-epoch-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ROB76_ARTIFACT_ROOT:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-76}"
RUN_DIR="${ROB76_RUN_DIR:-$ARTIFACT_ROOT/$RUN_ID}"
OUT_LOG="${ROB76_OUT_LOG:-$RUN_DIR/${RUN_ID}.out}"
ERR_LOG="${ROB76_ERR_LOG:-$RUN_DIR/${RUN_ID}.err}"
SUMMARY_FILE="${ROB76_SUMMARY_FILE:-$RUN_DIR/${RUN_ID}.summary.txt}"
BASE_CONFIG="${ROB76_BASE_CONFIG:-exp/configs/streaming_decoder_asr_100m.yaml}"
SOURCE_PAIRS="${ROB76_SOURCE_PAIRS:-/store/store5/data/spotify/renamed_audio_text_pairs_10_percent.json}"
MANIFEST_PATH="${ROB76_MANIFEST_PATH:-$RUN_DIR/spotify_10percent_mimas_manifest.json}"
RUNTIME_CONFIG="${ROB76_RUNTIME_CONFIG:-$RUN_DIR/streaming_decoder_asr_100m_mimas_full_epoch.yaml}"
CHECKPOINT_DIR="${ROB76_CHECKPOINT_DIR:-/store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_mimas_full_epoch_${RUN_ID}}"
WANDB_DIR="${ROB76_WANDB_DIR:-$ARTIFACT_ROOT/wandb}"
WANDB_NAME="${ROB76_WANDB_NAME:-streaming_decoder_asr_100m_mimas_full_epoch}"
BATCH_SIZE="${ROB76_BATCH_SIZE:-48}"
MAX_EPOCHS="${ROB76_MAX_EPOCHS:-1}"
LEARNING_RATE="${ROB76_LEARNING_RATE:-5e-5}"
DEBUG_GENERATE_EVERY_RECORDS="${ROB76_DEBUG_GENERATE_EVERY_RECORDS:-500}"
DEBUG_GENERATE_MAX_FRAMES="${ROB76_DEBUG_GENERATE_MAX_FRAMES:-0}"
NUM_WORKERS="${ROB76_NUM_WORKERS:-0}"
PREFETCH="${ROB76_PREFETCH:-1}"
PIN_MEMORY="${ROB76_PIN_MEMORY:-0}"
CALLBACK_PYTHON="${ROB76_CALLBACK_PYTHON:-python3}"
LINEAR_KEY_FILE="${ROB76_LINEAR_KEY_FILE:-$ARTIFACT_ROOT/.linear_api_key}"

mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR" "$WANDB_DIR"
exec > >(tee -a "$OUT_LOG") 2> >(tee -a "$ERR_LOG" >&2)

on_exit() {
  local status=$?
  set +e
  {
    echo "ended_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_code=${status}"
    echo "run_id=${RUN_ID}"
    echo "host=$(hostname)"
    echo "repo_dir=${REPO_DIR}"
    echo "commit=$(cd "$REPO_DIR" && git rev-parse HEAD 2>/dev/null)"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "runtime_config=${RUNTIME_CONFIG}"
    echo "manifest=${MANIFEST_PATH}"
    echo "checkpoint_dir=${CHECKPOINT_DIR}"
    echo "wandb_dir=${WANDB_DIR}"
    echo "wandb_name=${WANDB_NAME}"
    echo "batch_size=${BATCH_SIZE}"
    echo "max_epochs=${MAX_EPOCHS}"
    echo "learning_rate=${LEARNING_RATE}"
    echo "debug_generate_every_records=${DEBUG_GENERATE_EVERY_RECORDS}"
    echo "debug_generate_max_frames=${DEBUG_GENERATE_MAX_FRAMES}"
    echo "max_records=${ROB76_MAX_RECORDS:-unset}"
    echo "max_steps=${ROB76_MAX_STEPS:-unset}"
  } >> "$SUMMARY_FILE"
  if [[ "${ROB76_ENABLE_CALLBACK:-1}" == "1" ]]; then
    if [[ -z "${LINEAR_API_KEY:-}" && -f "$LINEAR_KEY_FILE" ]]; then
      export LINEAR_API_KEY
      LINEAR_API_KEY="$(cat "$LINEAR_KEY_FILE")"
    fi
    if [[ "${ROB76_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
      "$CALLBACK_PYTHON" "$REPO_DIR/symphony/scripts/linear_job_callback.py" \
        --issue-id ROB-76 \
        --state-name Todo \
        --job-id "$RUN_ID" \
        --job-label "Mimas session" \
        --exit-code "$status" \
        --log-out "$OUT_LOG" \
        --log-err "$ERR_LOG" \
        --output-path "$CHECKPOINT_DIR" \
        --summary-file "$SUMMARY_FILE" \
        --title "ROB-76 Mimas streaming decoder full-epoch training finished" \
        --dry-run >> "$SUMMARY_FILE" 2>&1
    else
      "$CALLBACK_PYTHON" "$REPO_DIR/symphony/scripts/linear_job_callback.py" \
        --issue-id ROB-76 \
        --state-name Todo \
        --job-id "$RUN_ID" \
        --job-label "Mimas session" \
        --exit-code "$status" \
        --log-out "$OUT_LOG" \
        --log-err "$ERR_LOG" \
        --output-path "$CHECKPOINT_DIR" \
        --summary-file "$SUMMARY_FILE" \
        --title "ROB-76 Mimas streaming decoder full-epoch training finished" >> "$SUMMARY_FILE" 2>&1
    fi
  fi
  return "$status"
}
trap on_exit EXIT
trap 'trap - TERM INT; exit 143' TERM
trap 'trap - TERM INT; exit 130' INT

{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "run_id=${RUN_ID}"
  echo "host=$(hostname)"
  echo "repo_dir=${REPO_DIR}"
  echo "commit=$(cd "$REPO_DIR" && git rev-parse HEAD)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
} > "$SUMMARY_FILE"

if [[ "${ROB76_CALLBACK_ONLY:-0}" == "1" ]]; then
  echo "callback-only dry run requested"
  exit 0
fi

cd "$REPO_DIR"
export PYTHONPATH="$PWD"

python symphony/scripts/prepare_rob76_mimas_full_epoch.py \
  --base-config "$BASE_CONFIG" \
  --source-pairs "$SOURCE_PAIRS" \
  --manifest-out "$MANIFEST_PATH" \
  --config-out "$RUNTIME_CONFIG" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --wandb-dir "$WANDB_DIR" \
  --wandb-name "$WANDB_NAME" \
  --batch-size "$BATCH_SIZE" \
  --max-epochs "$MAX_EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --debug-generate-every-records "$DEBUG_GENERATE_EVERY_RECORDS" \
  --debug-generate-max-frames "$DEBUG_GENERATE_MAX_FRAMES"

echo "runtime_config=${RUNTIME_CONFIG}"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
echo "num_workers=${NUM_WORKERS}"
echo "prefetch=${PREFETCH}"
echo "pin_memory=${PIN_MEMORY}"

train_args=(
  -config "$RUNTIME_CONFIG"
  -num_workers "$NUM_WORKERS"
  -prefetch "$PREFETCH"
  -reset_step
)
if [[ "$PIN_MEMORY" == "1" ]]; then
  train_args+=(-pin_memory)
fi
if [[ -n "${ROB76_MAX_RECORDS:-}" ]]; then
  train_args+=(-max_records "$ROB76_MAX_RECORDS")
fi
if [[ -n "${ROB76_MAX_STEPS:-}" ]]; then
  train_args+=(-max_steps "$ROB76_MAX_STEPS")
fi
if [[ "${ROB76_DISABLE_WANDB:-0}" == "1" ]]; then
  train_args+=(-disable_wandb)
fi

python exp/train_streaming_decoder_asr.py "${train_args[@]}"
