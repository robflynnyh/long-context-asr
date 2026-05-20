#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${ROB92_REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN_ID="${ROB92_RUN_ID:-rob92-mimas-5epoch-continuation-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ROB92_ARTIFACT_ROOT:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-92}"
RUN_DIR="${ROB92_RUN_DIR:-$ARTIFACT_ROOT/$RUN_ID}"
OUT_LOG="${ROB92_OUT_LOG:-$RUN_DIR/${RUN_ID}.out}"
ERR_LOG="${ROB92_ERR_LOG:-$RUN_DIR/${RUN_ID}.err}"
SUMMARY_FILE="${ROB92_SUMMARY_FILE:-$RUN_DIR/${RUN_ID}.summary.txt}"
EXECUTED_SCRIPT="${ROB92_EXECUTED_SCRIPT:-$RUN_DIR/$(basename "$0")}"
CALLBACK_SCRIPT="${ROB92_CALLBACK_SCRIPT:-$RUN_DIR/linear_job_callback.py}"
BASE_CONFIG="${ROB92_BASE_CONFIG:-exp/configs/streaming_decoder_asr_100m.yaml}"
SOURCE_PAIRS="${ROB92_SOURCE_PAIRS:-/store/store5/data/spotify/renamed_audio_text_pairs_10_percent.json}"
MANIFEST_PATH="${ROB92_MANIFEST_PATH:-$RUN_DIR/spotify_10percent_mimas_manifest.json}"
RUNTIME_CONFIG="${ROB92_RUNTIME_CONFIG:-$RUN_DIR/streaming_decoder_asr_100m_mimas_5epoch_continuation.yaml}"
SEED_CHECKPOINT="${ROB92_SEED_CHECKPOINT:-/store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_two_head_mimas_full_epoch_rob76-mimas-3epoch-b48-two-head-20260517T140145Z/step_82737.pt}"
CHECKPOINT_DIR="${ROB92_CHECKPOINT_DIR:-/store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_two_head_rob92_mimas_5epoch_${RUN_ID}}"
WANDB_DIR="${ROB92_WANDB_DIR:-$ARTIFACT_ROOT/wandb}"
WANDB_NAME="${ROB92_WANDB_NAME:-rob92_streaming_decoder_asr_5epoch_continuation}"
BATCH_SIZE="${ROB92_BATCH_SIZE:-48}"
MAX_EPOCHS="${ROB92_MAX_EPOCHS:-5}"
LEARNING_RATE="${ROB92_LEARNING_RATE:-5e-5}"
DEBUG_GENERATE_EVERY_RECORDS="${ROB92_DEBUG_GENERATE_EVERY_RECORDS:-500}"
DEBUG_GENERATE_MAX_FRAMES="${ROB92_DEBUG_GENERATE_MAX_FRAMES:-0}"
NUM_WORKERS="${ROB92_NUM_WORKERS:-0}"
PREFETCH="${ROB92_PREFETCH:-1}"
PIN_MEMORY="${ROB92_PIN_MEMORY:-0}"
CALLBACK_PYTHON="${ROB92_CALLBACK_PYTHON:-python3}"
LINEAR_KEY_FILE="${ROB92_LINEAR_KEY_FILE:-$ARTIFACT_ROOT/.linear_api_key}"

if [[ "${ROB92_RUN_DIR_EXEC:-0}" != "1" ]]; then
  mkdir -p "$RUN_DIR"
  cp "${BASH_SOURCE[0]}" "$EXECUTED_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_job_callback.py" "$CALLBACK_SCRIPT"
  chmod +x "$EXECUTED_SCRIPT"
  export ROB92_RUN_DIR_EXEC=1
  export ROB92_REPO_DIR="$REPO_DIR"
  export ROB92_RUN_ID="$RUN_ID"
  export ROB92_ARTIFACT_ROOT="$ARTIFACT_ROOT"
  export ROB92_RUN_DIR="$RUN_DIR"
  export ROB92_OUT_LOG="$OUT_LOG"
  export ROB92_ERR_LOG="$ERR_LOG"
  export ROB92_SUMMARY_FILE="$SUMMARY_FILE"
  export ROB92_EXECUTED_SCRIPT="$EXECUTED_SCRIPT"
  export ROB92_CALLBACK_SCRIPT="$CALLBACK_SCRIPT"
  exec bash "$EXECUTED_SCRIPT" "$@"
fi

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
    echo "seed_checkpoint=${SEED_CHECKPOINT}"
    echo "checkpoint_dir=${CHECKPOINT_DIR}"
    echo "wandb_dir=${WANDB_DIR}"
    echo "wandb_name=${WANDB_NAME}"
    echo "batch_size=${BATCH_SIZE}"
    echo "max_epochs=${MAX_EPOCHS}"
    echo "learning_rate=${LEARNING_RATE}"
    echo "debug_generate_every_records=${DEBUG_GENERATE_EVERY_RECORDS}"
    echo "debug_generate_max_frames=${DEBUG_GENERATE_MAX_FRAMES}"
    echo "max_records=${ROB92_MAX_RECORDS:-unset}"
    echo "max_steps=${ROB92_MAX_STEPS:-unset}"
  } >> "$SUMMARY_FILE"
  if [[ "${ROB92_ENABLE_CALLBACK:-1}" == "1" ]]; then
    if [[ -z "${LINEAR_API_KEY:-}" && -f "$LINEAR_KEY_FILE" ]]; then
      export LINEAR_API_KEY
      LINEAR_API_KEY="$(cat "$LINEAR_KEY_FILE")"
    fi
    callback_args=(
      "$CALLBACK_SCRIPT"
      --issue-id ROB-92
      --state-name Todo
      --job-id "$RUN_ID"
      --job-label "Mimas screen session"
      --exit-code "$status"
      --log-out "$OUT_LOG"
      --log-err "$ERR_LOG"
      --output-path "$CHECKPOINT_DIR"
      --summary-file "$SUMMARY_FILE"
      --title "ROB-92 Mimas streaming decoder 5-epoch continuation finished"
    )
    if [[ "${ROB92_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
      callback_args+=(--dry-run)
    fi
    "$CALLBACK_PYTHON" "${callback_args[@]}" >> "$SUMMARY_FILE" 2>&1
  fi
  return "$status"
}
trap on_exit EXIT
trap 'trap - TERM INT; exit 143' TERM
trap 'trap - INT TERM; exit 130' INT

{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "run_id=${RUN_ID}"
  echo "host=$(hostname)"
  echo "repo_dir=${REPO_DIR}"
  echo "commit=$(cd "$REPO_DIR" && git rev-parse HEAD)"
  echo "executed_script=${EXECUTED_SCRIPT}"
  echo "callback_script=${CALLBACK_SCRIPT}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
} > "$SUMMARY_FILE"

if [[ "${ROB92_CALLBACK_ONLY:-0}" == "1" ]]; then
  echo "callback-only dry run requested"
  exit 0
fi

cd "$REPO_DIR"
export PYTHONPATH="$PWD"

python symphony/scripts/prepare_rob92_mimas_continuation.py \
  --base-config "$BASE_CONFIG" \
  --source-pairs "$SOURCE_PAIRS" \
  --manifest-out "$MANIFEST_PATH" \
  --config-out "$RUNTIME_CONFIG" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --seed-checkpoint "$SEED_CHECKPOINT" \
  --wandb-dir "$WANDB_DIR" \
  --wandb-name "$WANDB_NAME" \
  --batch-size "$BATCH_SIZE" \
  --max-epochs "$MAX_EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --debug-generate-every-records "$DEBUG_GENERATE_EVERY_RECORDS" \
  --debug-generate-max-frames "$DEBUG_GENERATE_MAX_FRAMES"

echo "runtime_config=${RUNTIME_CONFIG}"
echo "seed_checkpoint=${SEED_CHECKPOINT}"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
echo "num_workers=${NUM_WORKERS}"
echo "prefetch=${PREFETCH}"
echo "pin_memory=${PIN_MEMORY}"

train_args=(
  -config "$RUNTIME_CONFIG"
  -num_workers "$NUM_WORKERS"
  -prefetch "$PREFETCH"
  -rm_sched
)
if [[ "$PIN_MEMORY" == "1" ]]; then
  train_args+=(-pin_memory)
fi
if [[ -n "${ROB92_MAX_RECORDS:-}" ]]; then
  train_args+=(-max_records "$ROB92_MAX_RECORDS")
fi
if [[ -n "${ROB92_MAX_STEPS:-}" ]]; then
  train_args+=(-max_steps "$ROB92_MAX_STEPS")
fi
if [[ "${ROB92_DISABLE_WANDB:-0}" == "1" ]]; then
  train_args+=(-disable_wandb)
fi

python exp/train_streaming_decoder_asr.py "${train_args[@]}"
