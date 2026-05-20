#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${ROB105_REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN_ID="${ROB105_RUN_ID:-rob105-streaming-rl-grpo-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ROB105_ARTIFACT_ROOT:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-105}"
RUN_DIR="${ROB105_RUN_DIR:-$ARTIFACT_ROOT/$RUN_ID}"
OUT_LOG="${ROB105_OUT_LOG:-$RUN_DIR/${RUN_ID}.out}"
ERR_LOG="${ROB105_ERR_LOG:-$RUN_DIR/${RUN_ID}.err}"
SUMMARY_FILE="${ROB105_SUMMARY_FILE:-$RUN_DIR/${RUN_ID}.summary.txt}"
EXECUTED_SCRIPT="${ROB105_EXECUTED_SCRIPT:-$RUN_DIR/$(basename "$0")}"
CALLBACK_SCRIPT="${ROB105_CALLBACK_SCRIPT:-$RUN_DIR/linear_job_callback.py}"
BASE_CONFIG="${ROB105_BASE_CONFIG:-exp/configs/streaming_decoder_asr_100m.yaml}"
SOURCE_PAIRS="${ROB105_SOURCE_PAIRS:-/store/store5/data/spotify/renamed_audio_text_pairs_10_percent.json}"
MANIFEST_PATH="${ROB105_MANIFEST_PATH:-$RUN_DIR/spotify_10percent_mimas_manifest.json}"
RUNTIME_CONFIG="${ROB105_RUNTIME_CONFIG:-$RUN_DIR/streaming_decoder_asr_100m_rl_grpo.yaml}"
SEED_CHECKPOINT="${ROB105_SEED_CHECKPOINT:-/store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_two_head_rob92_mimas_5epoch_rob92-mimas-5epoch-continuation-20260518T221140Z/step_137895.pt}"
CHECKPOINT_DIR="${ROB105_CHECKPOINT_DIR:-/store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_two_head_rob105_rl_grpo_${RUN_ID}}"
WANDB_DIR="${ROB105_WANDB_DIR:-$ARTIFACT_ROOT/wandb}"
WANDB_NAME="${ROB105_WANDB_NAME:-rob105_streaming_decoder_asr_rl_grpo}"
BATCH_SIZE="${ROB105_BATCH_SIZE:-8}"
MAX_STEPS="${ROB105_MAX_STEPS:-10000}"
SAVE_EVERY="${ROB105_SAVE_EVERY:-500}"
LEARNING_RATE="${ROB105_LEARNING_RATE:-1e-5}"
NUM_ROLLOUTS="${ROB105_NUM_ROLLOUTS:-8}"
TEMPERATURE="${ROB105_TEMPERATURE:-1.0}"
MAX_OUTPUT_FRAMES="${ROB105_MAX_OUTPUT_FRAMES:-96}"
REWARD_STD_MIN="${ROB105_REWARD_STD_MIN:-0.01}"
SAMPLE_TEXT_LOG_EVERY="${ROB105_SAMPLE_TEXT_LOG_EVERY:-10}"
NUM_WORKERS="${ROB105_NUM_WORKERS:-0}"
PREFETCH="${ROB105_PREFETCH:-1}"
PIN_MEMORY="${ROB105_PIN_MEMORY:-0}"
CALLBACK_PYTHON="${ROB105_CALLBACK_PYTHON:-python3}"
LINEAR_KEY_FILE="${ROB105_LINEAR_KEY_FILE:-$ARTIFACT_ROOT/.linear_api_key}"

if [[ "${ROB105_RUN_DIR_EXEC:-0}" != "1" ]]; then
  mkdir -p "$RUN_DIR"
  cp "${BASH_SOURCE[0]}" "$EXECUTED_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_job_callback.py" "$CALLBACK_SCRIPT"
  chmod +x "$EXECUTED_SCRIPT"
  export ROB105_RUN_DIR_EXEC=1
  export ROB105_REPO_DIR="$REPO_DIR"
  export ROB105_RUN_ID="$RUN_ID"
  export ROB105_ARTIFACT_ROOT="$ARTIFACT_ROOT"
  export ROB105_RUN_DIR="$RUN_DIR"
  export ROB105_OUT_LOG="$OUT_LOG"
  export ROB105_ERR_LOG="$ERR_LOG"
  export ROB105_SUMMARY_FILE="$SUMMARY_FILE"
  export ROB105_EXECUTED_SCRIPT="$EXECUTED_SCRIPT"
  export ROB105_CALLBACK_SCRIPT="$CALLBACK_SCRIPT"
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
    echo "max_steps=${MAX_STEPS}"
    echo "save_every=${SAVE_EVERY}"
    echo "learning_rate=${LEARNING_RATE}"
    echo "num_rollouts=${NUM_ROLLOUTS}"
    echo "temperature=${TEMPERATURE}"
    echo "max_output_frames=${MAX_OUTPUT_FRAMES}"
    echo "reward_std_min=${REWARD_STD_MIN}"
    echo "sample_text_log_every=${SAMPLE_TEXT_LOG_EVERY}"
    echo "max_records=${ROB105_MAX_RECORDS:-unset}"
    echo "train_max_steps_override=${ROB105_TRAIN_MAX_STEPS:-unset}"
  } >> "$SUMMARY_FILE"
  if [[ "${ROB105_ENABLE_CALLBACK:-1}" == "1" ]]; then
    if [[ -z "${LINEAR_API_KEY:-}" && -f "$LINEAR_KEY_FILE" ]]; then
      export LINEAR_API_KEY
      LINEAR_API_KEY="$(cat "$LINEAR_KEY_FILE")"
    fi
    callback_args=(
      "$CALLBACK_SCRIPT"
      --issue-id ROB-105
      --state-name Todo
      --job-id "$RUN_ID"
      --job-label "Mimas screen session"
      --exit-code "$status"
      --log-out "$OUT_LOG"
      --log-err "$ERR_LOG"
      --output-path "$CHECKPOINT_DIR"
      --summary-file "$SUMMARY_FILE"
      --title "ROB-105 Mimas streaming decoder RL GRPO run finished"
    )
    if [[ "${ROB105_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
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

if [[ "${ROB105_CALLBACK_ONLY:-0}" == "1" ]]; then
  echo "callback-only dry run requested"
  exit 0
fi

cd "$REPO_DIR"
export PYTHONPATH="$PWD"

python symphony/scripts/prepare_rob105_mimas_streaming_rl.py \
  --base-config "$BASE_CONFIG" \
  --source-pairs "$SOURCE_PAIRS" \
  --manifest-out "$MANIFEST_PATH" \
  --config-out "$RUNTIME_CONFIG" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --seed-checkpoint "$SEED_CHECKPOINT" \
  --wandb-dir "$WANDB_DIR" \
  --wandb-name "$WANDB_NAME" \
  --batch-size "$BATCH_SIZE" \
  --max-steps "$MAX_STEPS" \
  --save-every "$SAVE_EVERY" \
  --learning-rate "$LEARNING_RATE" \
  --num-rollouts "$NUM_ROLLOUTS" \
  --temperature "$TEMPERATURE" \
  --max-output-frames "$MAX_OUTPUT_FRAMES" \
  --reward-std-min "$REWARD_STD_MIN" \
  --sample-text-log-every "$SAMPLE_TEXT_LOG_EVERY"

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
)
if [[ "$PIN_MEMORY" == "1" ]]; then
  train_args+=(-pin_memory)
fi
if [[ -n "${ROB105_MAX_RECORDS:-}" ]]; then
  train_args+=(-max_records "$ROB105_MAX_RECORDS")
fi
if [[ -n "${ROB105_TRAIN_MAX_STEPS:-}" ]]; then
  train_args+=(-max_steps "$ROB105_TRAIN_MAX_STEPS")
fi
if [[ "${ROB105_DISABLE_WANDB:-0}" == "1" ]]; then
  train_args+=(-disable_wandb)
fi
if [[ "${ROB105_VALIDATE_ONLY:-0}" == "1" ]]; then
  train_args+=(--validate_config_only --validate_load_model)
fi
if [[ "${ROB105_SMOKE_ROLLOUT:-0}" == "1" ]]; then
  train_args+=(
    --smoke_rollout
    --smoke_num_rollouts "${ROB105_SMOKE_NUM_ROLLOUTS:-2}"
    --smoke_max_output_frames "${ROB105_SMOKE_MAX_OUTPUT_FRAMES:-8}"
    --smoke_max_records "${ROB105_SMOKE_MAX_RECORDS:-2}"
  )
  if [[ "${ROB105_SMOKE_CPU:-0}" == "1" ]]; then
    train_args+=(--smoke_cpu)
  fi
fi

python exp/train_streaming_decoder_asr_rl.py "${train_args[@]}"
