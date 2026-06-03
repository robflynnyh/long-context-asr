#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${ROB127_REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN_ID="${ROB127_RUN_ID:-rob127-rope-streaming-rl-grpo-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ROB127_ARTIFACT_ROOT:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-127}"
RUN_DIR="${ROB127_RUN_DIR:-$ARTIFACT_ROOT/$RUN_ID}"
OUT_LOG="${ROB127_OUT_LOG:-$RUN_DIR/${RUN_ID}.out}"
ERR_LOG="${ROB127_ERR_LOG:-$RUN_DIR/${RUN_ID}.err}"
SUMMARY_FILE="${ROB127_SUMMARY_FILE:-$RUN_DIR/${RUN_ID}.summary.txt}"
EXECUTED_SCRIPT="${ROB127_EXECUTED_SCRIPT:-$RUN_DIR/$(basename "$0")}"
CALLBACK_SCRIPT="${ROB127_CALLBACK_SCRIPT:-$RUN_DIR/linear_job_callback.py}"
BASE_CONFIG="${ROB127_BASE_CONFIG:-exp/configs/streaming_decoder_asr_100m.yaml}"
SOURCE_PAIRS="${ROB127_SOURCE_PAIRS:-/store/store5/data/spotify/renamed_audio_text_pairs_10_percent.json}"
MANIFEST_PATH="${ROB127_MANIFEST_PATH:-$RUN_DIR/spotify_10percent_mimas_manifest.json}"
RUNTIME_CONFIG="${ROB127_RUNTIME_CONFIG:-$RUN_DIR/streaming_decoder_asr_100m_rope_rl_grpo.yaml}"
SEED_CHECKPOINT="${ROB127_SEED_CHECKPOINT:-/store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_rope_rob116_mimas_5epoch_rob116-rope-mimas-5epoch-20260521T153748Z/step_98629.pt}"
CHECKPOINT_DIR="${ROB127_CHECKPOINT_DIR:-/store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_rope_rob127_rl_grpo_${RUN_ID}}"
WANDB_DIR="${ROB127_WANDB_DIR:-$ARTIFACT_ROOT/wandb}"
TMPDIR="${ROB127_TMPDIR:-$RUN_DIR/tmp}"
WANDB_NAME="${ROB127_WANDB_NAME:-rob127_rope_streaming_decoder_asr_rl_grpo}"
WANDB_ID="${ROB127_WANDB_ID:-}"
BATCH_SIZE="${ROB127_BATCH_SIZE:-32}"
MAX_STEPS="${ROB127_MAX_STEPS:-10000}"
SAVE_EVERY="${ROB127_SAVE_EVERY:-500}"
LEARNING_RATE="${ROB127_LEARNING_RATE:-1e-6}"
NUM_ROLLOUTS="${ROB127_NUM_ROLLOUTS:-6}"
MICROBATCH_SIZE="${ROB127_MICROBATCH_SIZE:-16}"
LOGPROB_MICROBATCH_SIZE="${ROB127_LOGPROB_MICROBATCH_SIZE:-$MICROBATCH_SIZE}"
TEMPERATURE="${ROB127_TEMPERATURE:-1.0}"
MAX_OUTPUT_FRAMES="${ROB127_MAX_OUTPUT_FRAMES:-}"
REWARD_STD_MIN="${ROB127_REWARD_STD_MIN:-0.01}"
SAMPLE_TEXT_LOG_EVERY="${ROB127_SAMPLE_TEXT_LOG_EVERY:-10}"
LATE_WORD_TOLERANCE_SECONDS="${ROB127_LATE_WORD_TOLERANCE_SECONDS:-2.0}"
LATE_WORD_PENALTY_MODE="${ROB127_LATE_WORD_PENALTY_MODE:-constant}"
LATE_WORD_PENALTY_PER_SECOND="${ROB127_LATE_WORD_PENALTY_PER_SECOND:-0.25}"
LATE_WORD_PENALTY_MAX="${ROB127_LATE_WORD_PENALTY_MAX:-1.0}"
ROTARY_BASE_FREQ="${ROB127_ROTARY_BASE_FREQ:-1500000}"
ROTARY_INTERPOLATION_FACTOR="${ROB127_ROTARY_INTERPOLATION_FACTOR:-1.0}"
NUM_WORKERS="${ROB127_NUM_WORKERS:-0}"
PREFETCH="${ROB127_PREFETCH:-1}"
PIN_MEMORY="${ROB127_PIN_MEMORY:-0}"
CALLBACK_PYTHON="${ROB127_CALLBACK_PYTHON:-python3}"
LINEAR_KEY_FILE="${ROB127_LINEAR_KEY_FILE:-$ARTIFACT_ROOT/.linear_api_key}"

if [[ "${ROB127_RUN_DIR_EXEC:-0}" != "1" ]]; then
  mkdir -p "$RUN_DIR"
  cp "${BASH_SOURCE[0]}" "$EXECUTED_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_job_callback.py" "$CALLBACK_SCRIPT"
  chmod +x "$EXECUTED_SCRIPT"
  export ROB127_RUN_DIR_EXEC=1
  export ROB127_REPO_DIR="$REPO_DIR"
  export ROB127_RUN_ID="$RUN_ID"
  export ROB127_ARTIFACT_ROOT="$ARTIFACT_ROOT"
  export ROB127_RUN_DIR="$RUN_DIR"
  export ROB127_OUT_LOG="$OUT_LOG"
  export ROB127_ERR_LOG="$ERR_LOG"
  export ROB127_SUMMARY_FILE="$SUMMARY_FILE"
  export ROB127_EXECUTED_SCRIPT="$EXECUTED_SCRIPT"
  export ROB127_CALLBACK_SCRIPT="$CALLBACK_SCRIPT"
  exec bash "$EXECUTED_SCRIPT" "$@"
fi

mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR" "$WANDB_DIR" "$TMPDIR"
export TMPDIR
exec > >(tee -a "$OUT_LOG") 2> >(tee -a "$ERR_LOG" >&2)

if [[ "$LATE_WORD_PENALTY_MODE" == "constant" ]]; then
  LATE_WORD_PENALTY_PER_SECOND_SUMMARY="inactive_constant_mode"
else
  LATE_WORD_PENALTY_PER_SECOND_SUMMARY="$LATE_WORD_PENALTY_PER_SECOND"
fi

on_exit() {
  local status=$?
  set +e
  {
    echo "ended_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_code=${status}"
    echo "run_id=${RUN_ID}"
    echo "host=$(hostname)"
    echo "repo_dir=${REPO_DIR}"
    echo "branch=$(cd "$REPO_DIR" && git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    echo "commit=$(cd "$REPO_DIR" && git rev-parse HEAD 2>/dev/null)"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "runtime_config=${RUNTIME_CONFIG}"
    echo "manifest=${MANIFEST_PATH}"
    echo "seed_checkpoint=${SEED_CHECKPOINT}"
    echo "checkpoint_dir=${CHECKPOINT_DIR}"
    echo "wandb_dir=${WANDB_DIR}"
    echo "tmpdir=${TMPDIR}"
    echo "wandb_name=${WANDB_NAME}"
    echo "wandb_id=${WANDB_ID:-new}"
    echo "batch_size=${BATCH_SIZE}"
    echo "max_steps=${MAX_STEPS}"
    echo "save_every=${SAVE_EVERY}"
    echo "learning_rate=${LEARNING_RATE}"
    echo "num_rollouts=${NUM_ROLLOUTS}"
    echo "microbatch_size=${MICROBATCH_SIZE}"
    echo "logprob_microbatch_size=${LOGPROB_MICROBATCH_SIZE}"
    echo "temperature=${TEMPERATURE}"
    echo "max_output_frames=${MAX_OUTPUT_FRAMES:-uncapped}"
    echo "reward_std_min=${REWARD_STD_MIN}"
    echo "sample_text_log_every=${SAMPLE_TEXT_LOG_EVERY}"
    echo "late_word_tolerance_seconds=${LATE_WORD_TOLERANCE_SECONDS}"
    echo "late_word_penalty_mode=${LATE_WORD_PENALTY_MODE}"
    echo "late_word_penalty_per_second=${LATE_WORD_PENALTY_PER_SECOND_SUMMARY}"
    echo "late_word_penalty_max=${LATE_WORD_PENALTY_MAX}"
    echo "use_rotary=true"
    echo "rotary_base_freq=${ROTARY_BASE_FREQ}"
    echo "rotary_interpolation_factor=${ROTARY_INTERPOLATION_FACTOR}"
    echo "max_records=${ROB127_MAX_RECORDS:-unset}"
    echo "train_max_steps_override=${ROB127_TRAIN_MAX_STEPS:-unset}"
  } >> "$SUMMARY_FILE"
  if [[ "${ROB127_ENABLE_CALLBACK:-1}" == "1" ]]; then
    if [[ -z "${LINEAR_API_KEY:-}" && -f "$LINEAR_KEY_FILE" ]]; then
      export LINEAR_API_KEY
      LINEAR_API_KEY="$(cat "$LINEAR_KEY_FILE")"
    fi
    callback_args=(
      "$CALLBACK_SCRIPT"
      --mode mimas
      --issue-id ROB-127
      --state-name Todo
      --job-id "$RUN_ID"
      --job-label "Mimas screen session"
      --exit-code "$status"
      --log-out "$OUT_LOG"
      --log-err "$ERR_LOG"
      --output-path "$CHECKPOINT_DIR"
      --artifact-path "$RUN_DIR"
      --summary-file "$SUMMARY_FILE"
      --title "ROB-127 Mimas RoPE streaming decoder RL GRPO run finished"
      --metadata "Runtime config=$RUNTIME_CONFIG"
      --metadata "Seed checkpoint=$SEED_CHECKPOINT"
      --metadata "W&B id=${WANDB_ID:-new}"
    )
    if [[ "${ROB127_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
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
  echo "branch=$(cd "$REPO_DIR" && git rev-parse --abbrev-ref HEAD)"
  echo "commit=$(cd "$REPO_DIR" && git rev-parse HEAD)"
  echo "executed_script=${EXECUTED_SCRIPT}"
  echo "callback_script=${CALLBACK_SCRIPT}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "tmpdir=${TMPDIR}"
  echo "wandb_id=${WANDB_ID:-new}"
} > "$SUMMARY_FILE"

if [[ "${ROB127_CALLBACK_ONLY:-0}" == "1" ]]; then
  echo "callback-only dry run requested"
  exit 0
fi

cd "$REPO_DIR"
export PYTHONPATH="$PWD"

prepare_args=(
  symphony/scripts/prepare_rob127_mimas_rope_streaming_rl.py
  --base-config "$BASE_CONFIG"
  --source-pairs "$SOURCE_PAIRS"
  --manifest-out "$MANIFEST_PATH"
  --config-out "$RUNTIME_CONFIG"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --seed-checkpoint "$SEED_CHECKPOINT"
  --wandb-dir "$WANDB_DIR"
  --wandb-name "$WANDB_NAME"
  --wandb-id "$WANDB_ID"
  --batch-size "$BATCH_SIZE"
  --max-steps "$MAX_STEPS"
  --save-every "$SAVE_EVERY"
  --learning-rate "$LEARNING_RATE"
  --num-rollouts "$NUM_ROLLOUTS"
  --microbatch-size "$MICROBATCH_SIZE"
  --logprob-microbatch-size "$LOGPROB_MICROBATCH_SIZE"
  --temperature "$TEMPERATURE"
  --reward-std-min "$REWARD_STD_MIN"
  --sample-text-log-every "$SAMPLE_TEXT_LOG_EVERY"
  --late-word-tolerance-seconds "$LATE_WORD_TOLERANCE_SECONDS"
  --late-word-penalty-mode "$LATE_WORD_PENALTY_MODE"
  --late-word-penalty-max "$LATE_WORD_PENALTY_MAX"
  --rotary-base-freq "$ROTARY_BASE_FREQ"
  --rotary-interpolation-factor "$ROTARY_INTERPOLATION_FACTOR"
)

if [[ "$LATE_WORD_PENALTY_MODE" != "constant" ]]; then
  prepare_args+=(--late-word-penalty-per-second "$LATE_WORD_PENALTY_PER_SECOND")
fi

if [[ -n "$MAX_OUTPUT_FRAMES" ]]; then
  prepare_args+=(--max-output-frames "$MAX_OUTPUT_FRAMES")
fi

python "${prepare_args[@]}"

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
if [[ -n "${ROB127_MAX_RECORDS:-}" ]]; then
  train_args+=(-max_records "$ROB127_MAX_RECORDS")
fi
if [[ -n "${ROB127_TRAIN_MAX_STEPS:-}" ]]; then
  train_args+=(-max_steps "$ROB127_TRAIN_MAX_STEPS")
fi
if [[ "${ROB127_DISABLE_WANDB:-0}" == "1" ]]; then
  train_args+=(-disable_wandb)
fi
if [[ "${ROB127_VALIDATE_ONLY:-0}" == "1" ]]; then
  train_args+=(--validate_config_only --validate_load_model)
fi
if [[ "${ROB127_SMOKE_ROLLOUT:-0}" == "1" ]]; then
  train_args+=(
    --smoke_rollout
    --smoke_num_rollouts "${ROB127_SMOKE_NUM_ROLLOUTS:-2}"
    --smoke_max_records "${ROB127_SMOKE_MAX_RECORDS:-2}"
  )
  if [[ -n "${ROB127_SMOKE_MAX_OUTPUT_FRAMES:-}" ]]; then
    train_args+=(--smoke_max_output_frames "$ROB127_SMOKE_MAX_OUTPUT_FRAMES")
  fi
  if [[ "${ROB127_SMOKE_CPU:-0}" == "1" ]]; then
    train_args+=(--smoke_cpu)
  fi
fi

python exp/train_streaming_decoder_asr_rl.py "${train_args[@]}"
