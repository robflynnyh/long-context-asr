#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${ROB209_REPO_DIR:-/users/acp21rjf/long-context-asr}"
REMOTE_BRANCH="${ROB209_REMOTE_BRANCH:-symphony/ROB-209-ordered-floras-bounded-history}"
TARGET_COMMIT="${ROB209_TARGET_COMMIT:-}"
if [[ -z "$TARGET_COMMIT" ]]; then
  TARGET_COMMIT="$(git -C "$REPO_DIR" ls-remote origin "refs/heads/${REMOTE_BRANCH}" 2>/dev/null | awk '{print $1}' || true)"
fi
if [[ -z "$TARGET_COMMIT" ]]; then
  TARGET_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
fi
RUN_ID="${ROB209_RUN_ID:-rob209-ordered-floras-bounded-history-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ROB209_ARTIFACT_ROOT:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-209}"
RUN_DIR="${ROB209_RUN_DIR:-$ARTIFACT_ROOT/$RUN_ID}"
LOG_PREFIX="${ROB209_LOG_PREFIX:-${SLURM_JOB_NAME:-rob209-ordered-floras}}"
OUT_LOG="${ROB209_OUT_LOG:-$ARTIFACT_ROOT/${LOG_PREFIX}-${SLURM_JOB_ID:-manual}.out}"
ERR_LOG="${ROB209_ERR_LOG:-$ARTIFACT_ROOT/${LOG_PREFIX}-${SLURM_JOB_ID:-manual}.err}"
SUMMARY_FILE="${ROB209_SUMMARY_FILE:-$RUN_DIR/${RUN_ID}.summary.txt}"
PREP_SUMMARY="${ROB209_PREP_SUMMARY:-$RUN_DIR/${RUN_ID}.prep.txt}"
EXECUTED_SCRIPT="${ROB209_EXECUTED_SCRIPT:-$RUN_DIR/$(basename "$0")}"
CALLBACK_SCRIPT="${ROB209_CALLBACK_SCRIPT:-$RUN_DIR/linear_stanage_callback.py}"
BASE_CONFIG="${ROB209_BASE_CONFIG:-exp/configs/streaming_decoder_asr_100m.yaml}"
SOURCE_PAIRS="${ROB209_SOURCE_PAIRS:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/manifests/floras50_safe_norm_drop_oov.json}"
RUNTIME_CONFIG="${ROB209_RUNTIME_CONFIG:-$RUN_DIR/streaming_decoder_asr_100m_rob209_ordered_floras_bounded_history.yaml}"
CHECKPOINT_DIR="${ROB209_CHECKPOINT_DIR:-$ARTIFACT_ROOT/checkpoints/${RUN_ID}}"
PRETRAINED_CHECKPOINT="${ROB209_PRETRAINED_CHECKPOINT:-/mnt/parscratch/users/acp21rjf/spotify/streaming_decoder_asr_100m_rope_rob123_full_spotify_2epoch_delay0p5_rob123-rope-full-spotify-2epoch-delay0p5-20260523T091306Z/step_272362.pt}"
WANDB_DIR="${ROB209_WANDB_DIR:-$ARTIFACT_ROOT/wandb}"
WANDB_NAME="${ROB209_WANDB_NAME:-rob209_ordered_floras_bounded_history}"
BATCH_SIZE="${ROB209_BATCH_SIZE:-88}"
MAX_EPOCHS="${ROB209_MAX_EPOCHS:-12}"
LEARNING_RATE="${ROB209_LEARNING_RATE:-5e-5}"
SAVE_EVERY_N_STEPS="${ROB209_SAVE_EVERY_N_STEPS:-2000}"
CHUNK_SIZE="${ROB209_CHUNK_SIZE:-2048}"
SUBSAMPLING_HISTORY_FRAMES="${ROB209_SUBSAMPLING_HISTORY_FRAMES:-2048}"
DECODER_HISTORY_FRAMES="${ROB209_DECODER_HISTORY_FRAMES:-2048}"
DELAY_SECONDS="${ROB209_DELAY_SECONDS:-0.5}"
DEBUG_GENERATE_EVERY_RECORDS="${ROB209_DEBUG_GENERATE_EVERY_RECORDS:-500}"
DEBUG_GENERATE_MAX_FRAMES="${ROB209_DEBUG_GENERATE_MAX_FRAMES:-0}"
RANDOM_SEED="${ROB209_RANDOM_SEED:-6433}"
VALIDATE_PATH_LIMIT="${ROB209_VALIDATE_PATH_LIMIT:-20}"
VALIDATE_ALL_PATHS="${ROB209_VALIDATE_ALL_PATHS:-0}"
NO_VALIDATE_PATHS="${ROB209_NO_VALIDATE_PATHS:-0}"
NUM_WORKERS="${ROB209_NUM_WORKERS:-0}"
PREFETCH="${ROB209_PREFETCH:-1}"
PIN_MEMORY="${ROB209_PIN_MEMORY:-0}"
RESET_STEP="${ROB209_RESET_STEP:-0}"
SKIP_GIT_UPDATE="${ROB209_SKIP_GIT_UPDATE:-0}"
CALLBACK_PYTHON="${ROB209_CALLBACK_PYTHON:-python3}"
LINEAR_KEY_FILE="${ROB209_LINEAR_KEY_FILE:-$ARTIFACT_ROOT/.linear_api_key}"

if [[ "${ROB209_RUN_DIR_EXEC:-0}" != "1" ]]; then
  mkdir -p "$RUN_DIR"
  cp "${BASH_SOURCE[0]}" "$EXECUTED_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_stanage_callback.py" "$CALLBACK_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_job_callback.py" "$RUN_DIR/linear_job_callback.py"
  chmod +x "$EXECUTED_SCRIPT" "$CALLBACK_SCRIPT"
  export ROB209_RUN_DIR_EXEC=1
  export ROB209_REPO_DIR="$REPO_DIR"
  export ROB209_REMOTE_BRANCH="$REMOTE_BRANCH"
  export ROB209_TARGET_COMMIT="$TARGET_COMMIT"
  export ROB209_RUN_ID="$RUN_ID"
  export ROB209_ARTIFACT_ROOT="$ARTIFACT_ROOT"
  export ROB209_RUN_DIR="$RUN_DIR"
  export ROB209_OUT_LOG="$OUT_LOG"
  export ROB209_ERR_LOG="$ERR_LOG"
  export ROB209_SUMMARY_FILE="$SUMMARY_FILE"
  export ROB209_PREP_SUMMARY="$PREP_SUMMARY"
  export ROB209_EXECUTED_SCRIPT="$EXECUTED_SCRIPT"
  export ROB209_CALLBACK_SCRIPT="$CALLBACK_SCRIPT"
  exec bash "$EXECUTED_SCRIPT" "$@"
fi

mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR" "$WANDB_DIR"

on_exit() {
  local status=$?
  set +e
  {
    echo "ended_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_code=${status}"
    echo "slurm_job_id=${SLURM_JOB_ID:-manual}"
    echo "run_id=${RUN_ID}"
    echo "host=$(hostname)"
    echo "repo_dir=${REPO_DIR}"
    echo "remote_branch=${REMOTE_BRANCH}"
    echo "target_commit=${TARGET_COMMIT}"
    echo "runtime_config=${RUNTIME_CONFIG}"
    echo "source_pairs=${SOURCE_PAIRS}"
    echo "checkpoint_dir=${CHECKPOINT_DIR}"
    echo "pretrained_checkpoint=${PRETRAINED_CHECKPOINT}"
    echo "wandb_dir=${WANDB_DIR}"
    echo "wandb_name=${WANDB_NAME}"
    echo "batch_size=${BATCH_SIZE}"
    echo "max_epochs=${MAX_EPOCHS}"
    echo "learning_rate=${LEARNING_RATE}"
    echo "save_every_n_steps=${SAVE_EVERY_N_STEPS}"
    echo "chunk_size=${CHUNK_SIZE}"
    echo "subsampling_history_frames=${SUBSAMPLING_HISTORY_FRAMES}"
    echo "decoder_history_frames=${DECODER_HISTORY_FRAMES}"
    echo "delay_seconds=${DELAY_SECONDS}"
    echo "reset_step=${RESET_STEP}"
    echo "skip_git_update=${SKIP_GIT_UPDATE}"
    echo "slurm_partition=${SLURM_JOB_PARTITION:-unset}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  } >> "$SUMMARY_FILE"
  if [[ "${ROB209_ENABLE_CALLBACK:-1}" == "1" ]]; then
    if [[ -z "${LINEAR_API_KEY:-}" && -f "$LINEAR_KEY_FILE" ]]; then
      export LINEAR_API_KEY
      LINEAR_API_KEY="$(cat "$LINEAR_KEY_FILE")"
    fi
    callback_args=(
      "$CALLBACK_SCRIPT"
      --issue-id ROB-209
      --state-name Todo
      --exit-code "$status"
      --log-out "$OUT_LOG"
      --log-err "$ERR_LOG"
      --artifact-path "$RUN_DIR"
      --checkpoint-path "$CHECKPOINT_DIR"
      --summary-file "$SUMMARY_FILE"
      --title "ROB-209 Stanage ordered Floras bounded-history training finished"
      --job-label "Stanage ordered Floras training"
      --metadata "branch=${REMOTE_BRANCH}"
      --metadata "commit=${TARGET_COMMIT}"
      --metadata "runtime_config=${RUNTIME_CONFIG}"
      --metadata "source_pairs=${SOURCE_PAIRS}"
      --metadata "pretrained_checkpoint=${PRETRAINED_CHECKPOINT}"
      --metadata "wandb_name=${WANDB_NAME}"
      --metadata "chunk_size=${CHUNK_SIZE}"
      --metadata "subsampling_history_frames=${SUBSAMPLING_HISTORY_FRAMES}"
      --metadata "decoder_history_frames=${DECODER_HISTORY_FRAMES}"
      --metadata "delay_seconds=${DELAY_SECONDS}"
    )
    if [[ "${ROB209_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
      callback_args+=(--dry-run)
    fi
    "$CALLBACK_PYTHON" "${callback_args[@]}" >> "$SUMMARY_FILE" 2>&1
  fi
  return "$status"
}
trap on_exit EXIT
trap 'trap - TERM INT; exit 143' TERM
trap 'trap - TERM INT; exit 130' INT

{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "slurm_job_id=${SLURM_JOB_ID:-manual}"
  echo "run_id=${RUN_ID}"
  echo "host=$(hostname)"
  echo "executed_script=${EXECUTED_SCRIPT}"
  echo "callback_script=${CALLBACK_SCRIPT}"
} > "$SUMMARY_FILE"

if [[ "${ROB209_CALLBACK_ONLY:-0}" == "1" ]]; then
  echo "callback-only dry run requested"
  exit 0
fi

cd "$REPO_DIR"
if [[ "$SKIP_GIT_UPDATE" == "1" ]]; then
  current_commit="$(git rev-parse HEAD)"
  if [[ "$current_commit" != "$TARGET_COMMIT" ]]; then
    echo "ROB209_SKIP_GIT_UPDATE=1 but repo is at $current_commit, expected $TARGET_COMMIT" >&2
    exit 2
  fi
else
  git fetch origin "$REMOTE_BRANCH"
  git checkout --detach "$TARGET_COMMIT"
fi
export PYTHONPATH="$PWD"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

prep_args=(
  --base-config "$BASE_CONFIG"
  --source-pairs "$SOURCE_PAIRS"
  --config-out "$RUNTIME_CONFIG"
  --summary-out "$PREP_SUMMARY"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --pretrained-checkpoint "$PRETRAINED_CHECKPOINT"
  --wandb-dir "$WANDB_DIR"
  --wandb-name "$WANDB_NAME"
  --batch-size "$BATCH_SIZE"
  --max-epochs "$MAX_EPOCHS"
  --learning-rate "$LEARNING_RATE"
  --save-every-n-steps "$SAVE_EVERY_N_STEPS"
  --chunk-size "$CHUNK_SIZE"
  --subsampling-history-frames "$SUBSAMPLING_HISTORY_FRAMES"
  --decoder-history-frames "$DECODER_HISTORY_FRAMES"
  --delay-seconds "$DELAY_SECONDS"
  --debug-generate-every-records "$DEBUG_GENERATE_EVERY_RECORDS"
  --debug-generate-max-frames "$DEBUG_GENERATE_MAX_FRAMES"
  --random-seed "$RANDOM_SEED"
  --validate-path-limit "$VALIDATE_PATH_LIMIT"
)
if [[ "$VALIDATE_ALL_PATHS" == "1" ]]; then
  prep_args+=(--validate-all-paths)
fi
if [[ "$NO_VALIDATE_PATHS" == "1" ]]; then
  prep_args+=(--no-validate-paths)
fi
python symphony/scripts/prepare_rob209_ordered_floras.py "${prep_args[@]}"

cat "$PREP_SUMMARY" >> "$SUMMARY_FILE"

train_args=(
  -config "$RUNTIME_CONFIG"
  -num_workers "$NUM_WORKERS"
  -prefetch "$PREFETCH"
)
if [[ "$RESET_STEP" == "1" ]]; then
  train_args+=(-reset_step)
fi
if [[ "$PIN_MEMORY" == "1" ]]; then
  train_args+=(-pin_memory)
fi
if [[ -n "${ROB209_MAX_RECORDS:-}" ]]; then
  train_args+=(-max_records "$ROB209_MAX_RECORDS")
fi
if [[ -n "${ROB209_MAX_STEPS:-}" ]]; then
  train_args+=(-max_steps "$ROB209_MAX_STEPS")
fi
if [[ "${ROB209_DISABLE_WANDB:-0}" == "1" ]]; then
  train_args+=(-disable_wandb)
fi

python exp/train_streaming_decoder_asr.py "${train_args[@]}"
