#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${ROB192_REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-192}"
REMOTE_BRANCH="${ROB192_REMOTE_BRANCH:-symphony/ROB-192-finetune-streaming-decoder-floras}"
TARGET_COMMIT="${ROB192_TARGET_COMMIT:?ROB192_TARGET_COMMIT must pin the code revision}"
RUN_ID="${ROB192_RUN_ID:-rob192-rope-streaming-floras50-lr5e-5-12ep-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ROB192_ARTIFACT_ROOT:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192}"
RUN_DIR="${ROB192_RUN_DIR:-$ARTIFACT_ROOT/$RUN_ID}"
OUT_LOG="${ROB192_OUT_LOG:-$ARTIFACT_ROOT/rob192-floras-${SLURM_JOB_ID:-manual}.out}"
ERR_LOG="${ROB192_ERR_LOG:-$ARTIFACT_ROOT/rob192-floras-${SLURM_JOB_ID:-manual}.err}"
SUMMARY_FILE="${ROB192_SUMMARY_FILE:-$RUN_DIR/${RUN_ID}.summary.txt}"
PREP_SUMMARY="${ROB192_PREP_SUMMARY:-$RUN_DIR/${RUN_ID}.prep.txt}"
EXECUTED_SCRIPT="${ROB192_EXECUTED_SCRIPT:-$RUN_DIR/$(basename "$0")}"
CALLBACK_SCRIPT="${ROB192_CALLBACK_SCRIPT:-$RUN_DIR/linear_stanage_callback.py}"
BASE_CONFIG="${ROB192_BASE_CONFIG:-exp/configs/streaming_decoder_asr_100m.yaml}"
FLORAS_MAPPING="${ROB192_FLORAS_MAPPING:-/users/acp21rjf/align_floras50/tmp/mapping.json}"
TOKENIZER="${ROB192_TOKENIZER:-lcasr/artifacts/tokenizer.model}"
MANIFEST_DIR="${ROB192_MANIFEST_DIR:-$RUN_DIR/manifests}"
MANIFEST_OUT="${ROB192_MANIFEST_OUT:-$MANIFEST_DIR/floras50_safe_norm_drop_oov.json}"
NORMALIZED_TXT_DIR="${ROB192_NORMALIZED_TXT_DIR:-$MANIFEST_DIR/normalized_txt}"
MANIFEST_SUMMARY_JSON="${ROB192_MANIFEST_SUMMARY_JSON:-$MANIFEST_DIR/floras50_safe_norm_drop_oov_summary.json}"
RUNTIME_CONFIG="${ROB192_RUNTIME_CONFIG:-$RUN_DIR/streaming_decoder_asr_100m_rope_floras50_lr5e-5_12ep.yaml}"
CHECKPOINT_DIR="${ROB192_CHECKPOINT_DIR:-$ARTIFACT_ROOT/checkpoints/streaming_decoder_asr_100m_rope_floras50_lr5e-5_12ep_${RUN_ID}}"
PRETRAINED_CHECKPOINT="${ROB192_PRETRAINED_CHECKPOINT:-/mnt/parscratch/users/acp21rjf/spotify/streaming_decoder_asr_100m_rope_rob123_full_spotify_2epoch_delay0p5_rob123-rope-full-spotify-2epoch-delay0p5-20260523T091306Z/step_272362.pt}"
WANDB_DIR="${ROB192_WANDB_DIR:-$ARTIFACT_ROOT/wandb}"
WANDB_PROJECT_NAME="${ROB192_WANDB_PROJECT_NAME:-floras50_streaming_decoder_supervised}"
WANDB_NAME="${ROB192_WANDB_NAME:-rob192_rope_streaming_decoder_floras50_lr5e-5_12ep}"
BATCH_SIZE="${ROB192_BATCH_SIZE:-88}"
MAX_EPOCHS="${ROB192_MAX_EPOCHS:-12}"
LEARNING_RATE="${ROB192_LEARNING_RATE:-5e-5}"
SAVE_EVERY_N_STEPS="${ROB192_SAVE_EVERY_N_STEPS:-50000}"
SUBSAMPLING_FACTOR="${ROB192_SUBSAMPLING_FACTOR:-8}"
ROTARY_BASE_FREQ="${ROB192_ROTARY_BASE_FREQ:-1500000}"
DELAY_SECONDS="${ROB192_DELAY_SECONDS:-0.5}"
DEBUG_GENERATE_EVERY_RECORDS="${ROB192_DEBUG_GENERATE_EVERY_RECORDS:-500}"
DEBUG_GENERATE_MAX_FRAMES="${ROB192_DEBUG_GENERATE_MAX_FRAMES:-0}"
VALIDATE_PATH_LIMIT="${ROB192_VALIDATE_PATH_LIMIT:-50}"
VALIDATE_ALL_PATHS="${ROB192_VALIDATE_ALL_PATHS:-0}"
MANIFEST_LIMIT="${ROB192_MANIFEST_LIMIT:-0}"
NUM_WORKERS="${ROB192_NUM_WORKERS:-0}"
PREFETCH="${ROB192_PREFETCH:-1}"
PIN_MEMORY="${ROB192_PIN_MEMORY:-0}"
SKIP_GIT_UPDATE="${ROB192_SKIP_GIT_UPDATE:-0}"
CALLBACK_PYTHON="${ROB192_CALLBACK_PYTHON:-python3}"
LINEAR_KEY_FILE="${ROB192_LINEAR_KEY_FILE:-$ARTIFACT_ROOT/.linear_api_key}"

if [[ "${ROB192_RUN_DIR_EXEC:-0}" != "1" ]]; then
  mkdir -p "$RUN_DIR"
  cp "${BASH_SOURCE[0]}" "$EXECUTED_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_stanage_callback.py" "$CALLBACK_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_job_callback.py" "$RUN_DIR/linear_job_callback.py"
  chmod +x "$EXECUTED_SCRIPT" "$CALLBACK_SCRIPT"
  export ROB192_RUN_DIR_EXEC=1
  export ROB192_REPO_DIR="$REPO_DIR"
  export ROB192_REMOTE_BRANCH="$REMOTE_BRANCH"
  export ROB192_TARGET_COMMIT="$TARGET_COMMIT"
  export ROB192_RUN_ID="$RUN_ID"
  export ROB192_ARTIFACT_ROOT="$ARTIFACT_ROOT"
  export ROB192_RUN_DIR="$RUN_DIR"
  export ROB192_OUT_LOG="$OUT_LOG"
  export ROB192_ERR_LOG="$ERR_LOG"
  export ROB192_SUMMARY_FILE="$SUMMARY_FILE"
  export ROB192_PREP_SUMMARY="$PREP_SUMMARY"
  export ROB192_EXECUTED_SCRIPT="$EXECUTED_SCRIPT"
  export ROB192_CALLBACK_SCRIPT="$CALLBACK_SCRIPT"
  exec bash "$EXECUTED_SCRIPT" "$@"
fi

mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR" "$WANDB_DIR" "$MANIFEST_DIR" "$NORMALIZED_TXT_DIR"

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
    echo "floras_mapping=${FLORAS_MAPPING}"
    echo "manifest=${MANIFEST_OUT}"
    echo "manifest_summary_json=${MANIFEST_SUMMARY_JSON}"
    echo "checkpoint_dir=${CHECKPOINT_DIR}"
    echo "pretrained_checkpoint=${PRETRAINED_CHECKPOINT}"
    echo "wandb_dir=${WANDB_DIR}"
    echo "wandb_project_name=${WANDB_PROJECT_NAME}"
    echo "wandb_name=${WANDB_NAME}"
    echo "batch_size=${BATCH_SIZE}"
    echo "max_epochs=${MAX_EPOCHS}"
    echo "learning_rate=${LEARNING_RATE}"
    echo "save_every_n_steps=${SAVE_EVERY_N_STEPS}"
    echo "subsampling_factor=${SUBSAMPLING_FACTOR}"
    echo "use_rotary=true"
    echo "rotary_base_freq=${ROTARY_BASE_FREQ}"
    echo "delay_seconds=${DELAY_SECONDS}"
    echo "skip_git_update=${SKIP_GIT_UPDATE}"
    echo "slurm_partition=${SLURM_JOB_PARTITION:-unset}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  } >> "$SUMMARY_FILE"
  if [[ "${ROB192_ENABLE_CALLBACK:-1}" == "1" ]]; then
    if [[ -z "${LINEAR_API_KEY:-}" && -f "$LINEAR_KEY_FILE" ]]; then
      export LINEAR_API_KEY
      LINEAR_API_KEY="$(cat "$LINEAR_KEY_FILE")"
    fi
    callback_args=(
      "$CALLBACK_SCRIPT"
      --issue-id ROB-192
      --state-name Todo
      --exit-code "$status"
      --log-out "$OUT_LOG"
      --log-err "$ERR_LOG"
      --artifact-path "$RUN_DIR"
      --checkpoint-path "$CHECKPOINT_DIR"
      --summary-file "$SUMMARY_FILE"
      --title "ROB-192 Stanage Floras streaming-decoder finetuning finished"
      --job-label "Stanage training"
      --metadata "branch=${REMOTE_BRANCH}"
      --metadata "commit=${TARGET_COMMIT}"
      --metadata "runtime_config=${RUNTIME_CONFIG}"
      --metadata "floras_mapping=${FLORAS_MAPPING}"
      --metadata "manifest=${MANIFEST_OUT}"
      --metadata "pretrained_checkpoint=${PRETRAINED_CHECKPOINT}"
      --metadata "wandb_name=${WANDB_NAME}"
      --metadata "learning_rate=${LEARNING_RATE}"
      --metadata "max_epochs=${MAX_EPOCHS}"
    )
    if [[ "${ROB192_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
      callback_args+=(--dry-run)
    fi
    "$CALLBACK_PYTHON" "${callback_args[@]}" >> "$SUMMARY_FILE" 2>&1
  fi
  return "$status"
}
trap on_exit EXIT
trap 'trap - TERM INT; exit 143' TERM
trap 'trap - INT; exit 130' INT

{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "slurm_job_id=${SLURM_JOB_ID:-manual}"
  echo "run_id=${RUN_ID}"
  echo "host=$(hostname)"
  echo "executed_script=${EXECUTED_SCRIPT}"
  echo "callback_script=${CALLBACK_SCRIPT}"
} > "$SUMMARY_FILE"

if [[ "${ROB192_CALLBACK_ONLY:-0}" == "1" ]]; then
  echo "callback-only dry run requested"
  exit 0
fi

cd "$REPO_DIR"
if [[ "$SKIP_GIT_UPDATE" == "1" ]]; then
  current_commit="$(git rev-parse HEAD)"
  if [[ "$current_commit" != "$TARGET_COMMIT" ]]; then
    echo "ROB192_SKIP_GIT_UPDATE=1 but repo is at $current_commit, expected $TARGET_COMMIT" >&2
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
  --mapping "$FLORAS_MAPPING"
  --tokenizer "$TOKENIZER"
  --manifest-out "$MANIFEST_OUT"
  --text-output-dir "$NORMALIZED_TXT_DIR"
  --manifest-summary-json "$MANIFEST_SUMMARY_JSON"
  --config-out "$RUNTIME_CONFIG"
  --summary-out "$PREP_SUMMARY"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --pretrained-checkpoint "$PRETRAINED_CHECKPOINT"
  --wandb-dir "$WANDB_DIR"
  --wandb-project-name "$WANDB_PROJECT_NAME"
  --wandb-name "$WANDB_NAME"
  --batch-size "$BATCH_SIZE"
  --max-epochs "$MAX_EPOCHS"
  --learning-rate "$LEARNING_RATE"
  --save-every-n-steps "$SAVE_EVERY_N_STEPS"
  --subsampling-factor "$SUBSAMPLING_FACTOR"
  --rotary-base-freq "$ROTARY_BASE_FREQ"
  --delay-seconds "$DELAY_SECONDS"
  --debug-generate-every-records "$DEBUG_GENERATE_EVERY_RECORDS"
  --debug-generate-max-frames "$DEBUG_GENERATE_MAX_FRAMES"
  --validate-path-limit "$VALIDATE_PATH_LIMIT"
  --manifest-limit "$MANIFEST_LIMIT"
)
if [[ "$VALIDATE_ALL_PATHS" == "1" ]]; then
  prep_args+=(--validate-all-paths)
fi
python symphony/scripts/prepare_rob192_stanage_streaming_floras.py "${prep_args[@]}"

cat "$PREP_SUMMARY" >> "$SUMMARY_FILE"

train_args=(
  -config "$RUNTIME_CONFIG"
  -num_workers "$NUM_WORKERS"
  -prefetch "$PREFETCH"
)
if [[ "$PIN_MEMORY" == "1" ]]; then
  train_args+=(-pin_memory)
fi
if [[ -n "${ROB192_MAX_RECORDS:-}" ]]; then
  train_args+=(-max_records "$ROB192_MAX_RECORDS")
fi
if [[ -n "${ROB192_MAX_STEPS:-}" ]]; then
  train_args+=(-max_steps "$ROB192_MAX_STEPS")
fi
if [[ "${ROB192_DISABLE_WANDB:-0}" == "1" ]]; then
  train_args+=(-disable_wandb)
fi

python exp/train_streaming_decoder_asr.py "${train_args[@]}"
