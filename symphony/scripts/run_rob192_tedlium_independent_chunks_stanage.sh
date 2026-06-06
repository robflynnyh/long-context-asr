#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${ROB192_REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-192}"
REMOTE_BRANCH="${ROB192_REMOTE_BRANCH:-symphony/ROB-192-finetune-streaming-decoder-floras}"
TARGET_COMMIT="${ROB192_TARGET_COMMIT:?ROB192_TARGET_COMMIT must pin the code revision}"
ARTIFACT_ROOT="${ROB192_TED_CHUNK_ARTIFACT_ROOT:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/eval}"
RUN_ID="${ROB192_TED_CHUNK_RUN_ID:-rob192-tedlium-independent-chunks-${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}}"
RUN_DIR="${ROB192_TED_CHUNK_RUN_DIR:-$ARTIFACT_ROOT/$RUN_ID}"
OUT_LOG="${ROB192_TED_CHUNK_OUT_LOG:-$RUN_DIR/${RUN_ID}.out}"
ERR_LOG="${ROB192_TED_CHUNK_ERR_LOG:-$RUN_DIR/${RUN_ID}.err}"
SUMMARY_FILE="${ROB192_TED_CHUNK_SUMMARY_FILE:-$RUN_DIR/${RUN_ID}.summary.txt}"
EXECUTED_SCRIPT="${ROB192_TED_CHUNK_EXECUTED_SCRIPT:-$RUN_DIR/$(basename "$0")}"
CALLBACK_SCRIPT="${ROB192_TED_CHUNK_CALLBACK_SCRIPT:-$RUN_DIR/linear_stanage_callback.py}"
CALLBACK_HELPER="${ROB192_TED_CHUNK_CALLBACK_HELPER:-$RUN_DIR/linear_job_callback.py}"
CHECKPOINT="${ROB192_TED_CHUNK_CHECKPOINT:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/checkpoints/streaming_decoder_asr_100m_rope_floras50_lr5e-5_12ep_rob192-rope-streaming-floras50-lr5e-5-12ep-10311112/step_321132.pt}"
TEDLIUM_ROOT="${ROB192_TEDLIUM_ROOT:-/mnt/parscratch/users/acp21rjf/TEDLIUM_release1}"
CHUNK_SIZE="${ROB192_TED_CHUNK_SIZE:-2048}"
CHUNK_OVERLAP="${ROB192_TED_CHUNK_OVERLAP:-0}"
MAX_RECORDINGS="${ROB192_TED_CHUNK_MAX_RECORDINGS:-}"
RECORDING_INDEX="${ROB192_TED_CHUNK_RECORDING_INDEX:-}"
DECODE_MODE="${ROB192_TED_CHUNK_DECODE_MODE:-greedy}"
TEMPERATURE="${ROB192_TED_CHUNK_TEMPERATURE:-1.0}"
MAX_OUTPUT_FRAMES="${ROB192_TED_CHUNK_MAX_OUTPUT_FRAMES:-}"
EVAL_DTYPE="${ROB192_TED_CHUNK_EVAL_DTYPE:-bfloat16}"
USE_KV_CACHE="${ROB192_TED_CHUNK_USE_KV_CACHE:-0}"
MAX_KV_CACHE_LENGTH="${ROB192_TED_CHUNK_MAX_KV_CACHE_LENGTH:-}"
MAX_KV_CACHE_SPECTROGRAM_LENGTH="${ROB192_TED_CHUNK_MAX_KV_CACHE_SPECTROGRAM_LENGTH:-}"
SKIP_GIT_UPDATE="${ROB192_SKIP_GIT_UPDATE:-0}"
LINEAR_KEY_FILE="${ROB192_LINEAR_KEY_FILE:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/.linear_api_key}"
LINEAR_ENV_FILE="${ROB192_LINEAR_ENV_FILE:-}"
CALLBACK_PYTHON="${ROB192_CALLBACK_PYTHON:-python3}"

load_linear_api_key() {
  if [[ -n "${LINEAR_API_KEY:-}" ]]; then
    return 0
  fi
  if [[ -f "$LINEAR_KEY_FILE" ]]; then
    export LINEAR_API_KEY
    LINEAR_API_KEY="$(cat "$LINEAR_KEY_FILE")"
    return 0
  fi

  local env_file
  for env_file in \
    "$LINEAR_ENV_FILE" \
    "$REPO_DIR/symphony/.env" \
    "$HOME/.config/long-context-asr/linear.env" \
    "$HOME/.config/sap-longcontext/linear.env"
  do
    if [[ -n "$env_file" && -f "$env_file" ]]; then
      set +u
      set -a
      # shellcheck disable=SC1090
      source "$env_file"
      set +a
      set -u
      if [[ -n "${LINEAR_API_KEY:-}" ]]; then
        export LINEAR_API_KEY
        return 0
      fi
    fi
  done
}

if [[ "${ROB192_TED_CHUNK_RUN_DIR_EXEC:-0}" != "1" ]]; then
  mkdir -p "$RUN_DIR" "$ARTIFACT_ROOT"
  cp "${BASH_SOURCE[0]}" "$EXECUTED_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_stanage_callback.py" "$CALLBACK_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_job_callback.py" "$CALLBACK_HELPER"
  chmod +x "$EXECUTED_SCRIPT" "$CALLBACK_SCRIPT"
  export ROB192_TED_CHUNK_RUN_DIR_EXEC=1
  export ROB192_REPO_DIR="$REPO_DIR"
  export ROB192_REMOTE_BRANCH="$REMOTE_BRANCH"
  export ROB192_TARGET_COMMIT="$TARGET_COMMIT"
  export ROB192_TED_CHUNK_ARTIFACT_ROOT="$ARTIFACT_ROOT"
  export ROB192_TED_CHUNK_RUN_ID="$RUN_ID"
  export ROB192_TED_CHUNK_RUN_DIR="$RUN_DIR"
  export ROB192_TED_CHUNK_OUT_LOG="$OUT_LOG"
  export ROB192_TED_CHUNK_ERR_LOG="$ERR_LOG"
  export ROB192_TED_CHUNK_SUMMARY_FILE="$SUMMARY_FILE"
  export ROB192_TED_CHUNK_EXECUTED_SCRIPT="$EXECUTED_SCRIPT"
  export ROB192_TED_CHUNK_CALLBACK_SCRIPT="$CALLBACK_SCRIPT"
  export ROB192_TED_CHUNK_CALLBACK_HELPER="$CALLBACK_HELPER"
  exec bash "$EXECUTED_SCRIPT" "$@"
fi

mkdir -p "$RUN_DIR"
exec > >(tee -a "$OUT_LOG") 2> >(tee -a "$ERR_LOG" >&2)

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
    echo "checkpoint=${CHECKPOINT}"
    echo "tedlium_root=${TEDLIUM_ROOT}"
    echo "chunk_size=${CHUNK_SIZE}"
    echo "chunk_overlap=${CHUNK_OVERLAP}"
    echo "decode_mode=${DECODE_MODE}"
    echo "max_output_frames=${MAX_OUTPUT_FRAMES:-unset}"
    echo "use_kv_cache=${USE_KV_CACHE}"
    echo "max_kv_cache_length=${MAX_KV_CACHE_LENGTH:-unset}"
    echo "max_kv_cache_spectrogram_length=${MAX_KV_CACHE_SPECTROGRAM_LENGTH:-unset}"
    echo "max_recordings=${MAX_RECORDINGS:-all}"
    echo "recording_index=${RECORDING_INDEX:-unset}"
    echo "run_dir=${RUN_DIR}"
    echo "result_csv=${RUN_DIR}/rob192_tedlium_independent_chunk_eval.csv"
    echo "slurm_partition=${SLURM_JOB_PARTITION:-unset}"
  } >> "$SUMMARY_FILE"
  if [[ "${ROB192_TED_CHUNK_ENABLE_CALLBACK:-1}" == "1" ]]; then
    load_linear_api_key
    callback_args=(
      "$CALLBACK_SCRIPT"
      --issue-id ROB-192
      --state-name Todo
      --exit-code "$status"
      --log-out "$OUT_LOG"
      --log-err "$ERR_LOG"
      --artifact-path "$RUN_DIR"
      --checkpoint-path "$(dirname "$CHECKPOINT")"
      --summary-file "$SUMMARY_FILE"
      --title "ROB-192 TEDLIUM independent chunk eval finished"
      --job-label "Stanage TEDLIUM chunk diagnostic"
      --metadata "branch=${REMOTE_BRANCH}"
      --metadata "commit=${TARGET_COMMIT}"
      --metadata "checkpoint=${CHECKPOINT}"
      --metadata "chunk_size=${CHUNK_SIZE}"
      --metadata "chunk_overlap=${CHUNK_OVERLAP}"
      --metadata "decode_mode=${DECODE_MODE}"
      --metadata "max_output_frames=${MAX_OUTPUT_FRAMES:-unset}"
      --metadata "use_kv_cache=${USE_KV_CACHE}"
      --metadata "max_kv_cache_length=${MAX_KV_CACHE_LENGTH:-unset}"
      --metadata "max_kv_cache_spectrogram_length=${MAX_KV_CACHE_SPECTROGRAM_LENGTH:-unset}"
      --metadata "result_csv=${RUN_DIR}/rob192_tedlium_independent_chunk_eval.csv"
    )
    if [[ "${ROB192_TED_CHUNK_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
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
  echo "slurm_job_id=${SLURM_JOB_ID:-manual}"
  echo "run_id=${RUN_ID}"
  echo "host=$(hostname)"
  echo "repo_dir=${REPO_DIR}"
  echo "executed_script=${EXECUTED_SCRIPT}"
  echo "callback_script=${CALLBACK_SCRIPT}"
  echo "checkpoint=${CHECKPOINT}"
  echo "tedlium_root=${TEDLIUM_ROOT}"
  echo "chunk_size=${CHUNK_SIZE}"
  echo "chunk_overlap=${CHUNK_OVERLAP}"
  echo "max_output_frames=${MAX_OUTPUT_FRAMES:-unset}"
  echo "use_kv_cache=${USE_KV_CACHE}"
  echo "max_kv_cache_length=${MAX_KV_CACHE_LENGTH:-unset}"
  echo "max_kv_cache_spectrogram_length=${MAX_KV_CACHE_SPECTROGRAM_LENGTH:-unset}"
} > "$SUMMARY_FILE"

if [[ "${ROB192_TED_CHUNK_CALLBACK_ONLY:-0}" == "1" ]]; then
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
export LCASR_TEDLIUM_ROOT="$TEDLIUM_ROOT"

if [[ "${ROB192_TED_CHUNK_FORCE_CPU:-0}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

args=(
  "$REPO_DIR/symphony/scripts/rob192_tedlium_independent_chunk_eval.py"
  --checkpoint "$CHECKPOINT"
  --tedlium-root "$TEDLIUM_ROOT"
  --split test
  --output-dir "$RUN_DIR"
  --chunk-size "$CHUNK_SIZE"
  --chunk-overlap "$CHUNK_OVERLAP"
  --decode-mode "$DECODE_MODE"
  --temperature "$TEMPERATURE"
  --eval-dtype "$EVAL_DTYPE"
)
if [[ -n "$MAX_OUTPUT_FRAMES" ]]; then
  args+=(--max-output-frames "$MAX_OUTPUT_FRAMES")
fi
if [[ -n "$MAX_RECORDINGS" ]]; then
  args+=(--max-recordings "$MAX_RECORDINGS")
fi
if [[ -n "$RECORDING_INDEX" ]]; then
  args+=(--recording-index "$RECORDING_INDEX")
fi
if [[ "$USE_KV_CACHE" == "1" ]]; then
  args+=(--use-kv-cache)
fi
if [[ -n "$MAX_KV_CACHE_LENGTH" ]]; then
  args+=(--max-kv-cache-length "$MAX_KV_CACHE_LENGTH")
fi
if [[ -n "$MAX_KV_CACHE_SPECTROGRAM_LENGTH" ]]; then
  args+=(--max-kv-cache-spectrogram-length "$MAX_KV_CACHE_SPECTROGRAM_LENGTH")
fi

python "${args[@]}"
