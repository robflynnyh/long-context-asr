#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${ROB192_REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-192}"
REMOTE_BRANCH="${ROB192_REMOTE_BRANCH:-symphony/ROB-192-finetune-streaming-decoder-floras}"
TARGET_COMMIT="${ROB192_TARGET_COMMIT:?ROB192_TARGET_COMMIT must pin the code revision}"
ARTIFACT_ROOT="${ROB192_EVAL_ARTIFACT_ROOT:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/eval}"
RUN_ID="${ROB192_EVAL_RUN_ID:-rob192-final-streaming-eval-${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}}"
RUN_DIR="${ROB192_EVAL_RUN_DIR:-$ARTIFACT_ROOT/$RUN_ID}"
OUT_LOG="${ROB192_EVAL_OUT_LOG:-$RUN_DIR/${RUN_ID}.out}"
ERR_LOG="${ROB192_EVAL_ERR_LOG:-$RUN_DIR/${RUN_ID}.err}"
SUMMARY_FILE="${ROB192_EVAL_SUMMARY_FILE:-$RUN_DIR/${RUN_ID}.summary.txt}"
EVAL_CONFIG="${ROB192_EVAL_CONFIG:-$RUN_DIR/rob192_streaming_decoder_eval.yaml}"
RESULT_CSV="${ROB192_EVAL_RESULT_CSV:-$RUN_DIR/rob192_streaming_decoder_eval.csv}"
EXECUTED_SCRIPT="${ROB192_EVAL_EXECUTED_SCRIPT:-$RUN_DIR/$(basename "$0")}"
CALLBACK_SCRIPT="${ROB192_EVAL_CALLBACK_SCRIPT:-$RUN_DIR/linear_stanage_callback.py}"
CALLBACK_HELPER="${ROB192_EVAL_CALLBACK_HELPER:-$RUN_DIR/linear_job_callback.py}"
CHECKPOINT="${ROB192_EVAL_CHECKPOINT:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/checkpoints/streaming_decoder_asr_100m_rope_floras50_lr5e-5_12ep_rob192-rope-streaming-floras50-lr5e-5-12ep-10311112/step_321132.pt}"
TEDLIUM_ROOT="${ROB192_TEDLIUM_ROOT:-/mnt/parscratch/users/acp21rjf/TEDLIUM_release1}"
DECODE_MODE="${ROB192_EVAL_DECODE_MODE:-greedy}"
TEMPERATURE="${ROB192_EVAL_TEMPERATURE:-1.0}"
MAX_KV_CACHE_SPECTROGRAM_LENGTH="${ROB192_MAX_KV_CACHE_SPECTROGRAM_LENGTH:-2048}"
BREAK_EVAL="${ROB192_EVAL_BREAK:-0}"
INCLUDE_PER_RECORDING="${ROB192_INCLUDE_PER_RECORDING:-1}"
OMP_THREADS="${ROB192_OMP_THREADS:-16}"
SKIP_GIT_UPDATE="${ROB192_SKIP_GIT_UPDATE:-0}"
CALLBACK_PYTHON="${ROB192_CALLBACK_PYTHON:-python3}"
LINEAR_KEY_FILE="${ROB192_LINEAR_KEY_FILE:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/.linear_api_key}"
LINEAR_ENV_FILE="${ROB192_LINEAR_ENV_FILE:-}"

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

if [[ "${ROB192_EVAL_RUN_DIR_EXEC:-0}" != "1" ]]; then
  mkdir -p "$RUN_DIR" "$ARTIFACT_ROOT"
  cp "${BASH_SOURCE[0]}" "$EXECUTED_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_stanage_callback.py" "$CALLBACK_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_job_callback.py" "$CALLBACK_HELPER"
  chmod +x "$EXECUTED_SCRIPT" "$CALLBACK_SCRIPT"
  export ROB192_EVAL_RUN_DIR_EXEC=1
  export ROB192_REPO_DIR="$REPO_DIR"
  export ROB192_REMOTE_BRANCH="$REMOTE_BRANCH"
  export ROB192_TARGET_COMMIT="$TARGET_COMMIT"
  export ROB192_EVAL_ARTIFACT_ROOT="$ARTIFACT_ROOT"
  export ROB192_EVAL_RUN_ID="$RUN_ID"
  export ROB192_EVAL_RUN_DIR="$RUN_DIR"
  export ROB192_EVAL_OUT_LOG="$OUT_LOG"
  export ROB192_EVAL_ERR_LOG="$ERR_LOG"
  export ROB192_EVAL_SUMMARY_FILE="$SUMMARY_FILE"
  export ROB192_EVAL_CONFIG="$EVAL_CONFIG"
  export ROB192_EVAL_RESULT_CSV="$RESULT_CSV"
  export ROB192_EVAL_EXECUTED_SCRIPT="$EXECUTED_SCRIPT"
  export ROB192_EVAL_CALLBACK_SCRIPT="$CALLBACK_SCRIPT"
  export ROB192_EVAL_CALLBACK_HELPER="$CALLBACK_HELPER"
  exec bash "$EXECUTED_SCRIPT" "$@"
fi

mkdir -p "$RUN_DIR" "$(dirname "$RESULT_CSV")"
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
    echo "eval_config=${EVAL_CONFIG}"
    echo "result_csv=${RESULT_CSV}"
    echo "decode_mode=${DECODE_MODE}"
    echo "temperature=${TEMPERATURE}"
    echo "use_kv_cache=true"
    echo "max_kv_cache_spectrogram_length=${MAX_KV_CACHE_SPECTROGRAM_LENGTH}"
    echo "break_eval=${BREAK_EVAL}"
    echo "slurm_partition=${SLURM_JOB_PARTITION:-unset}"
  } >> "$SUMMARY_FILE"
  if [[ "${ROB192_EVAL_ENABLE_CALLBACK:-1}" == "1" ]]; then
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
      --title "ROB-192 TEDLIUM and Earnings CPU eval finished"
      --job-label "Stanage CPU eval"
      --metadata "branch=${REMOTE_BRANCH}"
      --metadata "commit=${TARGET_COMMIT}"
      --metadata "checkpoint=${CHECKPOINT}"
      --metadata "result_csv=${RESULT_CSV}"
      --metadata "decode_mode=${DECODE_MODE}"
      --metadata "use_kv_cache=true"
      --metadata "max_kv_cache_spectrogram_length=${MAX_KV_CACHE_SPECTROGRAM_LENGTH}"
      --metadata "break_eval=${BREAK_EVAL}"
    )
    if [[ "${ROB192_EVAL_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
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
} > "$SUMMARY_FILE"

if [[ "${ROB192_EVAL_CALLBACK_ONLY:-0}" == "1" ]]; then
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
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="$OMP_THREADS"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

python - "$EVAL_CONFIG" "$RESULT_CSV" "$CHECKPOINT" "$DECODE_MODE" "$TEMPERATURE" "$MAX_KV_CACHE_SPECTROGRAM_LENGTH" "$BREAK_EVAL" "$INCLUDE_PER_RECORDING" <<'PY'
import sys
from pathlib import Path

config_path, result_csv, checkpoint, decode_mode, temperature, cache_spec_len, break_eval, include_per_recording = sys.argv[1:9]
if not Path(checkpoint).exists():
    raise SystemExit(f"missing checkpoint: {checkpoint}")

config = f"""models:
  - name: rob192_floras_step_321132_kvcache_greedy
    path: "{checkpoint}"
    seq_len: -1
    overlap_ratio: 0
    repeat: 1
    checkpoint_step: 321132
datasets:
  - name: tedlium
    splits: ["test"]
  - name: earnings22
    splits: ["test"]
args:
  model_class: StreamingDecoderASR
  run_eval_with: run
  save_dataframe_path: "{result_csv}"
  verbose: false
  include_per_recording_evaluations: {include_per_recording}
  break_eval: {break_eval}
  transcribe_kwargs:
    decode_mode: "{decode_mode}"
    temperature: {temperature}
    use_kv_cache: true
    max_kv_cache_spectrogram_length: {cache_spec_len}
"""
Path(config_path).write_text(config, encoding="utf-8")
PY

{
  echo "checkpoint_exists=$(test -f "$CHECKPOINT" && echo yes || echo no)"
  echo "tedlium_root_exists=$(test -d "$TEDLIUM_ROOT" && echo yes || echo no)"
  echo "eval_config=${EVAL_CONFIG}"
  echo "result_csv=${RESULT_CSV}"
  echo "omp_threads=${OMP_THREADS}"
} >> "$SUMMARY_FILE"

cd "$REPO_DIR/eval"
python eval_manager.py -config "$EVAL_CONFIG"

python - "$RESULT_CSV" "$SUMMARY_FILE" <<'PY'
import csv
import sys

csv_path, summary_path = sys.argv[1:3]
rows = list(csv.DictReader(open(csv_path, newline="")))
aggregates = [
    row for row in rows
    if row.get("recording") == "all" and row.get("split") == "test"
]
with open(summary_path, "a", encoding="utf-8") as handle:
    handle.write(f"rows={len(rows)}\n")
    if not aggregates:
        handle.write("eval_aggregate=missing\n")
    for row in aggregates:
        handle.write(
            "eval_aggregate="
            f"dataset={row.get('dataset')} "
            f"name={row.get('name')} "
            f"step={row.get('checkpoint_step')} "
            f"wer={row.get('wer')} "
            f"words={row.get('words')} "
            f"ins_rate={row.get('ins_rate')} "
            f"del_rate={row.get('del_rate')} "
            f"sub_rate={row.get('sub_rate')}\n"
        )
PY
