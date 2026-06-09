#!/bin/bash
#SBATCH --job-name=rob209-online-eval
#SBATCH --time=02:00:00
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-209/rob209-online-eval-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-209/rob209-online-eval-%j.err

set -Eeuo pipefail

REPO_DIR="${ROB209_REPO_DIR:-/users/acp21rjf/long-context-asr}"
REMOTE_BRANCH="${ROB209_REMOTE_BRANCH:-symphony/ROB-209-ordered-floras-bounded-history}"
TARGET_COMMIT="${ROB209_TARGET_COMMIT:-}"
if [[ -z "$TARGET_COMMIT" ]]; then
  TARGET_COMMIT="$(git -C "$REPO_DIR" ls-remote origin "refs/heads/${REMOTE_BRANCH}" 2>/dev/null | awk '{print $1}' || true)"
fi
if [[ -z "$TARGET_COMMIT" ]]; then
  TARGET_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
fi

RUN_ID="${ROB209_EVAL_RUN_ID:-rob209-online-streaming-eval-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ROB209_ARTIFACT_ROOT:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-209}"
RUN_DIR="${ROB209_EVAL_RUN_DIR:-$ARTIFACT_ROOT/$RUN_ID}"
LOG_PREFIX="${ROB209_EVAL_LOG_PREFIX:-${SLURM_JOB_NAME:-rob209-online-eval}}"
OUT_LOG="${ROB209_EVAL_OUT_LOG:-$ARTIFACT_ROOT/${LOG_PREFIX}-${SLURM_JOB_ID:-manual}.out}"
ERR_LOG="${ROB209_EVAL_ERR_LOG:-$ARTIFACT_ROOT/${LOG_PREFIX}-${SLURM_JOB_ID:-manual}.err}"
SUMMARY_FILE="${ROB209_EVAL_SUMMARY_FILE:-$RUN_DIR/${RUN_ID}.summary.txt}"
EVAL_CONFIG="${ROB209_EVAL_CONFIG:-$RUN_DIR/rob209_online_streaming_eval.yaml}"
RESULT_CSV="${ROB209_EVAL_RESULT_CSV:-$RUN_DIR/rob209_online_streaming_eval.csv}"
EXECUTED_SCRIPT="${ROB209_EVAL_EXECUTED_SCRIPT:-$RUN_DIR/$(basename "$0")}"
CALLBACK_SCRIPT="${ROB209_EVAL_CALLBACK_SCRIPT:-$RUN_DIR/linear_stanage_callback.py}"
CALLBACK_HELPER="${ROB209_EVAL_CALLBACK_HELPER:-$RUN_DIR/linear_job_callback.py}"
CALLBACK_PYTHON="${ROB209_CALLBACK_PYTHON:-python3}"
LINEAR_KEY_FILE="${ROB209_LINEAR_KEY_FILE:-$ARTIFACT_ROOT/.linear_api_key}"
LINEAR_ENV_FILE="${ROB209_LINEAR_ENV_FILE:-}"

CHECKPOINT="${ROB209_EVAL_CHECKPOINT:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-209/checkpoints/rob209-ordered-floras-debuggen-20260608T2215Z/step_36425.pt}"
CHECKPOINT_STEP="${ROB209_EVAL_CHECKPOINT_STEP:-}"
DATASET="${ROB209_EVAL_DATASET:-floras50}"
SPLIT="${ROB209_EVAL_SPLIT:-test}"
FLORAS50_ROOT="${ROB209_FLORAS50_ROOT:-/mnt/parscratch/users/acp21rjf/floras50_eval}"
MODEL_NAME="${ROB209_EVAL_MODEL_NAME:-rob209_step_${CHECKPOINT_STEP:-unknown}_online_chunked_kv}"
MAX_SEQUENCE_LENGTH="${ROB209_EVAL_MAX_SEQUENCE_LENGTH:--1}"
MAX_OUTPUT_FRAMES="${ROB209_EVAL_MAX_OUTPUT_FRAMES:-0}"
BREAK_EVAL="${ROB209_EVAL_BREAK:-1}"
INCLUDE_PER_RECORDING="${ROB209_INCLUDE_PER_RECORDING:-1}"
VERBOSE="${ROB209_EVAL_VERBOSE:-0}"
DECODE_MODE="${ROB209_EVAL_DECODE_MODE:-greedy}"
TEMPERATURE="${ROB209_EVAL_TEMPERATURE:-1.0}"
CHUNK_SIZE="${ROB209_EVAL_CHUNK_SIZE:-2048}"
SUBSAMPLING_HISTORY_FRAMES="${ROB209_EVAL_SUBSAMPLING_HISTORY_FRAMES:-2048}"
DECODER_HISTORY_FRAMES="${ROB209_EVAL_DECODER_HISTORY_FRAMES:-2048}"
FINAL_FLUSH_FRAMES="${ROB209_EVAL_FINAL_FLUSH_FRAMES:-75}"
OMP_THREADS="${ROB209_EVAL_OMP_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

bool_yaml() {
  case "$1" in
    1|true|TRUE|yes|YES|on|ON) printf 'true' ;;
    *) printf 'false' ;;
  esac
}

load_linear_api_key() {
  if [[ -n "${LINEAR_API_KEY:-}" ]]; then
    return 0
  fi
  if [[ -f "$LINEAR_KEY_FILE" ]]; then
    export LINEAR_API_KEY
    LINEAR_API_KEY="$(cat "$LINEAR_KEY_FILE")"
    echo "linear_key_source=${LINEAR_KEY_FILE}" >> "$SUMMARY_FILE"
    return 0
  fi

  local env_file
  local env_candidates=()
  if [[ -n "$LINEAR_ENV_FILE" ]]; then
    env_candidates+=("$LINEAR_ENV_FILE")
  fi
  env_candidates+=(
    "$RUN_DIR/linear.env"
    "$REPO_DIR/symphony/.env"
    "$HOME/.config/long-context-asr/linear.env"
    "$HOME/.config/sap-longcontext/linear.env"
  )
  for env_file in "${env_candidates[@]}"; do
    if [[ -f "$env_file" ]]; then
      set +u
      set -a
      # shellcheck disable=SC1090
      source "$env_file"
      set +a
      set -u
      if [[ -n "${LINEAR_API_KEY:-}" ]]; then
        echo "linear_key_source=${env_file}" >> "$SUMMARY_FILE"
        return 0
      fi
    fi
  done

  echo "linear_key_source=missing" >> "$SUMMARY_FILE"
  return 1
}

if [[ "${ROB209_EVAL_RUN_DIR_EXEC:-0}" != "1" ]]; then
  mkdir -p "$RUN_DIR"
  cp "${BASH_SOURCE[0]}" "$EXECUTED_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_stanage_callback.py" "$CALLBACK_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_job_callback.py" "$CALLBACK_HELPER"
  chmod +x "$EXECUTED_SCRIPT" "$CALLBACK_SCRIPT"
  export ROB209_EVAL_RUN_DIR_EXEC=1
  export ROB209_REPO_DIR="$REPO_DIR"
  export ROB209_REMOTE_BRANCH="$REMOTE_BRANCH"
  export ROB209_TARGET_COMMIT="$TARGET_COMMIT"
  export ROB209_EVAL_RUN_ID="$RUN_ID"
  export ROB209_ARTIFACT_ROOT="$ARTIFACT_ROOT"
  export ROB209_EVAL_RUN_DIR="$RUN_DIR"
  export ROB209_EVAL_OUT_LOG="$OUT_LOG"
  export ROB209_EVAL_ERR_LOG="$ERR_LOG"
  export ROB209_EVAL_SUMMARY_FILE="$SUMMARY_FILE"
  export ROB209_EVAL_CONFIG="$EVAL_CONFIG"
  export ROB209_EVAL_RESULT_CSV="$RESULT_CSV"
  export ROB209_EVAL_EXECUTED_SCRIPT="$EXECUTED_SCRIPT"
  export ROB209_EVAL_CALLBACK_SCRIPT="$CALLBACK_SCRIPT"
  export ROB209_EVAL_CALLBACK_HELPER="$CALLBACK_HELPER"
  exec bash "$EXECUTED_SCRIPT" "$@"
fi

mkdir -p "$RUN_DIR" "$(dirname "$RESULT_CSV")"

on_exit() {
  local status=$?
  trap - EXIT
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
    echo "checkpoint_step=${CHECKPOINT_STEP}"
    echo "dataset=${DATASET}"
    echo "split=${SPLIT}"
    echo "eval_config=${EVAL_CONFIG}"
    echo "result_csv=${RESULT_CSV}"
    echo "decode_mode=${DECODE_MODE}"
    echo "use_kv_cache=true"
    echo "chunked_kv_cache=true"
    echo "chunk_size=${CHUNK_SIZE}"
    echo "subsampling_history_frames=${SUBSAMPLING_HISTORY_FRAMES}"
    echo "decoder_history_frames=${DECODER_HISTORY_FRAMES}"
    echo "final_flush_frames=${FINAL_FLUSH_FRAMES}"
    echo "max_sequence_length=${MAX_SEQUENCE_LENGTH}"
    echo "max_output_frames=${MAX_OUTPUT_FRAMES}"
    echo "break_eval=${BREAK_EVAL}"
    echo "include_per_recording=${INCLUDE_PER_RECORDING}"
    echo "slurm_partition=${SLURM_JOB_PARTITION:-unset}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  } >> "$SUMMARY_FILE"
  if [[ "${ROB209_ENABLE_CALLBACK:-1}" == "1" ]]; then
    load_linear_api_key || true
    callback_args=(
      "$CALLBACK_SCRIPT"
      --issue-id ROB-209
      --state-name Todo
      --exit-code "$status"
      --log-out "$OUT_LOG"
      --log-err "$ERR_LOG"
      --artifact-path "$RUN_DIR"
      --summary-file "$SUMMARY_FILE"
      --title "ROB-209 Stanage online streaming eval finished"
      --job-label "Stanage online streaming eval"
      --metadata "branch=${REMOTE_BRANCH}"
      --metadata "commit=${TARGET_COMMIT}"
      --metadata "checkpoint=${CHECKPOINT}"
      --metadata "dataset=${DATASET}"
      --metadata "split=${SPLIT}"
      --metadata "result_csv=${RESULT_CSV}"
      --metadata "decode_mode=${DECODE_MODE}"
      --metadata "chunked_kv_cache=true"
      --metadata "chunk_size=${CHUNK_SIZE}"
      --metadata "decoder_history_frames=${DECODER_HISTORY_FRAMES}"
      --metadata "subsampling_history_frames=${SUBSAMPLING_HISTORY_FRAMES}"
      --metadata "final_flush_frames=${FINAL_FLUSH_FRAMES}"
      --metadata "break_eval=${BREAK_EVAL}"
    )
    if [[ "${ROB209_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
      callback_args+=(--dry-run)
    fi
    "$CALLBACK_PYTHON" "${callback_args[@]}" >> "$SUMMARY_FILE" 2>&1
    callback_status=$?
    echo "callback_exit_code=${callback_status}" >> "$SUMMARY_FILE"
    if [[ "$callback_status" -ne 0 ]]; then
      echo "ROB-209 Linear callback failed with exit status ${callback_status}" >&2
    fi
  fi
  exit "$status"
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
current_commit="$(git rev-parse HEAD)"
if [[ "$current_commit" == "$TARGET_COMMIT" ]]; then
  echo "repo_update=already_at_target" >> "$SUMMARY_FILE"
elif [[ "${ROB209_SKIP_GIT_UPDATE:-0}" == "1" ]]; then
  echo "ROB209_SKIP_GIT_UPDATE=1 but repo is at $current_commit, expected $TARGET_COMMIT" >&2
  exit 2
else
  if ! git cat-file -e "${TARGET_COMMIT}^{commit}" 2>/dev/null; then
    git fetch origin "$REMOTE_BRANCH"
  fi
  git checkout --detach "$TARGET_COMMIT"
fi
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS="$OMP_THREADS"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "missing checkpoint: $CHECKPOINT" >&2
  exit 2
fi
if [[ "$DATASET" == "floras50" ]]; then
  if [[ ! -d "$FLORAS50_ROOT/$SPLIT/audio" || ! -d "$FLORAS50_ROOT/$SPLIT/text" ]]; then
    echo "missing Floras50 $SPLIT audio/text directories under $FLORAS50_ROOT" >&2
    exit 2
  fi
fi

if [[ -z "$CHECKPOINT_STEP" ]]; then
  checkpoint_name="$(basename "$CHECKPOINT")"
  CHECKPOINT_STEP="${checkpoint_name#step_}"
  CHECKPOINT_STEP="${CHECKPOINT_STEP%.pt}"
fi
MODEL_NAME="${ROB209_EVAL_MODEL_NAME:-rob209_step_${CHECKPOINT_STEP}_online_chunked_kv}"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

python - "$EVAL_CONFIG" "$RESULT_CSV" "$CHECKPOINT" "$CHECKPOINT_STEP" "$MODEL_NAME" \
  "$DATASET" "$SPLIT" "$MAX_SEQUENCE_LENGTH" "$MAX_OUTPUT_FRAMES" \
  "$(bool_yaml "$BREAK_EVAL")" "$(bool_yaml "$INCLUDE_PER_RECORDING")" "$(bool_yaml "$VERBOSE")" \
  "$DECODE_MODE" "$TEMPERATURE" "$CHUNK_SIZE" "$SUBSAMPLING_HISTORY_FRAMES" \
  "$DECODER_HISTORY_FRAMES" "$FINAL_FLUSH_FRAMES" <<'PY'
import sys
from pathlib import Path

(
    config_path,
    result_csv,
    checkpoint,
    checkpoint_step,
    model_name,
    dataset,
    split,
    max_sequence_length,
    max_output_frames,
    break_eval,
    include_per_recording,
    verbose,
    decode_mode,
    temperature,
    chunk_size,
    subsampling_history,
    decoder_history,
    final_flush,
) = sys.argv[1:19]

extra_decode = ""
if int(max_output_frames) > 0:
    extra_decode = f"        max_output_frames: {int(max_output_frames)}\n"

config = f"""models:
  - name: {model_name}
    repeat: 1
    seq_len: {int(max_sequence_length)}
    overlap_ratio: 0.0
    path: "{checkpoint}"
    checkpoint_step: {checkpoint_step}
    decoding: online_chunked_kv
    args:
      transcribe_kwargs:
        decode_mode: "{decode_mode}"
        temperature: {float(temperature)}
        use_kv_cache: true
        chunked_kv_cache: true
        kv_cache_chunk_spectrogram_length: {int(chunk_size)}
        subsampling_history_spectrogram_length: {int(subsampling_history)}
        decoder_history_spectrogram_length: {int(decoder_history)}
        final_flush_spectrogram_length: {int(final_flush)}
{extra_decode}datasets:
  - name: {dataset}
    splits: ["{split}"]
args:
  model_class: StreamingDecoderASR
  run_eval_with: run
  save_dataframe_path: "{result_csv}"
  verbose: {verbose}
  include_per_recording_evaluations: {include_per_recording}
  evaluation_mode: averaged_moving_window
  break_eval: {break_eval}
"""

Path(config_path).write_text(config, encoding="utf-8")
PY

{
  echo "checkpoint_exists=yes"
  echo "floras50_root=${FLORAS50_ROOT}"
  echo "eval_config=${EVAL_CONFIG}"
  echo "result_csv=${RESULT_CSV}"
  echo "model_name=${MODEL_NAME}"
  echo "checkpoint_step=${CHECKPOINT_STEP}"
  echo "omp_threads=${OMP_THREADS}"
} >> "$SUMMARY_FILE"

cd "$REPO_DIR/eval"
python eval_manager.py -config "$EVAL_CONFIG"

python - "$RESULT_CSV" "$SUMMARY_FILE" "$DATASET" "$SPLIT" <<'PY'
import csv
import sys

csv_path, summary_path, dataset, split = sys.argv[1:5]
rows = list(csv.DictReader(open(csv_path, newline="")))
aggregate = [
    row
    for row in rows
    if row.get("dataset") == dataset and row.get("split") == split and row.get("recording") == "all"
]
with open(summary_path, "a", encoding="utf-8") as handle:
    handle.write(f"rows={len(rows)}\n")
    if not aggregate:
        handle.write(f"{dataset}_{split}_aggregate=missing\n")
    for row in aggregate:
        handle.write(
            f"{dataset}_{split}_aggregate="
            f"name={row.get('name')} "
            f"step={row.get('checkpoint_step')} "
            f"wer={row.get('wer')} "
            f"cer={row.get('cer')} "
            f"words={row.get('words')} "
            f"chars={row.get('chars')} "
            f"ins_rate={row.get('ins_rate')} "
            f"del_rate={row.get('del_rate')} "
            f"sub_rate={row.get('sub_rate')} "
            f"hyp_words={row.get('hyp_words')} "
            f"ref_words={row.get('ref_words')}\n"
        )
PY
