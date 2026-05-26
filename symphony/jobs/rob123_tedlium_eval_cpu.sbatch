#!/bin/bash
#SBATCH --job-name=rob123-ted-cpu
#SBATCH --partition=interactive
#SBATCH --time=08:00:00
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=16
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval/rob123-ted-cpu-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval/rob123-ted-cpu-%j.err

set -Eeuo pipefail

REPO_DIR="${ROB123_REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-123}"
REMOTE_BRANCH="${ROB123_REMOTE_BRANCH:-symphony/ROB-123-rope-full-spotify-stanage}"
TARGET_COMMIT="${ROB123_TARGET_COMMIT:?ROB123_TARGET_COMMIT must pin the code revision}"
ARTIFACT_ROOT="${ROB123_ARTIFACT_ROOT:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval}"
RUN_ID="${ROB123_RUN_ID:-rob123-tedlium-cpu-kvcache-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${ROB123_RUN_DIR:-$ARTIFACT_ROOT/$RUN_ID}"
OUT_LOG="${ROB123_OUT_LOG:-$ARTIFACT_ROOT/rob123-ted-cpu-${SLURM_JOB_ID:-manual}.out}"
ERR_LOG="${ROB123_ERR_LOG:-$ARTIFACT_ROOT/rob123-ted-cpu-${SLURM_JOB_ID:-manual}.err}"
SUMMARY_FILE="${ROB123_SUMMARY_FILE:-$RUN_DIR/${RUN_ID}.summary.txt}"
EVAL_CONFIG="${ROB123_EVAL_CONFIG:-$RUN_DIR/rob123_tedlium_eval.yaml}"
RESULT_CSV="${ROB123_RESULT_CSV:-$RUN_DIR/rob123_tedlium_eval.csv}"
EXECUTED_SCRIPT="${ROB123_EXECUTED_SCRIPT:-$RUN_DIR/$(basename "$0")}"
CALLBACK_SCRIPT="${ROB123_CALLBACK_SCRIPT:-$RUN_DIR/linear_stanage_callback.py}"
CHECKPOINT="${ROB123_CHECKPOINT:-/mnt/parscratch/users/acp21rjf/spotify/streaming_decoder_asr_100m_rope_rob123_full_spotify_2epoch_delay0p5_rob123-rope-full-spotify-2epoch-delay0p5-20260523T091306Z/step_272362.pt}"
TEDLIUM_ROOT="${ROB123_TEDLIUM_ROOT:-/mnt/parscratch/users/acp21rjf/TEDLIUM_release1}"
BREAK_EVAL="${ROB123_EVAL_BREAK:-0}"
INCLUDE_PER_RECORDING="${ROB123_INCLUDE_PER_RECORDING:-1}"
MAX_KV_CACHE_SPECTROGRAM_LENGTH="${ROB123_MAX_KV_CACHE_SPECTROGRAM_LENGTH:-2048}"
DECODE_MODE="${ROB123_DECODE_MODE:-greedy}"
TEMPERATURE="${ROB123_TEMPERATURE:-1.0}"
CALLBACK_PYTHON="${ROB123_CALLBACK_PYTHON:-python3}"
LINEAR_KEY_FILE="${ROB123_LINEAR_KEY_FILE:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/.linear_api_key}"

if [[ "${ROB123_RUN_DIR_EXEC:-0}" != "1" ]]; then
  mkdir -p "$RUN_DIR" "$ARTIFACT_ROOT"
  cp "${BASH_SOURCE[0]}" "$EXECUTED_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_stanage_callback.py" "$CALLBACK_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_job_callback.py" "$RUN_DIR/linear_job_callback.py"
  chmod +x "$EXECUTED_SCRIPT" "$CALLBACK_SCRIPT"
  export ROB123_RUN_DIR_EXEC=1
  export ROB123_REPO_DIR="$REPO_DIR"
  export ROB123_REMOTE_BRANCH="$REMOTE_BRANCH"
  export ROB123_TARGET_COMMIT="$TARGET_COMMIT"
  export ROB123_ARTIFACT_ROOT="$ARTIFACT_ROOT"
  export ROB123_RUN_ID="$RUN_ID"
  export ROB123_RUN_DIR="$RUN_DIR"
  export ROB123_OUT_LOG="$OUT_LOG"
  export ROB123_ERR_LOG="$ERR_LOG"
  export ROB123_SUMMARY_FILE="$SUMMARY_FILE"
  export ROB123_EVAL_CONFIG="$EVAL_CONFIG"
  export ROB123_RESULT_CSV="$RESULT_CSV"
  export ROB123_EXECUTED_SCRIPT="$EXECUTED_SCRIPT"
  export ROB123_CALLBACK_SCRIPT="$CALLBACK_SCRIPT"
  exec bash "$EXECUTED_SCRIPT" "$@"
fi

mkdir -p "$RUN_DIR" "$(dirname "$RESULT_CSV")"

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
  } >> "$SUMMARY_FILE"
  if [[ "${ROB123_ENABLE_CALLBACK:-1}" == "1" ]]; then
    if [[ -z "${LINEAR_API_KEY:-}" && -f "$LINEAR_KEY_FILE" ]]; then
      export LINEAR_API_KEY
      LINEAR_API_KEY="$(cat "$LINEAR_KEY_FILE")"
    fi
    callback_args=(
      "$CALLBACK_SCRIPT"
      --issue-id ROB-123
      --state-name Todo
      --exit-code "$status"
      --log-out "$OUT_LOG"
      --log-err "$ERR_LOG"
      --artifact-path "$RUN_DIR"
      --checkpoint-path "$CHECKPOINT"
      --summary-file "$SUMMARY_FILE"
      --title "ROB-123 TEDLIUM CPU eval finished"
      --job-label "Stanage CPU TEDLIUM eval"
      --metadata "branch=${REMOTE_BRANCH}"
      --metadata "commit=${TARGET_COMMIT}"
      --metadata "checkpoint=${CHECKPOINT}"
      --metadata "result_csv=${RESULT_CSV}"
      --metadata "decode_mode=${DECODE_MODE}"
      --metadata "temperature=${TEMPERATURE}"
      --metadata "use_kv_cache=true"
      --metadata "max_kv_cache_spectrogram_length=${MAX_KV_CACHE_SPECTROGRAM_LENGTH}"
      --metadata "break_eval=${BREAK_EVAL}"
    )
    if [[ "${ROB123_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
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

if [[ "${ROB123_CALLBACK_ONLY:-0}" == "1" ]]; then
  echo "callback-only dry run requested"
  exit 0
fi

cd "$REPO_DIR"
git fetch origin "$REMOTE_BRANCH"
git checkout --detach "$TARGET_COMMIT"
export PYTHONPATH="$PWD"
export LCASR_TEDLIUM_ROOT="$TEDLIUM_ROOT"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

cat > "$EVAL_CONFIG" <<YAML
models:
  - name: rob123_rope_full_spotify_2epoch_step272362_kvcache
    path: "$CHECKPOINT"
    seq_len: -1
    overlap_ratio: 0
    repeat: 1
datasets:
  - name: tedlium
    splits: ["test"]
args:
  model_class: StreamingDecoderASR
  run_eval_with: run
  save_dataframe_path: "$RESULT_CSV"
  verbose: false
  include_per_recording_evaluations: $INCLUDE_PER_RECORDING
  break_eval: $BREAK_EVAL
  transcribe_kwargs:
    decode_mode: "$DECODE_MODE"
    temperature: $TEMPERATURE
    use_kv_cache: true
    max_kv_cache_spectrogram_length: $MAX_KV_CACHE_SPECTROGRAM_LENGTH
YAML

{
  echo "checkpoint_exists=$(test -f "$CHECKPOINT" && echo yes || echo no)"
  echo "tedlium_root_exists=$(test -d "$TEDLIUM_ROOT" && echo yes || echo no)"
  echo "eval_config=${EVAL_CONFIG}"
  echo "result_csv=${RESULT_CSV}"
  echo "decode_mode=${DECODE_MODE}"
  echo "temperature=${TEMPERATURE}"
  echo "break_eval=${BREAK_EVAL}"
} >> "$SUMMARY_FILE"

cd "$REPO_DIR/eval"
python eval_manager.py -config "$EVAL_CONFIG"

python - "$RESULT_CSV" "$SUMMARY_FILE" <<'PY'
import csv
import sys

csv_path, summary_path = sys.argv[1:3]
rows = list(csv.DictReader(open(csv_path, newline="")))
aggregate = [row for row in rows if row.get("dataset") == "tedlium" and row.get("split") == "test" and row.get("recording") == "all"]
with open(summary_path, "a") as handle:
    handle.write(f"rows={len(rows)}\n")
    if aggregate:
        row = aggregate[-1]
        handle.write(f"tedlium_test_wer={row.get('wer')}\n")
        handle.write(f"tedlium_test_words={row.get('words')}\n")
        handle.write(f"tedlium_test_ins_rate={row.get('ins_rate')}\n")
        handle.write(f"tedlium_test_del_rate={row.get('del_rate')}\n")
        handle.write(f"tedlium_test_sub_rate={row.get('sub_rate')}\n")
    else:
        handle.write("tedlium_test_aggregate=missing\n")
PY
