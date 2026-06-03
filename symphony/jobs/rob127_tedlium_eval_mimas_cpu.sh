#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${ROB127_REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN_ID="${ROB127_RUN_ID:-rob127-tedlium-cpu-kvcache-greedy-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ROB127_ARTIFACT_ROOT:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-127/eval}"
RUN_DIR="${ROB127_RUN_DIR:-$ARTIFACT_ROOT/$RUN_ID}"
OUT_LOG="${ROB127_OUT_LOG:-$RUN_DIR/${RUN_ID}.out}"
ERR_LOG="${ROB127_ERR_LOG:-$RUN_DIR/${RUN_ID}.err}"
SUMMARY_FILE="${ROB127_SUMMARY_FILE:-$RUN_DIR/${RUN_ID}.summary.txt}"
EVAL_CONFIG="${ROB127_EVAL_CONFIG:-$RUN_DIR/rob127_tedlium_eval.yaml}"
RESULT_CSV="${ROB127_RESULT_CSV:-$RUN_DIR/rob127_tedlium_eval.csv}"
EXECUTED_SCRIPT="${ROB127_EXECUTED_SCRIPT:-$RUN_DIR/$(basename "$0")}"
CALLBACK_SCRIPT="${ROB127_CALLBACK_SCRIPT:-$RUN_DIR/linear_mimas_callback.py}"
CALLBACK_HELPER="${ROB127_CALLBACK_HELPER:-$RUN_DIR/linear_job_callback.py}"
TEDLIUM_ROOT="${ROB127_TEDLIUM_ROOT:-/store/store4/data/TEDLIUM_release1/legacy}"
CHECKPOINT_DIR="${ROB127_CHECKPOINT_DIR:-/store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_rope_rob127_rl_grpo_rob127-rope-streaming-rl-grpo-b32-r32-tol2-constant-logprobmb1-10k-step0-20260524T221606Z}"
CHECKPOINT_STEPS="${ROB127_CHECKPOINT_STEPS:-0 500 1000 1500}"
BREAK_EVAL="${ROB127_EVAL_BREAK:-0}"
INCLUDE_PER_RECORDING="${ROB127_INCLUDE_PER_RECORDING:-1}"
MAX_KV_CACHE_SPECTROGRAM_LENGTH="${ROB127_MAX_KV_CACHE_SPECTROGRAM_LENGTH:-2048}"
DECODE_MODE="${ROB127_DECODE_MODE:-greedy}"
TEMPERATURE="${ROB127_TEMPERATURE:-1.0}"
OMP_THREADS="${ROB127_OMP_THREADS:-16}"
CALLBACK_PYTHON="${ROB127_CALLBACK_PYTHON:-python3}"
LINEAR_KEY_FILE="${ROB127_LINEAR_KEY_FILE:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-127/.linear_api_key}"

if [[ "${ROB127_RUN_DIR_EXEC:-0}" != "1" ]]; then
  mkdir -p "$RUN_DIR" "$ARTIFACT_ROOT"
  cp "${BASH_SOURCE[0]}" "$EXECUTED_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_mimas_callback.py" "$CALLBACK_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_job_callback.py" "$CALLBACK_HELPER"
  chmod +x "$EXECUTED_SCRIPT" "$CALLBACK_SCRIPT"
  export ROB127_RUN_DIR_EXEC=1
  export ROB127_REPO_DIR="$REPO_DIR"
  export ROB127_RUN_ID="$RUN_ID"
  export ROB127_ARTIFACT_ROOT="$ARTIFACT_ROOT"
  export ROB127_RUN_DIR="$RUN_DIR"
  export ROB127_OUT_LOG="$OUT_LOG"
  export ROB127_ERR_LOG="$ERR_LOG"
  export ROB127_SUMMARY_FILE="$SUMMARY_FILE"
  export ROB127_EVAL_CONFIG="$EVAL_CONFIG"
  export ROB127_RESULT_CSV="$RESULT_CSV"
  export ROB127_EXECUTED_SCRIPT="$EXECUTED_SCRIPT"
  export ROB127_CALLBACK_SCRIPT="$CALLBACK_SCRIPT"
  export ROB127_CALLBACK_HELPER="$CALLBACK_HELPER"
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
    echo "run_id=${RUN_ID}"
    echo "host=$(hostname)"
    echo "repo_dir=${REPO_DIR}"
    echo "branch=$(cd "$REPO_DIR" && git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    echo "commit=$(cd "$REPO_DIR" && git rev-parse HEAD 2>/dev/null)"
    echo "checkpoint_dir=${CHECKPOINT_DIR}"
    echo "checkpoint_steps=${CHECKPOINT_STEPS}"
    echo "tedlium_root=${TEDLIUM_ROOT}"
    echo "eval_config=${EVAL_CONFIG}"
    echo "result_csv=${RESULT_CSV}"
    echo "decode_mode=${DECODE_MODE}"
    echo "temperature=${TEMPERATURE}"
    echo "use_kv_cache=true"
    echo "max_kv_cache_spectrogram_length=${MAX_KV_CACHE_SPECTROGRAM_LENGTH}"
    echo "break_eval=${BREAK_EVAL}"
  } >> "$SUMMARY_FILE"
  if [[ "${ROB127_ENABLE_CALLBACK:-1}" == "1" ]]; then
    if [[ -z "${LINEAR_API_KEY:-}" && -f "$LINEAR_KEY_FILE" ]]; then
      export LINEAR_API_KEY
      LINEAR_API_KEY="$(cat "$LINEAR_KEY_FILE")"
    fi
    callback_args=(
      "$CALLBACK_SCRIPT"
      --issue-id ROB-127
      --state-name Todo
      --job-id "$RUN_ID"
      --job-label "Mimas CPU TEDLIUM eval"
      --exit-code "$status"
      --log-out "$OUT_LOG"
      --log-err "$ERR_LOG"
      --artifact-path "$RUN_DIR"
      --summary-file "$SUMMARY_FILE"
      --title "ROB-127 TEDLIUM CPU eval finished"
      --metadata "checkpoint_dir=${CHECKPOINT_DIR}"
      --metadata "checkpoint_steps=${CHECKPOINT_STEPS}"
      --metadata "result_csv=${RESULT_CSV}"
      --metadata "decode_mode=${DECODE_MODE}"
      --metadata "use_kv_cache=true"
      --metadata "max_kv_cache_spectrogram_length=${MAX_KV_CACHE_SPECTROGRAM_LENGTH}"
      --metadata "break_eval=${BREAK_EVAL}"
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
  echo "checkpoint_dir=${CHECKPOINT_DIR}"
  echo "checkpoint_steps=${CHECKPOINT_STEPS}"
} > "$SUMMARY_FILE"

if [[ "${ROB127_CALLBACK_ONLY:-0}" == "1" ]]; then
  echo "callback-only dry run requested"
  exit 0
fi

cd "$REPO_DIR"
export PYTHONPATH="$PWD"
export LCASR_TEDLIUM_ROOT="$TEDLIUM_ROOT"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="$OMP_THREADS"

python - "$EVAL_CONFIG" "$RESULT_CSV" "$CHECKPOINT_DIR" "$CHECKPOINT_STEPS" "$DECODE_MODE" "$TEMPERATURE" "$MAX_KV_CACHE_SPECTROGRAM_LENGTH" "$BREAK_EVAL" "$INCLUDE_PER_RECORDING" <<'PY'
import sys
from pathlib import Path

config_path, result_csv, checkpoint_dir, steps_text, decode_mode, temperature, cache_spec_len, break_eval, include_per_recording = sys.argv[1:10]
checkpoint_dir = Path(checkpoint_dir)
models = []
for step in steps_text.split():
    checkpoint = checkpoint_dir / f"step_{step}.pt"
    if not checkpoint.exists():
        raise SystemExit(f"missing checkpoint: {checkpoint}")
    models.append(
        f"""  - name: rob127_step_{step}_kvcache_greedy
    path: "{checkpoint}"
    seq_len: -1
    overlap_ratio: 0
    repeat: 1
    checkpoint_step: {step}
"""
    )

config = f"""models:
{''.join(models)}datasets:
  - name: tedlium
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
  echo "checkpoint_exists=$(for step in $CHECKPOINT_STEPS; do test -f "$CHECKPOINT_DIR/step_${step}.pt" || exit 1; done && echo yes || echo no)"
  echo "tedlium_root_exists=$(test -d "$TEDLIUM_ROOT" && echo yes || echo no)"
  echo "eval_config=${EVAL_CONFIG}"
  echo "result_csv=${RESULT_CSV}"
  echo "decode_mode=${DECODE_MODE}"
  echo "temperature=${TEMPERATURE}"
  echo "break_eval=${BREAK_EVAL}"
  echo "omp_threads=${OMP_THREADS}"
} >> "$SUMMARY_FILE"

cd "$REPO_DIR/eval"
python eval_manager.py -config "$EVAL_CONFIG"

python - "$RESULT_CSV" "$SUMMARY_FILE" <<'PY'
import csv
import sys

csv_path, summary_path = sys.argv[1:3]
rows = list(csv.DictReader(open(csv_path, newline="")))
aggregate = [
    row
    for row in rows
    if row.get("dataset") == "tedlium" and row.get("split") == "test" and row.get("recording") == "all"
]
with open(summary_path, "a", encoding="utf-8") as handle:
    handle.write(f"rows={len(rows)}\n")
    if not aggregate:
        handle.write("tedlium_test_aggregate=missing\n")
    for row in aggregate:
        handle.write(
            "tedlium_test_aggregate="
            f"name={row.get('name')} "
            f"step={row.get('checkpoint_step')} "
            f"wer={row.get('wer')} "
            f"words={row.get('words')} "
            f"ins_rate={row.get('ins_rate')} "
            f"del_rate={row.get('del_rate')} "
            f"sub_rate={row.get('sub_rate')}\n"
        )
PY
