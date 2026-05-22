#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${ROB119_REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODE="${ROB119_MODE:-full}"
RUN_ID="${ROB119_RUN_ID:-rob119-${MODE}-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ROB119_ARTIFACT_ROOT:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-119}"
RUN_DIR="${ROB119_RUN_DIR:-$ARTIFACT_ROOT/$RUN_ID}"
OUT_LOG="${ROB119_OUT_LOG:-$RUN_DIR/${RUN_ID}.out}"
ERR_LOG="${ROB119_ERR_LOG:-$RUN_DIR/${RUN_ID}.err}"
SUMMARY_FILE="${ROB119_SUMMARY_FILE:-$RUN_DIR/${RUN_ID}.summary.txt}"
RESULT_SUMMARY="${ROB119_RESULT_SUMMARY:-$RUN_DIR/OUTCOME.md}"
EXECUTED_SCRIPT="${ROB119_EXECUTED_SCRIPT:-$RUN_DIR/$(basename "$0")}"
CALLBACK_SCRIPT="${ROB119_CALLBACK_SCRIPT:-$RUN_DIR/linear_mimas_callback.py}"
CALLBACK_LIB="${ROB119_CALLBACK_LIB:-$RUN_DIR/linear_job_callback.py}"
LINEAR_KEY_FILE="${ROB119_LINEAR_KEY_FILE:-$ARTIFACT_ROOT/.linear_api_key}"
CALLBACK_PYTHON="${ROB119_CALLBACK_PYTHON:-python3}"

TEDLIUM_ROOT="${ROB119_TEDLIUM_ROOT:-/store/store4/data/TEDLIUM_release1/legacy}"
CHECKPOINT_CACHE="${ROB119_CHECKPOINT_CACHE:-$ARTIFACT_ROOT/source-checkpoints}"
SOURCE_CHECKPOINT_DIR="${ROB119_SOURCE_CHECKPOINT_DIR:-/mnt/parscratch/users/acp21rjf/spotify/bestrq_ssl/rob100_papermask_p012_l4_sc_off_20260520}"
TRAIN_MANIFEST="${ROB119_TRAIN_MANIFEST:-$RUN_DIR/tedlium_train_manifest.json}"
SPEC_DIR="${ROB119_SPEC_DIR:-$ARTIFACT_ROOT/tedlium_train_specs}"
TRANSCRIPT_DIR="${ROB119_TRANSCRIPT_DIR:-$ARTIFACT_ROOT/tedlium_train_transcripts}"
CHECKPOINT_ROOT="${ROB119_CHECKPOINT_ROOT:-$ARTIFACT_ROOT/checkpoints/$RUN_ID}"
RUN_MANIFEST="${ROB119_RUN_MANIFEST:-$RUN_DIR/run_manifest.json}"
EVAL_CONFIG="${ROB119_EVAL_CONFIG:-$RUN_DIR/rob119_tedlium_eval.yaml}"
RESULT_CSV="${ROB119_RESULT_CSV:-$RUN_DIR/tedlium_test_results.csv}"

NUM_WORKERS="${ROB119_NUM_WORKERS:-0}"
PREFETCH="${ROB119_PREFETCH:-1}"
PIN_MEMORY="${ROB119_PIN_MEMORY:-0}"
BATCH_SIZE="${ROB119_BATCH_SIZE:-16}"
MAX_EPOCHS="${ROB119_MAX_EPOCHS:-10}"
LEARNING_RATE="${ROB119_LEARNING_RATE:-1e-3}"
CHECKPOINT_LABELS="${ROB119_CHECKPOINT_LABELS:-primary}"
SMOKE_MAX_RECORDS="${ROB119_SMOKE_MAX_RECORDS:-2}"
SMOKE_BATCH_SIZE="${ROB119_SMOKE_BATCH_SIZE:-2}"
FULL_MAX_RECORDS="${ROB119_FULL_MAX_RECORDS:-}"
DISABLE_WANDB="${ROB119_DISABLE_WANDB:-0}"
SMOKE_ENABLE_WANDB="${ROB119_SMOKE_ENABLE_WANDB:-0}"

if [[ "${ROB119_RUN_DIR_EXEC:-0}" != "1" ]]; then
  mkdir -p "$RUN_DIR"
  cp "${BASH_SOURCE[0]}" "$EXECUTED_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_mimas_callback.py" "$CALLBACK_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_job_callback.py" "$CALLBACK_LIB"
  chmod +x "$EXECUTED_SCRIPT"
  export ROB119_RUN_DIR_EXEC=1
  export ROB119_REPO_DIR="$REPO_DIR"
  export ROB119_MODE="$MODE"
  export ROB119_RUN_ID="$RUN_ID"
  export ROB119_ARTIFACT_ROOT="$ARTIFACT_ROOT"
  export ROB119_RUN_DIR="$RUN_DIR"
  export ROB119_OUT_LOG="$OUT_LOG"
  export ROB119_ERR_LOG="$ERR_LOG"
  export ROB119_SUMMARY_FILE="$SUMMARY_FILE"
  export ROB119_RESULT_SUMMARY="$RESULT_SUMMARY"
  export ROB119_EXECUTED_SCRIPT="$EXECUTED_SCRIPT"
  export ROB119_CALLBACK_SCRIPT="$CALLBACK_SCRIPT"
  exec bash "$EXECUTED_SCRIPT" "$@"
fi

mkdir -p "$RUN_DIR" "$CHECKPOINT_CACHE" "$CHECKPOINT_ROOT"
exec > >(tee -a "$OUT_LOG") 2> >(tee -a "$ERR_LOG" >&2)

on_exit() {
  local status=$?
  set +e
  {
    echo "ended_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_code=${status}"
    echo "run_id=${RUN_ID}"
    echo "mode=${MODE}"
    echo "host=$(hostname)"
    echo "repo_dir=${REPO_DIR}"
    echo "commit=$(cd "$REPO_DIR" && git rev-parse HEAD 2>/dev/null)"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "source_checkpoint_dir=${SOURCE_CHECKPOINT_DIR}"
    echo "run_manifest=${RUN_MANIFEST}"
    echo "result_csv=${RESULT_CSV}"
    echo "result_summary=${RESULT_SUMMARY}"
    echo "checkpoint_root=${CHECKPOINT_ROOT}"
    echo "learning_rate=${LEARNING_RATE}"
    echo "max_epochs=${MAX_EPOCHS}"
    echo "checkpoint_labels=${CHECKPOINT_LABELS}"
  } >> "$SUMMARY_FILE"
  if [[ "${ROB119_ENABLE_CALLBACK:-1}" == "1" ]]; then
    if [[ -z "${LINEAR_API_KEY:-}" && -f "$LINEAR_KEY_FILE" ]]; then
      export LINEAR_API_KEY
      LINEAR_API_KEY="$(cat "$LINEAR_KEY_FILE")"
    fi
    callback_args=(
      --issue-id ROB-119
      --state-name Todo
      --job-id "$RUN_ID"
      --job-label "Mimas screen run"
      --exit-code "$status"
      --log-out "$OUT_LOG"
      --log-err "$ERR_LOG"
      --output-path "$RUN_DIR"
      --summary-file "$RESULT_SUMMARY"
      --title "ROB-119 ROB-100 weighted BiLSTM CTC probe finished"
    )
    if [[ "${ROB119_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
      callback_args+=(--dry-run)
    fi
    "$CALLBACK_PYTHON" "$CALLBACK_SCRIPT" "${callback_args[@]}" >> "$SUMMARY_FILE" 2>&1
  fi
  return "$status"
}
trap on_exit EXIT
trap 'trap - TERM INT; exit 143' TERM
trap 'trap - INT; exit 130' INT

{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "run_id=${RUN_ID}"
  echo "mode=${MODE}"
  echo "host=$(hostname)"
  echo "repo_dir=${REPO_DIR}"
  echo "commit=$(cd "$REPO_DIR" && git rev-parse HEAD)"
  echo "executed_script=${EXECUTED_SCRIPT}"
  echo "callback_script=${CALLBACK_SCRIPT}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "source_checkpoint_dir=${SOURCE_CHECKPOINT_DIR}"
  echo "learning_rate=${LEARNING_RATE}"
  echo "max_epochs=${MAX_EPOCHS}"
  echo "checkpoint_labels=${CHECKPOINT_LABELS}"
} > "$SUMMARY_FILE"

if [[ "${ROB119_CALLBACK_ONLY:-0}" == "1" ]]; then
  echo "callback-only dry run requested"
  exit 0
fi

cd "$REPO_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LCASR_TEDLIUM_ROOT="$TEDLIUM_ROOT"
export ROB119_RUN_MANIFEST="$RUN_MANIFEST"

for checkpoint in step_105360.pt step_99264.pt; do
  if [[ ! -s "$CHECKPOINT_CACHE/$checkpoint" ]]; then
    rsync -av "acp21rjf@stanage.shef.ac.uk:${SOURCE_CHECKPOINT_DIR}/${checkpoint}" "$CHECKPOINT_CACHE/"
  fi
done

manifest_args=(
  --tedlium-root "$TEDLIUM_ROOT"
  --split train
  --manifest-out "$TRAIN_MANIFEST"
  --spec-dir "$SPEC_DIR"
  --transcript-dir "$TRANSCRIPT_DIR"
)
if [[ "$MODE" == "smoke" ]]; then
  manifest_args+=(--max-records "$SMOKE_MAX_RECORDS")
elif [[ -n "$FULL_MAX_RECORDS" ]]; then
  manifest_args+=(--max-records "$FULL_MAX_RECORDS")
fi
python symphony/scripts/prepare_rob91_tedlium_manifest.py "${manifest_args[@]}"

prepare_args=(
  --run-dir "$RUN_DIR"
  --checkpoint-cache "$CHECKPOINT_CACHE"
  --checkpoint-root "$CHECKPOINT_ROOT"
  --train-manifest "$TRAIN_MANIFEST"
  --manifest-out "$RUN_MANIFEST"
  --source-checkpoint-dir "$SOURCE_CHECKPOINT_DIR"
  --batch-size "$BATCH_SIZE"
  --smoke-batch-size "$SMOKE_BATCH_SIZE"
  --max-epochs "$MAX_EPOCHS"
  --learning-rate "$LEARNING_RATE"
  --checkpoint-labels "$CHECKPOINT_LABELS"
)
if [[ "$MODE" == "smoke" ]]; then
  prepare_args+=(--smoke)
  if [[ "$SMOKE_ENABLE_WANDB" == "1" ]]; then
    prepare_args+=(--enable-smoke-wandb)
  else
    prepare_args+=(--disable-wandb)
  fi
elif [[ "$DISABLE_WANDB" == "1" ]]; then
  prepare_args+=(--disable-wandb)
fi
python symphony/scripts/prepare_rob119_probe_config.py "${prepare_args[@]}"

python - <<'PY'
import json, os
manifest = json.load(open(os.environ["ROB119_RUN_MANIFEST"]))
for run in manifest["runs"]:
    if not os.path.exists(run["local_checkpoint"]):
        raise SystemExit(f"missing checkpoint: {run['local_checkpoint']}")
print("checkpoint path check ok")
PY

while read -r config_path; do
  train_args=(-config "$config_path" -num_workers "$NUM_WORKERS" -prefetch "$PREFETCH" -reset_step)
  if [[ "$PIN_MEMORY" == "1" ]]; then
    train_args+=(-pin_memory)
  fi
  if [[ "$MODE" == "smoke" ]]; then
    train_args+=(--max_records "$SMOKE_MAX_RECORDS" --max_steps "$SMOKE_MAX_RECORDS")
    if [[ "$SMOKE_ENABLE_WANDB" != "1" ]]; then
      train_args+=(--disable_wandb)
    fi
  fi
  python exp/train.py "${train_args[@]}"
done < <(python - <<'PY'
import json, os
manifest = json.load(open(os.environ["ROB119_RUN_MANIFEST"]))
for run in manifest["runs"]:
    print(run["train_config"])
PY
)

eval_args=(
  --run-manifest "$RUN_MANIFEST"
  --eval-config-out "$EVAL_CONFIG"
  --csv-out "$RESULT_CSV"
)
if [[ "$MODE" == "smoke" ]]; then
  eval_args+=(--break-eval)
else
  eval_args+=(--include-per-recording)
fi
python symphony/scripts/make_rob119_eval_config.py "${eval_args[@]}"

(
  cd eval
  python eval_manager.py -config "$EVAL_CONFIG"
)

python symphony/scripts/summarize_rob119_results.py \
  --run-manifest "$RUN_MANIFEST" \
  --csv "$RESULT_CSV" \
  --summary-out "$RESULT_SUMMARY"

cat "$RESULT_SUMMARY"
