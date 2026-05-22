#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${ROB125_REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODE="${ROB125_MODE:-full}"
RUN_ID="${ROB125_RUN_ID:-rob125-${MODE}-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ROB125_ARTIFACT_ROOT:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-125}"
RUN_DIR="${ROB125_RUN_DIR:-$ARTIFACT_ROOT/$RUN_ID}"
OUT_LOG="${ROB125_OUT_LOG:-$RUN_DIR/${RUN_ID}.out}"
ERR_LOG="${ROB125_ERR_LOG:-$RUN_DIR/${RUN_ID}.err}"
SUMMARY_FILE="${ROB125_SUMMARY_FILE:-$RUN_DIR/${RUN_ID}.summary.txt}"
RESULT_SUMMARY="${ROB125_RESULT_SUMMARY:-$RUN_DIR/OUTCOME.md}"
EXECUTED_SCRIPT="${ROB125_EXECUTED_SCRIPT:-$RUN_DIR/$(basename "$0")}"
CALLBACK_SCRIPT="${ROB125_CALLBACK_SCRIPT:-$RUN_DIR/linear_mimas_callback.py}"
CALLBACK_LIB="${ROB125_CALLBACK_LIB:-$RUN_DIR/linear_job_callback.py}"
LINEAR_KEY_FILE="${ROB125_LINEAR_KEY_FILE:-$ARTIFACT_ROOT/.linear_api_key}"
CALLBACK_PYTHON="${ROB125_CALLBACK_PYTHON:-python3}"

TEDLIUM_ROOT="${ROB125_TEDLIUM_ROOT:-/store/store4/data/TEDLIUM_release1/legacy}"
SUPERVISED_CHECKPOINT="${ROB125_SUPERVISED_CHECKPOINT:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-81/checkpoints/supervised_floras50_spotifytok_safe_norm_drop_oov_lr1e-4_12ep_nw0/step_323484.pt}"
KNOWN_GOOD_EVIDENCE="${ROB125_KNOWN_GOOD_EVIDENCE:-ROB-81 TEDLIUM eval WER 0.08258018784334574 over 28215 words}"
TRAIN_MANIFEST="${ROB125_TRAIN_MANIFEST:-$RUN_DIR/tedlium_train_manifest.json}"
SPEC_DIR="${ROB125_SPEC_DIR:-$ARTIFACT_ROOT/tedlium_train_specs}"
TRANSCRIPT_DIR="${ROB125_TRANSCRIPT_DIR:-$ARTIFACT_ROOT/tedlium_train_transcripts}"
CHECKPOINT_ROOT="${ROB125_CHECKPOINT_ROOT:-$ARTIFACT_ROOT/checkpoints/$RUN_ID}"
RUN_MANIFEST="${ROB125_RUN_MANIFEST:-$RUN_DIR/run_manifest.json}"
EVAL_CONFIG="${ROB125_EVAL_CONFIG:-$RUN_DIR/rob125_tedlium_eval.yaml}"
RESULT_CSV="${ROB125_RESULT_CSV:-$RUN_DIR/tedlium_test_results.csv}"

NUM_WORKERS="${ROB125_NUM_WORKERS:-0}"
PREFETCH="${ROB125_PREFETCH:-1}"
PIN_MEMORY="${ROB125_PIN_MEMORY:-0}"
BATCH_SIZE="${ROB125_BATCH_SIZE:-8}"
MAX_EPOCHS="${ROB125_MAX_EPOCHS:-4}"
LEARNING_RATE="${ROB125_LEARNING_RATE:-1e-3}"
LEARNING_RATES="${ROB125_LEARNING_RATES:-}"
SCHEDULER="${ROB125_SCHEDULER:-constant}"
WARMUP_STEPS="${ROB125_WARMUP_STEPS:-0}"
SMOKE_MAX_RECORDS="${ROB125_SMOKE_MAX_RECORDS:-2}"
SMOKE_BATCH_SIZE="${ROB125_SMOKE_BATCH_SIZE:-1}"
FULL_MAX_RECORDS="${ROB125_FULL_MAX_RECORDS:-}"
DISABLE_WANDB="${ROB125_DISABLE_WANDB:-0}"
SMOKE_ENABLE_WANDB="${ROB125_SMOKE_ENABLE_WANDB:-0}"

if [[ "${ROB125_RUN_DIR_EXEC:-0}" != "1" ]]; then
  mkdir -p "$RUN_DIR"
  cp "${BASH_SOURCE[0]}" "$EXECUTED_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_mimas_callback.py" "$CALLBACK_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_job_callback.py" "$CALLBACK_LIB"
  chmod +x "$EXECUTED_SCRIPT"
  export ROB125_RUN_DIR_EXEC=1
  export ROB125_REPO_DIR="$REPO_DIR"
  export ROB125_MODE="$MODE"
  export ROB125_RUN_ID="$RUN_ID"
  export ROB125_ARTIFACT_ROOT="$ARTIFACT_ROOT"
  export ROB125_RUN_DIR="$RUN_DIR"
  export ROB125_OUT_LOG="$OUT_LOG"
  export ROB125_ERR_LOG="$ERR_LOG"
  export ROB125_SUMMARY_FILE="$SUMMARY_FILE"
  export ROB125_RESULT_SUMMARY="$RESULT_SUMMARY"
  export ROB125_EXECUTED_SCRIPT="$EXECUTED_SCRIPT"
  export ROB125_CALLBACK_SCRIPT="$CALLBACK_SCRIPT"
  exec bash "$EXECUTED_SCRIPT" "$@"
fi

mkdir -p \
  "$RUN_DIR" \
  "$CHECKPOINT_ROOT" \
  "$RUN_DIR/tmp" \
  "$RUN_DIR/wandb" \
  "$RUN_DIR/wandb/wandb" \
  "$RUN_DIR/wandb-cache" \
  "$RUN_DIR/wandb-config" \
  "$RUN_DIR/wandb-data" \
  "$RUN_DIR/wandb-artifacts" \
  "$RUN_DIR/xdg-cache"
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
    echo "supervised_checkpoint=${SUPERVISED_CHECKPOINT}"
    echo "known_good_evidence=${KNOWN_GOOD_EVIDENCE}"
    echo "run_manifest=${RUN_MANIFEST}"
    echo "result_csv=${RESULT_CSV}"
    echo "result_summary=${RESULT_SUMMARY}"
    echo "checkpoint_root=${CHECKPOINT_ROOT}"
    echo "learning_rate=${LEARNING_RATE}"
    echo "learning_rates=${LEARNING_RATES:-unset}"
    echo "scheduler=${SCHEDULER}"
    echo "warmup_steps=${WARMUP_STEPS}"
    echo "max_epochs=${MAX_EPOCHS}"
  } >> "$SUMMARY_FILE"
  if [[ "${ROB125_ENABLE_CALLBACK:-1}" == "1" ]]; then
    if [[ -z "${LINEAR_API_KEY:-}" && -f "$LINEAR_KEY_FILE" ]]; then
      export LINEAR_API_KEY
      LINEAR_API_KEY="$(cat "$LINEAR_KEY_FILE")"
    fi
    if [[ -z "${LINEAR_API_KEY:-}" && -f "$REPO_DIR/symphony/.env" ]]; then
      set -a
      # shellcheck disable=SC1091
      source "$REPO_DIR/symphony/.env"
      set +a
    fi
    callback_args=(
      --issue-id ROB-125
      --state-name Todo
      --job-id "$RUN_ID"
      --job-label "Mimas screen run"
      --exit-code "$status"
      --log-out "$OUT_LOG"
      --log-err "$ERR_LOG"
      --output-path "$RUN_DIR"
      --summary-file "$RESULT_SUMMARY"
      --title "ROB-125 supervised weighted BiLSTM CTC probe finished"
    )
    if [[ "${ROB125_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
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
  echo "supervised_checkpoint=${SUPERVISED_CHECKPOINT}"
  echo "known_good_evidence=${KNOWN_GOOD_EVIDENCE}"
  echo "learning_rate=${LEARNING_RATE}"
  echo "learning_rates=${LEARNING_RATES:-unset}"
  echo "scheduler=${SCHEDULER}"
  echo "warmup_steps=${WARMUP_STEPS}"
  echo "max_epochs=${MAX_EPOCHS}"
} > "$SUMMARY_FILE"

if [[ "${ROB125_CALLBACK_ONLY:-0}" == "1" ]]; then
  echo "callback-only dry run requested"
  exit 0
fi

cd "$REPO_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LCASR_TEDLIUM_ROOT="$TEDLIUM_ROOT"
export ROB125_RUN_MANIFEST="$RUN_MANIFEST"
export TMPDIR="${TMPDIR:-$RUN_DIR/tmp}"
export WANDB_DIR="${WANDB_DIR:-$RUN_DIR/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$RUN_DIR/wandb-cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$RUN_DIR/wandb-config}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$RUN_DIR/wandb-data}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$RUN_DIR/wandb-artifacts}"
export WANDB_DISABLE_CODE="${WANDB_DISABLE_CODE:-true}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUN_DIR/xdg-cache}"

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
  --checkpoint-root "$CHECKPOINT_ROOT"
  --train-manifest "$TRAIN_MANIFEST"
  --manifest-out "$RUN_MANIFEST"
  --supervised-checkpoint "$SUPERVISED_CHECKPOINT"
  --known-good-evidence "$KNOWN_GOOD_EVIDENCE"
  --batch-size "$BATCH_SIZE"
  --smoke-batch-size "$SMOKE_BATCH_SIZE"
  --max-epochs "$MAX_EPOCHS"
  --learning-rate "$LEARNING_RATE"
  --scheduler "$SCHEDULER"
  --warmup-steps "$WARMUP_STEPS"
)
if [[ -n "$LEARNING_RATES" ]]; then
  prepare_args+=(--learning-rates "$LEARNING_RATES")
fi
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
python symphony/scripts/prepare_rob125_supervised_probe_config.py "${prepare_args[@]}"

python - <<'PY'
import json, os
manifest = json.load(open(os.environ["ROB125_RUN_MANIFEST"]))
for run in manifest["runs"]:
    if not os.path.exists(run["local_checkpoint"]):
        raise SystemExit(f"missing checkpoint: {run['local_checkpoint']}")
print("checkpoint path check ok")
PY

while IFS=$'\t' read -r config_path checkpoint_dir; do
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
  python - "$checkpoint_dir" <<'PY'
import sys
from pathlib import Path

checkpoint_dir = Path(sys.argv[1])
checkpoints = sorted(
    checkpoint_dir.glob("step_*.pt"),
    key=lambda path: int(path.stem.split("_")[1]),
)
if not checkpoints:
    raise SystemExit(f"no checkpoints found after training: {checkpoint_dir}")
latest = checkpoints[-1]
removed = 0
for path in checkpoints[:-1]:
    path.unlink()
    removed += 1
print(f"checkpoint cleanup: kept {latest}, removed {removed} older checkpoint(s)")
PY
done < <(python - <<'PY'
import json, os
manifest = json.load(open(os.environ["ROB125_RUN_MANIFEST"]))
for run in manifest["runs"]:
    print(f"{run['train_config']}\t{run['checkpoint_dir']}")
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
python symphony/scripts/make_rob125_eval_config.py "${eval_args[@]}"

(
  cd eval
  python eval_manager.py -config "$EVAL_CONFIG"
)

python symphony/scripts/summarize_rob125_results.py \
  --run-manifest "$RUN_MANIFEST" \
  --csv "$RESULT_CSV" \
  --summary-out "$RESULT_SUMMARY"

cat "$RESULT_SUMMARY"
