#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${ROB126_REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODE="${ROB126_MODE:-full}"
RUN_ID="${ROB126_RUN_ID:-rob126-${MODE}-top${ROB126_UNFREEZE_TOP_N_LAYERS:-2}-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ROB126_ARTIFACT_ROOT:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-126}"
RUN_DIR="${ROB126_RUN_DIR:-$ARTIFACT_ROOT/$RUN_ID}"
OUT_LOG="${ROB126_OUT_LOG:-$RUN_DIR/${RUN_ID}.out}"
ERR_LOG="${ROB126_ERR_LOG:-$RUN_DIR/${RUN_ID}.err}"
SUMMARY_FILE="${ROB126_SUMMARY_FILE:-$RUN_DIR/${RUN_ID}.summary.txt}"
RESULT_SUMMARY="${ROB126_RESULT_SUMMARY:-$RUN_DIR/OUTCOME.md}"
EXECUTED_SCRIPT="${ROB126_EXECUTED_SCRIPT:-$RUN_DIR/$(basename "$0")}"
CALLBACK_SCRIPT="${ROB126_CALLBACK_SCRIPT:-$RUN_DIR/linear_mimas_callback.py}"
CALLBACK_LIB="${ROB126_CALLBACK_LIB:-$RUN_DIR/linear_job_callback.py}"
LINEAR_KEY_FILE="${ROB126_LINEAR_KEY_FILE:-$ARTIFACT_ROOT/.linear_api_key}"
LINEAR_FALLBACK_KEY_FILE="${ROB126_LINEAR_FALLBACK_KEY_FILE:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-119/.linear_api_key}"
CALLBACK_PYTHON="${ROB126_CALLBACK_PYTHON:-python3}"

TEDLIUM_ROOT="${ROB126_TEDLIUM_ROOT:-/store/store4/data/TEDLIUM_release1/legacy}"
CHECKPOINT_CACHE="${ROB126_CHECKPOINT_CACHE:-$ARTIFACT_ROOT/source-checkpoints}"
SOURCE_CHECKPOINT_DIR="${ROB126_SOURCE_CHECKPOINT_DIR:-/mnt/parscratch/users/acp21rjf/spotify/bestrq_ssl/rob100_papermask_p012_l4_sc_off_20260520}"
TRAIN_UTTERANCE_DIR="${ROB126_TRAIN_UTTERANCE_DIR:-$ARTIFACT_ROOT/tedlium_train_utterances}"
if [[ "$MODE" == "smoke" ]]; then
  TRAIN_UTTERANCE_DIR="${ROB126_TRAIN_UTTERANCE_DIR:-$RUN_DIR/tedlium_train_utterances}"
fi
UTTERANCE_SUMMARY="${ROB126_UTTERANCE_SUMMARY:-$RUN_DIR/tedlium_train_utterances.json}"
CHECKPOINT_ROOT="${ROB126_CHECKPOINT_ROOT:-$ARTIFACT_ROOT/checkpoints/$RUN_ID}"
RUN_MANIFEST="${ROB126_RUN_MANIFEST:-$RUN_DIR/run_manifest.json}"
EVAL_CONFIG="${ROB126_EVAL_CONFIG:-$RUN_DIR/rob126_tedlium_eval.yaml}"
RESULT_CSV="${ROB126_RESULT_CSV:-$RUN_DIR/tedlium_test_results.csv}"

NUM_WORKERS="${ROB126_NUM_WORKERS:-0}"
PREFETCH="${ROB126_PREFETCH:-1}"
PIN_MEMORY="${ROB126_PIN_MEMORY:-0}"
BATCH_SIZE="${ROB126_BATCH_SIZE:-32}"
MAX_EPOCHS="${ROB126_MAX_EPOCHS:-4}"
LEARNING_RATE="${ROB126_LEARNING_RATE:-3e-4}"
LEARNING_RATES="${ROB126_LEARNING_RATES:-}"
SCHEDULER="${ROB126_SCHEDULER:-constant}"
WARMUP_STEPS="${ROB126_WARMUP_STEPS:-0}"
CHECKPOINT_LABELS="${ROB126_CHECKPOINT_LABELS:-primary}"
BACKUP_REASON="${ROB126_BACKUP_REASON:-}"
SMOKE_MAX_RECORDINGS="${ROB126_SMOKE_MAX_RECORDINGS:-1}"
SMOKE_MAX_UTTERANCES="${ROB126_SMOKE_MAX_UTTERANCES:-4}"
SMOKE_BATCH_SIZE="${ROB126_SMOKE_BATCH_SIZE:-2}"
FULL_MAX_RECORDINGS="${ROB126_FULL_MAX_RECORDINGS:-}"
FULL_MAX_UTTERANCES="${ROB126_FULL_MAX_UTTERANCES:-}"
DISABLE_WANDB="${ROB126_DISABLE_WANDB:-0}"
SMOKE_ENABLE_WANDB="${ROB126_SMOKE_ENABLE_WANDB:-0}"
UNFREEZE_TOP_N_LAYERS="${ROB126_UNFREEZE_TOP_N_LAYERS:-2}"
ENCODER_LR_SCALE="${ROB126_ENCODER_LR_SCALE:-0.25}"

if [[ "${ROB126_RUN_DIR_EXEC:-0}" != "1" ]]; then
  mkdir -p "$RUN_DIR"
  cp "${BASH_SOURCE[0]}" "$EXECUTED_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_mimas_callback.py" "$CALLBACK_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_job_callback.py" "$CALLBACK_LIB"
  chmod +x "$EXECUTED_SCRIPT"
  export ROB126_RUN_DIR_EXEC=1
  export ROB126_REPO_DIR="$REPO_DIR"
  export ROB126_MODE="$MODE"
  export ROB126_RUN_ID="$RUN_ID"
  export ROB126_ARTIFACT_ROOT="$ARTIFACT_ROOT"
  export ROB126_RUN_DIR="$RUN_DIR"
  export ROB126_OUT_LOG="$OUT_LOG"
  export ROB126_ERR_LOG="$ERR_LOG"
  export ROB126_SUMMARY_FILE="$SUMMARY_FILE"
  export ROB126_RESULT_SUMMARY="$RESULT_SUMMARY"
  export ROB126_EXECUTED_SCRIPT="$EXECUTED_SCRIPT"
  export ROB126_CALLBACK_SCRIPT="$CALLBACK_SCRIPT"
  exec bash "$EXECUTED_SCRIPT" "$@"
fi

mkdir -p \
  "$RUN_DIR" \
  "$CHECKPOINT_CACHE" \
  "$CHECKPOINT_ROOT" \
  "$RUN_DIR/tmp" \
  "$RUN_DIR/wandb" \
  "$RUN_DIR/wandb/wandb" \
  "$RUN_DIR/wandb-cache" \
  "$RUN_DIR/wandb-config" \
  "$RUN_DIR/wandb-data" \
  "$RUN_DIR/wandb-artifacts" \
  "$RUN_DIR/xdg-cache" \
  "$TRAIN_UTTERANCE_DIR"
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
    echo "train_utterance_dir=${TRAIN_UTTERANCE_DIR}"
    echo "utterance_summary=${UTTERANCE_SUMMARY}"
    echo "run_manifest=${RUN_MANIFEST}"
    echo "result_csv=${RESULT_CSV}"
    echo "result_summary=${RESULT_SUMMARY}"
    echo "checkpoint_root=${CHECKPOINT_ROOT}"
    echo "learning_rate=${LEARNING_RATE}"
    echo "learning_rates=${LEARNING_RATES:-unset}"
    echo "scheduler=${SCHEDULER}"
    echo "warmup_steps=${WARMUP_STEPS}"
    echo "max_epochs=${MAX_EPOCHS}"
    echo "checkpoint_labels=${CHECKPOINT_LABELS}"
    echo "unfreeze_top_n_layers=${UNFREEZE_TOP_N_LAYERS}"
    echo "encoder_lr_scale=${ENCODER_LR_SCALE}"
  } >> "$SUMMARY_FILE"
  if [[ "${ROB126_ENABLE_CALLBACK:-1}" == "1" ]]; then
    if [[ -z "${LINEAR_API_KEY:-}" ]]; then
      if [[ -f "$LINEAR_KEY_FILE" ]]; then
        export LINEAR_API_KEY
        LINEAR_API_KEY="$(cat "$LINEAR_KEY_FILE")"
      elif [[ -f "$LINEAR_FALLBACK_KEY_FILE" ]]; then
        export LINEAR_API_KEY
        LINEAR_API_KEY="$(cat "$LINEAR_FALLBACK_KEY_FILE")"
      fi
    fi
    callback_args=(
      --issue-id ROB-126
      --state-name Todo
      --job-id "$RUN_ID"
      --job-label "Mimas screen run"
      --exit-code "$status"
      --log-out "$OUT_LOG"
      --log-err "$ERR_LOG"
      --output-path "$RUN_DIR"
      --summary-file "$RESULT_SUMMARY"
      --title "ROB-126 ROB-100 top-layer unfrozen CTC probe finished"
    )
    if [[ "${ROB126_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
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
  echo "train_utterance_dir=${TRAIN_UTTERANCE_DIR}"
  echo "utterance_summary=${UTTERANCE_SUMMARY}"
  echo "learning_rate=${LEARNING_RATE}"
  echo "learning_rates=${LEARNING_RATES:-unset}"
  echo "scheduler=${SCHEDULER}"
  echo "warmup_steps=${WARMUP_STEPS}"
  echo "max_epochs=${MAX_EPOCHS}"
  echo "checkpoint_labels=${CHECKPOINT_LABELS}"
  echo "unfreeze_top_n_layers=${UNFREEZE_TOP_N_LAYERS}"
  echo "encoder_lr_scale=${ENCODER_LR_SCALE}"
} > "$SUMMARY_FILE"

if [[ "${ROB126_CALLBACK_ONLY:-0}" == "1" ]]; then
  echo "callback-only dry run requested"
  exit 0
fi

if [[ "$MODE" != "smoke" && "${CUDA_VISIBLE_DEVICES:-}" != "1" && "${CUDA_VISIBLE_DEVICES:-}" != "2" ]]; then
  echo "ROB-126 full runs must run through with-gpu pool 1,2; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 2
fi

cd "$REPO_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LCASR_TEDLIUM_ROOT="$TEDLIUM_ROOT"
export ROB126_RUN_MANIFEST="$RUN_MANIFEST"
export TMPDIR="${TMPDIR:-$RUN_DIR/tmp}"
export WANDB_DIR="${WANDB_DIR:-$RUN_DIR/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$RUN_DIR/wandb-cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$RUN_DIR/wandb-config}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$RUN_DIR/wandb-data}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$RUN_DIR/wandb-artifacts}"
export WANDB_DISABLE_CODE="${WANDB_DISABLE_CODE:-true}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUN_DIR/xdg-cache}"

IFS=',' read -r -a selected_labels <<< "$CHECKPOINT_LABELS"
selected_checkpoints=()
for label in "${selected_labels[@]}"; do
  label="${label//[[:space:]]/}"
  case "$label" in
    primary)
      selected_checkpoints+=("step_105360.pt")
      ;;
    backup)
      if [[ -z "$BACKUP_REASON" ]]; then
        echo "ROB126_BACKUP_REASON is required when CHECKPOINT_LABELS includes backup" >&2
        exit 2
      fi
      selected_checkpoints+=("step_99264.pt")
      ;;
    *)
      echo "unknown ROB126_CHECKPOINT_LABELS entry: $label" >&2
      exit 2
      ;;
  esac
done

for checkpoint in "${selected_checkpoints[@]}"; do
  if [[ ! -s "$CHECKPOINT_CACHE/$checkpoint" ]]; then
    if [[ -s "/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-119/source-checkpoints/$checkpoint" ]]; then
      cp "/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-119/source-checkpoints/$checkpoint" "$CHECKPOINT_CACHE/"
    else
      rsync -av "acp21rjf@stanage.shef.ac.uk:${SOURCE_CHECKPOINT_DIR}/${checkpoint}" "$CHECKPOINT_CACHE/"
    fi
  fi
done

utterance_args=(
  --tedlium-root "$TEDLIUM_ROOT"
  --split train
  --output-dir "$TRAIN_UTTERANCE_DIR"
  --summary-out "$UTTERANCE_SUMMARY"
)
if [[ "$MODE" == "smoke" ]]; then
  utterance_args+=(--max-recordings "$SMOKE_MAX_RECORDINGS" --max-utterances "$SMOKE_MAX_UTTERANCES")
else
  if [[ -n "$FULL_MAX_RECORDINGS" ]]; then
    utterance_args+=(--max-recordings "$FULL_MAX_RECORDINGS")
  fi
  if [[ -n "$FULL_MAX_UTTERANCES" ]]; then
    utterance_args+=(--max-utterances "$FULL_MAX_UTTERANCES")
  fi
fi
python symphony/scripts/prepare_rob126_tedlium_utterances.py "${utterance_args[@]}"

prepare_args=(
  --run-dir "$RUN_DIR"
  --checkpoint-cache "$CHECKPOINT_CACHE"
  --checkpoint-root "$CHECKPOINT_ROOT"
  --train-data-path "$TRAIN_UTTERANCE_DIR"
  --train-data-format utterance_folder
  --manifest-out "$RUN_MANIFEST"
  --source-checkpoint-dir "$SOURCE_CHECKPOINT_DIR"
  --batch-size "$BATCH_SIZE"
  --smoke-batch-size "$SMOKE_BATCH_SIZE"
  --max-epochs "$MAX_EPOCHS"
  --learning-rate "$LEARNING_RATE"
  --scheduler "$SCHEDULER"
  --warmup-steps "$WARMUP_STEPS"
  --checkpoint-labels "$CHECKPOINT_LABELS"
  --unfreeze-top-n-layers "$UNFREEZE_TOP_N_LAYERS"
  --encoder-lr-scale "$ENCODER_LR_SCALE"
)
if [[ -n "$BACKUP_REASON" ]]; then
  prepare_args+=(--backup-reason "$BACKUP_REASON")
fi
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
python symphony/scripts/prepare_rob126_probe_config.py "${prepare_args[@]}"

python - <<'PY'
import json
import os

manifest = json.load(open(os.environ["ROB126_RUN_MANIFEST"]))
for run in manifest["runs"]:
    if not os.path.exists(run["local_checkpoint"]):
        raise SystemExit(f"missing checkpoint: {run['local_checkpoint']}")
    print(f"checkpoint path check ok: {run['label']} {run['local_checkpoint']}")
    print(f"trainable_encoder_layers: {run['trainable_encoder_layers']}")
PY

while IFS=$'\t' read -r config_path checkpoint_dir; do
  train_args=(-config "$config_path" -num_workers "$NUM_WORKERS" -prefetch "$PREFETCH" -reset_step)
  if [[ "$PIN_MEMORY" == "1" ]]; then
    train_args+=(-pin_memory)
  fi
  if [[ "$MODE" == "smoke" ]]; then
    train_args+=(--max_records "$SMOKE_MAX_UTTERANCES" --max_steps "$SMOKE_MAX_UTTERANCES")
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
import json
import os

manifest = json.load(open(os.environ["ROB126_RUN_MANIFEST"]))
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
python symphony/scripts/make_rob126_eval_config.py "${eval_args[@]}"

(
  cd eval
  python eval_manager.py -config "$EVAL_CONFIG"
)

python symphony/scripts/summarize_rob126_results.py \
  --run-manifest "$RUN_MANIFEST" \
  --csv "$RESULT_CSV" \
  --summary-out "$RESULT_SUMMARY"

cat "$RESULT_SUMMARY"
