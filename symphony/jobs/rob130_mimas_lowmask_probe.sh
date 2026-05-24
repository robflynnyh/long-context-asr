#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${ROB130_REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODE="${ROB130_MODE:-full}"
RUN_ID="${ROB130_RUN_ID:-rob130-lowmask-${MODE}-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ROB130_ARTIFACT_ROOT:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-130}"
RUN_DIR="${ROB130_RUN_DIR:-$ARTIFACT_ROOT/$RUN_ID}"
OUT_LOG="${ROB130_OUT_LOG:-$RUN_DIR/${RUN_ID}.out}"
ERR_LOG="${ROB130_ERR_LOG:-$RUN_DIR/${RUN_ID}.err}"
SUMMARY_FILE="${ROB130_SUMMARY_FILE:-$RUN_DIR/${RUN_ID}.summary.txt}"
RESULT_SUMMARY="${ROB130_RESULT_SUMMARY:-$RUN_DIR/OUTCOME.md}"
EXECUTED_SCRIPT="${ROB130_EXECUTED_SCRIPT:-$RUN_DIR/$(basename "$0")}"
CALLBACK_SCRIPT="${ROB130_CALLBACK_SCRIPT:-$RUN_DIR/linear_mimas_callback.py}"
CALLBACK_LIB="${ROB130_CALLBACK_LIB:-$RUN_DIR/linear_job_callback.py}"
LINEAR_KEY_FILE="${ROB130_LINEAR_KEY_FILE:-$ARTIFACT_ROOT/.linear_api_key}"
LINEAR_FALLBACK_KEY_FILE="${ROB130_LINEAR_FALLBACK_KEY_FILE:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-119/.linear_api_key}"
CALLBACK_PYTHON="${ROB130_CALLBACK_PYTHON:-python3}"
WITH_GPU="${ROB130_WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"

TRAIN_UTTERANCE_DIR="${ROB130_TRAIN_UTTERANCE_DIR:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-126/tedlium_train_utterances}"
UTTERANCE_SENTINEL="${ROB130_UTTERANCE_SENTINEL:-$TRAIN_UTTERANCE_DIR/_SUCCESS.clean_stm_target_v2.json}"
TEDLIUM_ROOT="${ROB130_TEDLIUM_ROOT:-/store/store4/data/TEDLIUM_release1/legacy}"
CACHE_CONTRACT="${ROB130_CACHE_CONTRACT:-$RUN_DIR/cache_contract.json}"
CHECKPOINT_CACHE="${ROB130_CHECKPOINT_CACHE:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-91/source-checkpoints}"
SOURCE_CHECKPOINT_ROOT="${ROB130_SOURCE_CHECKPOINT_ROOT:-/mnt/parscratch/users/acp21rjf/spotify/bestrq_ssl/6l_2048_1epoch_lr3e4_20260516}"
CHECKPOINT_LABELS="${ROB130_CHECKPOINT_LABELS:-100pct}"
CHECKPOINT_ROOT="${ROB130_CHECKPOINT_ROOT:-$ARTIFACT_ROOT/checkpoints/$RUN_ID}"
RUN_MANIFEST="${ROB130_RUN_MANIFEST:-$RUN_DIR/run_manifest.json}"
EVAL_CONFIG="${ROB130_EVAL_CONFIG:-$RUN_DIR/rob130_tedlium_eval.yaml}"
RESULT_CSV="${ROB130_RESULT_CSV:-$RUN_DIR/tedlium_test_results.csv}"

NUM_WORKERS="${ROB130_NUM_WORKERS:-0}"
PREFETCH="${ROB130_PREFETCH:-1}"
PIN_MEMORY="${ROB130_PIN_MEMORY:-0}"
BATCH_SIZE="${ROB130_BATCH_SIZE:-32}"
SMOKE_BATCH_SIZE="${ROB130_SMOKE_BATCH_SIZE:-2}"
SMOKE_MAX_UTTERANCES="${ROB130_SMOKE_MAX_UTTERANCES:-4}"
MAX_EPOCHS="${ROB130_MAX_EPOCHS:-4}"
LEARNING_RATE="${ROB130_LEARNING_RATE:-1e-3}"
LEARNING_RATES="${ROB130_LEARNING_RATES:-}"
HEADS="${ROB130_HEADS:-random_linear,random_bilstm}"
SCHEDULER="${ROB130_SCHEDULER:-constant}"
WARMUP_STEPS="${ROB130_WARMUP_STEPS:-0}"
# Full checkpoints are ~425 MB for this probe. Keep periodic checkpointing off
# by default; exp/train.py still writes the final checkpoint after training.
SAVE_EVERY_N_STEPS="${ROB130_SAVE_EVERY_N_STEPS:-1000000000}"
SEQ_LEN="${ROB130_SEQ_LEN:-2048}"
DISABLE_WANDB="${ROB130_DISABLE_WANDB:-0}"
SMOKE_ENABLE_WANDB="${ROB130_SMOKE_ENABLE_WANDB:-0}"
GPU_POOL="${ROB130_GPU_POOL:-1,2}"
SMOKE_GPU_POOL="${ROB130_SMOKE_GPU_POOL:-1,2}"
GPU_NUM="${ROB130_GPU_NUM:-1}"
GPU_IDLE_SECONDS="${ROB130_GPU_IDLE_SECONDS:-300}"
GPU_STAGE="${ROB130_GPU_STAGE:-0}"

if [[ "${ROB130_RUN_DIR_EXEC:-0}" != "1" ]]; then
  mkdir -p "$RUN_DIR"
  cp "${BASH_SOURCE[0]}" "$EXECUTED_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_mimas_callback.py" "$CALLBACK_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_job_callback.py" "$CALLBACK_LIB"
  chmod +x "$EXECUTED_SCRIPT"
  export ROB130_RUN_DIR_EXEC=1
  export ROB130_REPO_DIR="$REPO_DIR"
  export ROB130_MODE="$MODE"
  export ROB130_RUN_ID="$RUN_ID"
  export ROB130_ARTIFACT_ROOT="$ARTIFACT_ROOT"
  export ROB130_RUN_DIR="$RUN_DIR"
  export ROB130_OUT_LOG="$OUT_LOG"
  export ROB130_ERR_LOG="$ERR_LOG"
  export ROB130_SUMMARY_FILE="$SUMMARY_FILE"
  export ROB130_RESULT_SUMMARY="$RESULT_SUMMARY"
  export ROB130_EXECUTED_SCRIPT="$EXECUTED_SCRIPT"
  export ROB130_CALLBACK_SCRIPT="$CALLBACK_SCRIPT"
  exec bash "$EXECUTED_SCRIPT" "$@"
fi

mkdir -p "$RUN_DIR" "$CHECKPOINT_ROOT" "$RUN_DIR/tmp" "$RUN_DIR/wandb" "$RUN_DIR/wandb-cache" "$RUN_DIR/xdg-cache"
exec > >(tee -a "$OUT_LOG") 2> >(tee -a "$ERR_LOG" >&2)

on_exit() {
  local status=$?
  set +e
  {
    echo "ended_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_code=${status}"
    echo "run_id=${RUN_ID}"
    echo "mode=${MODE}"
    echo "gpu_stage=${GPU_STAGE}"
    echo "host=$(hostname)"
    echo "repo_dir=${REPO_DIR}"
    echo "commit=$(cd "$REPO_DIR" && git rev-parse HEAD 2>/dev/null)"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "train_utterance_dir=${TRAIN_UTTERANCE_DIR}"
    echo "utterance_sentinel=${UTTERANCE_SENTINEL}"
    echo "checkpoint_cache=${CHECKPOINT_CACHE}"
    echo "checkpoint_labels=${CHECKPOINT_LABELS}"
    echo "run_manifest=${RUN_MANIFEST}"
    echo "result_summary=${RESULT_SUMMARY}"
    echo "result_csv=${RESULT_CSV}"
  } >> "$SUMMARY_FILE"
  if [[ "${ROB130_ENABLE_CALLBACK:-1}" == "1" ]]; then
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
      --issue-id ROB-130
      --state-name Todo
      --job-id "$RUN_ID"
      --job-label "Mimas screen run"
      --exit-code "$status"
      --log-out "$OUT_LOG"
      --log-err "$ERR_LOG"
      --output-path "$RUN_DIR"
      --summary-file "$RESULT_SUMMARY"
      --title "ROB-130 old low-mask corrected-utterance probe finished"
      --artifact-path "$RUN_MANIFEST"
      --artifact-path "$CACHE_CONTRACT"
      --artifact-path "$RESULT_CSV"
      --metadata "Run manifest=$RUN_MANIFEST"
      --metadata "Result CSV=$RESULT_CSV"
      --metadata "Cache contract=$CACHE_CONTRACT"
    )
    if [[ "${ROB130_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
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
  echo "gpu_stage=${GPU_STAGE}"
  echo "host=$(hostname)"
  echo "repo_dir=${REPO_DIR}"
  echo "commit=$(cd "$REPO_DIR" && git rev-parse HEAD)"
  echo "executed_script=${EXECUTED_SCRIPT}"
  echo "callback_script=${CALLBACK_SCRIPT}"
  echo "heads=${HEADS}"
  echo "checkpoint_labels=${CHECKPOINT_LABELS}"
  echo "learning_rate=${LEARNING_RATE}"
  echo "learning_rates=${LEARNING_RATES:-unset}"
  echo "scheduler=${SCHEDULER}"
  echo "max_epochs=${MAX_EPOCHS}"
} > "$SUMMARY_FILE"

if [[ "${ROB130_CALLBACK_ONLY:-0}" == "1" ]]; then
  echo "callback-only dry run requested"
  exit 0
fi

if [[ "$GPU_STAGE" == "1" && "$MODE" != "smoke" && "${CUDA_VISIBLE_DEVICES:-}" != "1" && "${CUDA_VISIBLE_DEVICES:-}" != "2" ]]; then
  echo "ROB-130 runs must run through with-gpu pool 1,2; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 2
fi

cd "$REPO_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LCASR_TEDLIUM_ROOT="$TEDLIUM_ROOT"
export ROB130_RUN_MANIFEST="$RUN_MANIFEST"
export TMPDIR="${TMPDIR:-$RUN_DIR/tmp}"
export WANDB_DIR="${WANDB_DIR:-$RUN_DIR/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$RUN_DIR/wandb-cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$RUN_DIR/wandb-config}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$RUN_DIR/wandb-data}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$RUN_DIR/wandb-artifacts}"
export WANDB_DISABLE_CODE="${WANDB_DISABLE_CODE:-true}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUN_DIR/xdg-cache}"

if [[ "$GPU_STAGE" != "1" ]]; then
  python symphony/scripts/check_rob130_utterance_cache.py \
    --utterance-dir "$TRAIN_UTTERANCE_DIR" \
    --sentinel "$UTTERANCE_SENTINEL" \
    --summary-out "$CACHE_CONTRACT"

  prepare_args=(
    --run-dir "$RUN_DIR"
    --checkpoint-root "$CHECKPOINT_ROOT"
    --train-data-path "$TRAIN_UTTERANCE_DIR"
    --cache-contract "$CACHE_CONTRACT"
    --manifest-out "$RUN_MANIFEST"
    --checkpoint-cache "$CHECKPOINT_CACHE"
    --source-checkpoint-root "$SOURCE_CHECKPOINT_ROOT"
    --checkpoint-labels "$CHECKPOINT_LABELS"
    --heads "$HEADS"
    --batch-size "$BATCH_SIZE"
    --smoke-batch-size "$SMOKE_BATCH_SIZE"
    --max-epochs "$MAX_EPOCHS"
    --learning-rate "$LEARNING_RATE"
    --scheduler "$SCHEDULER"
    --warmup-steps "$WARMUP_STEPS"
    --save-every-n-steps "$SAVE_EVERY_N_STEPS"
    --seq-len "$SEQ_LEN"
  )
  if [[ -n "$LEARNING_RATES" ]]; then
    prepare_args+=(--learning-rates "$LEARNING_RATES")
  fi
  if [[ "$MODE" == "smoke" ]]; then
    prepare_args+=(--smoke --smoke-max-records "$SMOKE_MAX_UTTERANCES")
    if [[ "$SMOKE_ENABLE_WANDB" == "1" ]]; then
      prepare_args+=(--enable-smoke-wandb)
    else
      prepare_args+=(--disable-wandb)
    fi
  elif [[ "$DISABLE_WANDB" == "1" ]]; then
    prepare_args+=(--disable-wandb)
  fi
  python symphony/scripts/prepare_rob130_probe_config.py "${prepare_args[@]}"

  python - <<'PY'
import json
import os

manifest = json.load(open(os.environ["ROB130_RUN_MANIFEST"]))
for run in manifest["runs"]:
    if run["trainable_encoder_layers"]:
        raise SystemExit(f"ROB-130 must keep encoder frozen, got {run['trainable_encoder_layers']}")
    if not os.path.exists(run["local_checkpoint"]):
        raise SystemExit(f"missing local ROB-70 checkpoint: {run['local_checkpoint']}")
    print(f"checkpoint path check ok: {run['label']} {run['local_checkpoint']}")
    print(f"frozen audit expected: trainable_encoder_layers={run['trainable_encoder_layers']}")
PY

  selected_pool="$GPU_POOL"
  if [[ "$MODE" == "smoke" ]]; then
    selected_pool="$SMOKE_GPU_POOL"
  fi
  echo "CPU prep complete; acquiring GPU through with-gpu pool ${selected_pool}."
  export ROB130_GPU_STAGE=1
  ROB130_ENABLE_CALLBACK=0 "$WITH_GPU" "$selected_pool" --num "$GPU_NUM" --idle-seconds "$GPU_IDLE_SECONDS" -- bash "$EXECUTED_SCRIPT"
  exit $?
fi

while IFS=$'\t' read -r config_path checkpoint_dir; do
  train_args=(-config "$config_path" -num_workers "$NUM_WORKERS" -prefetch "$PREFETCH" -reset_step)
  if [[ "$PIN_MEMORY" == "1" ]]; then
    train_args+=(-pin_memory)
  fi
  if [[ "$MODE" == "smoke" ]]; then
    train_args+=(--max_records "$SMOKE_MAX_UTTERANCES" --max_steps "$SMOKE_MAX_UTTERANCES" --disable_wandb)
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

manifest = json.load(open(os.environ["ROB130_RUN_MANIFEST"]))
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
python symphony/scripts/make_rob130_eval_config.py "${eval_args[@]}"

(
  cd eval
  python eval_manager.py -config "$EVAL_CONFIG"
)

python symphony/scripts/summarize_rob130_results.py \
  --run-manifest "$RUN_MANIFEST" \
  --csv "$RESULT_CSV" \
  --summary-out "$RESULT_SUMMARY"

cat "$RESULT_SUMMARY"
