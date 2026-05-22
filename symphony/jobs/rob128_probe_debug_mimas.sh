#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${ROB128_REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
STAGE="${ROB128_STAGE:-ladder}"
MODE="${ROB128_MODE:-full}"
RUN_ID="${ROB128_RUN_ID:-rob128-${STAGE}-${MODE}-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ROB128_ARTIFACT_ROOT:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-128}"
RUN_DIR="${ROB128_RUN_DIR:-$ARTIFACT_ROOT/$RUN_ID}"
OUT_LOG="${ROB128_OUT_LOG:-$RUN_DIR/${RUN_ID}.out}"
ERR_LOG="${ROB128_ERR_LOG:-$RUN_DIR/${RUN_ID}.err}"
SUMMARY_FILE="${ROB128_SUMMARY_FILE:-$RUN_DIR/${RUN_ID}.summary.txt}"
RESULT_SUMMARY="${ROB128_RESULT_SUMMARY:-$RUN_DIR/OUTCOME.md}"
EXECUTED_SCRIPT="${ROB128_EXECUTED_SCRIPT:-$RUN_DIR/$(basename "$0")}"
CALLBACK_SCRIPT="${ROB128_CALLBACK_SCRIPT:-$RUN_DIR/linear_mimas_callback.py}"
CALLBACK_LIB="${ROB128_CALLBACK_LIB:-$RUN_DIR/linear_job_callback.py}"
LINEAR_KEY_FILE="${ROB128_LINEAR_KEY_FILE:-$ARTIFACT_ROOT/.linear_api_key}"
LINEAR_FALLBACK_KEY_FILE="${ROB128_LINEAR_FALLBACK_KEY_FILE:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-119/.linear_api_key}"
CALLBACK_PYTHON="${ROB128_CALLBACK_PYTHON:-python3}"

TEDLIUM_ROOT="${ROB128_TEDLIUM_ROOT:-/store/store4/data/TEDLIUM_release1/legacy}"
SUPERVISED_CHECKPOINT="${ROB128_SUPERVISED_CHECKPOINT:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-81/checkpoints/supervised_floras50_spotifytok_safe_norm_drop_oov_lr1e-4_12ep_nw0/step_323484.pt}"
TRAIN_UTTERANCE_DIR="${ROB128_TRAIN_UTTERANCE_DIR:-$RUN_DIR/tedlium_train_utterances}"
UTTERANCE_SUMMARY="${ROB128_UTTERANCE_SUMMARY:-$RUN_DIR/tedlium_train_utterances.json}"
CHECKPOINT_ROOT="${ROB128_CHECKPOINT_ROOT:-$ARTIFACT_ROOT/checkpoints/$RUN_ID}"
RUN_MANIFEST="${ROB128_RUN_MANIFEST:-$RUN_DIR/run_manifest.json}"
EVAL_DIR="${ROB128_EVAL_DIR:-$RUN_DIR/eval}"
PASS_JSON="${ROB128_PASS_JSON:-$RUN_DIR/overfit_pass.json}"

NUM_WORKERS="${ROB128_NUM_WORKERS:-0}"
PREFETCH="${ROB128_PREFETCH:-1}"
PIN_MEMORY="${ROB128_PIN_MEMORY:-0}"
BATCH_SIZE="${ROB128_BATCH_SIZE:-32}"
SMOKE_BATCH_SIZE="${ROB128_SMOKE_BATCH_SIZE:-2}"
MAX_EPOCHS="${ROB128_MAX_EPOCHS:-80}"
ABLATION_MAX_EPOCHS="${ROB128_ABLATION_MAX_EPOCHS:-20}"
LEARNING_RATE="${ROB128_LEARNING_RATE:-1e-3}"
SCHEDULER="${ROB128_SCHEDULER:-constant}"
WARMUP_STEPS="${ROB128_WARMUP_STEPS:-0}"
SAVE_EVERY_N_STEPS="${ROB128_SAVE_EVERY_N_STEPS:-200}"
SEQ_LEN="${ROB128_SEQ_LEN:-4096}"
OVERFIT_RECORDINGS="${ROB128_OVERFIT_RECORDINGS:-1}"
OVERFIT_MAX_UTTERANCES="${ROB128_OVERFIT_MAX_UTTERANCES:-}"
SMOKE_MAX_UTTERANCES="${ROB128_SMOKE_MAX_UTTERANCES:-4}"
DISABLE_WANDB="${ROB128_DISABLE_WANDB:-1}"
SMOKE_ENABLE_WANDB="${ROB128_SMOKE_ENABLE_WANDB:-0}"
PASS_WER_THRESHOLD="${ROB128_PASS_WER_THRESHOLD:-0.20}"
PASS_BLANK_THRESHOLD="${ROB128_PASS_BLANK_THRESHOLD:-0.60}"

if [[ "${ROB128_RUN_DIR_EXEC:-0}" != "1" ]]; then
  mkdir -p "$RUN_DIR"
  cp "${BASH_SOURCE[0]}" "$EXECUTED_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_mimas_callback.py" "$CALLBACK_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_job_callback.py" "$CALLBACK_LIB"
  chmod +x "$EXECUTED_SCRIPT"
  export ROB128_RUN_DIR_EXEC=1
  export ROB128_REPO_DIR="$REPO_DIR"
  export ROB128_STAGE="$STAGE"
  export ROB128_MODE="$MODE"
  export ROB128_RUN_ID="$RUN_ID"
  export ROB128_ARTIFACT_ROOT="$ARTIFACT_ROOT"
  export ROB128_RUN_DIR="$RUN_DIR"
  export ROB128_OUT_LOG="$OUT_LOG"
  export ROB128_ERR_LOG="$ERR_LOG"
  export ROB128_SUMMARY_FILE="$SUMMARY_FILE"
  export ROB128_RESULT_SUMMARY="$RESULT_SUMMARY"
  export ROB128_EXECUTED_SCRIPT="$EXECUTED_SCRIPT"
  export ROB128_CALLBACK_SCRIPT="$CALLBACK_SCRIPT"
  exec bash "$EXECUTED_SCRIPT" "$@"
fi

mkdir -p "$RUN_DIR" "$CHECKPOINT_ROOT" "$RUN_DIR/tmp" "$RUN_DIR/wandb" "$EVAL_DIR"
exec > >(tee -a "$OUT_LOG") 2> >(tee -a "$ERR_LOG" >&2)

on_exit() {
  local status=$?
  set +e
  {
    echo "ended_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_code=${status}"
    echo "run_id=${RUN_ID}"
    echo "stage=${STAGE}"
    echo "mode=${MODE}"
    echo "host=$(hostname)"
    echo "repo_dir=${REPO_DIR}"
    echo "commit=$(cd "$REPO_DIR" && git rev-parse HEAD 2>/dev/null)"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "supervised_checkpoint=${SUPERVISED_CHECKPOINT}"
    echo "train_utterance_dir=${TRAIN_UTTERANCE_DIR}"
    echo "run_manifest=${RUN_MANIFEST}"
    echo "result_summary=${RESULT_SUMMARY}"
    echo "eval_dir=${EVAL_DIR}"
    echo "checkpoint_root=${CHECKPOINT_ROOT}"
  } >> "$SUMMARY_FILE"
  if [[ "${ROB128_ENABLE_CALLBACK:-1}" == "1" ]]; then
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
      --issue-id ROB-128
      --state-name Todo
      --job-id "$RUN_ID"
      --job-label "Mimas screen run"
      --exit-code "$status"
      --log-out "$OUT_LOG"
      --log-err "$ERR_LOG"
      --output-path "$RUN_DIR"
      --summary-file "$RESULT_SUMMARY"
      --title "ROB-128 supervised-feature CTC probe debug finished"
      --artifact-path "$RUN_MANIFEST"
      --artifact-path "$EVAL_DIR"
      --metadata "Run manifest=$RUN_MANIFEST"
      --metadata "Eval dir=$EVAL_DIR"
    )
    if [[ "${ROB128_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
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
  echo "stage=${STAGE}"
  echo "mode=${MODE}"
  echo "host=$(hostname)"
  echo "repo_dir=${REPO_DIR}"
  echo "commit=$(cd "$REPO_DIR" && git rev-parse HEAD)"
  echo "executed_script=${EXECUTED_SCRIPT}"
  echo "callback_script=${CALLBACK_SCRIPT}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
} > "$SUMMARY_FILE"

if [[ "${ROB128_CALLBACK_ONLY:-0}" == "1" ]]; then
  echo "callback-only dry run requested"
  exit 0
fi

if [[ "$MODE" != "smoke" && "${CUDA_VISIBLE_DEVICES:-}" != "1" && "${CUDA_VISIBLE_DEVICES:-}" != "2" ]]; then
  echo "ROB-128 full runs must run through with-gpu pool 1,2; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 2
fi

cd "$REPO_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LCASR_TEDLIUM_ROOT="$TEDLIUM_ROOT"
export TMPDIR="${TMPDIR:-$RUN_DIR/tmp}"
export WANDB_DIR="${WANDB_DIR:-$RUN_DIR/wandb}"
export WANDB_DISABLE_CODE="${WANDB_DISABLE_CODE:-true}"
export ROB128_RUN_MANIFEST="$RUN_MANIFEST"

utterance_args=(
  --tedlium-root "$TEDLIUM_ROOT"
  --split train
  --output-dir "$TRAIN_UTTERANCE_DIR"
  --summary-out "$UTTERANCE_SUMMARY"
)
if [[ "$MODE" == "smoke" ]]; then
  utterance_args+=(--max-recordings 1 --max-utterances "$SMOKE_MAX_UTTERANCES")
else
  utterance_args+=(--max-recordings "$OVERFIT_RECORDINGS")
  if [[ -n "$OVERFIT_MAX_UTTERANCES" ]]; then
    utterance_args+=(--max-utterances "$OVERFIT_MAX_UTTERANCES")
  fi
fi
python symphony/scripts/prepare_rob128_tedlium_utterances.py "${utterance_args[@]}"

run_suite() {
  local stage="$1"
  local heads="$2"
  local max_epochs="$3"
  local manifest="$4"
  local summary="$5"
  local pass_json="$6"

  prepare_args=(
    --run-dir "$RUN_DIR"
    --checkpoint-root "$CHECKPOINT_ROOT"
    --train-data-path "$TRAIN_UTTERANCE_DIR"
    --utterance-summary "$UTTERANCE_SUMMARY"
    --manifest-out "$manifest"
    --supervised-checkpoint "$SUPERVISED_CHECKPOINT"
    --heads "$heads"
    --batch-size "$BATCH_SIZE"
    --smoke-batch-size "$SMOKE_BATCH_SIZE"
    --max-epochs "$max_epochs"
    --learning-rate "$LEARNING_RATE"
    --scheduler "$SCHEDULER"
    --warmup-steps "$WARMUP_STEPS"
    --save-every-n-steps "$SAVE_EVERY_N_STEPS"
    --seq-len "$SEQ_LEN"
    --stage "$stage"
  )
  if [[ "$DISABLE_WANDB" == "1" ]]; then
    prepare_args+=(--disable-wandb)
  fi
  if [[ "$MODE" == "smoke" ]]; then
    prepare_args+=(--smoke)
    if [[ "$SMOKE_ENABLE_WANDB" == "1" ]]; then
      prepare_args+=(--enable-smoke-wandb)
    fi
  fi
  python symphony/scripts/prepare_rob128_probe_config.py "${prepare_args[@]}"

  while IFS=$'\t' read -r label config_path checkpoint_dir; do
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
checkpoints = sorted(checkpoint_dir.glob("step_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
if not checkpoints:
    raise SystemExit(f"no checkpoints found after training: {checkpoint_dir}")
for path in checkpoints[:-1]:
    path.unlink()
print(f"checkpoint cleanup: kept {checkpoints[-1]}")
PY
    python symphony/scripts/eval_rob128_utterance_probe.py \
      --utterance-dir "$TRAIN_UTTERANCE_DIR" \
      --output-json "$EVAL_DIR/${label}.json" \
      --label "$label" \
      --batch-size "$BATCH_SIZE" \
      --checkpoint-dir "$checkpoint_dir"
  done < <(python - "$manifest" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1]))
for run in manifest["runs"]:
    print(f"{run['label']}\t{run['train_config']}\t{run['checkpoint_dir']}")
PY
)

  python symphony/scripts/eval_rob128_utterance_probe.py \
    --utterance-dir "$TRAIN_UTTERANCE_DIR" \
    --output-json "$EVAL_DIR/source_ctc_eval_only.json" \
    --label source_ctc_eval_only \
    --batch-size "$BATCH_SIZE" \
    --source-ctc-checkpoint "$SUPERVISED_CHECKPOINT"

  python symphony/scripts/summarize_rob128_probe_debug.py \
    --run-manifest "$manifest" \
    --eval-dir "$EVAL_DIR" \
    --summary-out "$summary" \
    --pass-json-out "$pass_json" \
    --pass-wer-threshold "$PASS_WER_THRESHOLD" \
    --pass-blank-threshold "$PASS_BLANK_THRESHOLD"
}

OVERFIT_MANIFEST="$RUN_MANIFEST"
run_suite overfit random_bilstm "$MAX_EPOCHS" "$OVERFIT_MANIFEST" "$RESULT_SUMMARY" "$PASS_JSON"

if [[ "$STAGE" == "ladder" ]]; then
  if python - "$PASS_JSON" <<'PY'
import json
import sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("passed") else 1)
PY
  then
    ABLATION_MANIFEST="$RUN_DIR/ablation_manifest.json"
    ABLATION_SUMMARY="$RUN_DIR/ABLATION.md"
    run_suite ablation source_linear_trainable,random_linear,random_bilstm "$ABLATION_MAX_EPOCHS" "$ABLATION_MANIFEST" "$ABLATION_SUMMARY" "$RUN_DIR/ablation_pass.json"
    {
      echo ""
      echo "## Head/Initialization Ablation"
      echo ""
      cat "$ABLATION_SUMMARY"
    } >> "$RESULT_SUMMARY"
  else
    echo "One-record overfit failed; stopping before ablation." | tee -a "$RESULT_SUMMARY"
  fi
fi

cat "$RESULT_SUMMARY"
