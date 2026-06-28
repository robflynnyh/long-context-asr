#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${ROB319_REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN_ID="${ROB319_RUN_ID:-rob319-eggroll-deletion-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ROB319_ARTIFACT_ROOT:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search}"
RUN_DIR="${ROB319_RUN_DIR:-$ARTIFACT_ROOT/$RUN_ID}"
OUT_LOG="${ROB319_OUT_LOG:-$RUN_DIR/${RUN_ID}.out}"
ERR_LOG="${ROB319_ERR_LOG:-$RUN_DIR/${RUN_ID}.err}"
SUMMARY_FILE="${ROB319_SUMMARY_FILE:-$RUN_DIR/${RUN_ID}.summary.txt}"
EXECUTED_SCRIPT="${ROB319_EXECUTED_SCRIPT:-$RUN_DIR/$(basename "$0")}"
CALLBACK_SCRIPT="${ROB319_CALLBACK_SCRIPT:-$RUN_DIR/linear_job_callback.py}"
CONFIG="${ROB319_CONFIG:-symphony/configs/rob319_eggroll_deletion_search.yaml}"
CHECKPOINT_ROOT="${ROB319_CHECKPOINT_ROOT:-/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb_18l_1024D}"
EARNINGS_ROOT="${ROB319_EARNINGS_ROOT:-/store/store4/data/earnings-22}"
DEVICES="${ROB319_DEVICES:-cuda:0,cuda:1}"
CALLBACK_PYTHON="${ROB319_CALLBACK_PYTHON:-python3}"
LINEAR_KEY_FILE="${ROB319_LINEAR_KEY_FILE:-$ARTIFACT_ROOT/.linear_api_key}"

if [[ "${ROB319_RUN_DIR_EXEC:-0}" != "1" ]]; then
  mkdir -p "$RUN_DIR"
  cp "${BASH_SOURCE[0]}" "$EXECUTED_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_job_callback.py" "$CALLBACK_SCRIPT"
  cp "$REPO_DIR/$CONFIG" "$RUN_DIR/$(basename "$CONFIG")"
  chmod +x "$EXECUTED_SCRIPT"
  export ROB319_RUN_DIR_EXEC=1
  export ROB319_REPO_DIR="$REPO_DIR"
  export ROB319_RUN_ID="$RUN_ID"
  export ROB319_ARTIFACT_ROOT="$ARTIFACT_ROOT"
  export ROB319_RUN_DIR="$RUN_DIR"
  export ROB319_OUT_LOG="$OUT_LOG"
  export ROB319_ERR_LOG="$ERR_LOG"
  export ROB319_SUMMARY_FILE="$SUMMARY_FILE"
  export ROB319_EXECUTED_SCRIPT="$EXECUTED_SCRIPT"
  export ROB319_CALLBACK_SCRIPT="$CALLBACK_SCRIPT"
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
    echo "run_id=${RUN_ID}"
    echo "host=$(hostname)"
    echo "repo_dir=${REPO_DIR}"
    echo "branch=$(cd "$REPO_DIR" && git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    echo "commit=$(cd "$REPO_DIR" && git rev-parse HEAD 2>/dev/null)"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "config=${CONFIG}"
    echo "checkpoint_root=${CHECKPOINT_ROOT}"
    echo "earnings_root=${EARNINGS_ROOT}"
    echo "artifact_root=${ARTIFACT_ROOT}"
    echo "run_dir=${RUN_DIR}"
    echo "devices=${DEVICES}"
    echo "extra_pythonpath=${ROB319_EXTRA_PYTHONPATH:-unset}"
    echo "num_pairs=${ROB319_NUM_PAIRS:-config_default}"
    echo "rank=${ROB319_RANK:-config_default}"
    echo "sigma=${ROB319_SIGMA:-config_default}"
    echo "eta=${ROB319_ETA:-config_default}"
    echo "evaluation_mode=${ROB319_EVALUATION_MODE:-config_default}"
    echo "windowed_decode_strategy=${ROB319_WINDOWED_DECODE_STRATEGY:-config_default}"
    echo "autocast_dtype=${ROB319_AUTOCAST_DTYPE:-config_default}"
    echo "max_audio_frames=${ROB319_MAX_AUDIO_FRAMES:-config_default}"
    echo "max_search_blocks=${ROB319_MAX_SEARCH_BLOCKS:-all_configured}"
    echo "max_validation_blocks=${ROB319_MAX_VALIDATION_BLOCKS:-all_configured}"
    echo "wandb_mode=${ROB319_WANDB_MODE:-config_default}"
  } >> "$SUMMARY_FILE"
  if [[ "${ROB319_ENABLE_CALLBACK:-1}" == "1" ]]; then
    if [[ -z "${LINEAR_API_KEY:-}" && -f "$LINEAR_KEY_FILE" ]]; then
      export LINEAR_API_KEY
      LINEAR_API_KEY="$(cat "$LINEAR_KEY_FILE")"
    fi
    callback_args=(
      "$CALLBACK_SCRIPT"
      --mode mimas
      --issue-id ROB-319
      --state-name Todo
      --job-id "$RUN_ID"
      --job-label "Mimas screen session"
      --exit-code "$status"
      --log-out "$OUT_LOG"
      --log-err "$ERR_LOG"
      --artifact-path "$RUN_DIR"
      --output-path "$RUN_DIR"
      --summary-file "$SUMMARY_FILE"
      --title "ROB-319 Mimas EGGROLL deletion search finished"
      --metadata "Config=$CONFIG"
      --metadata "Checkpoint root=$CHECKPOINT_ROOT"
      --metadata "Earnings root=$EARNINGS_ROOT"
      --metadata "Artifact root=$ARTIFACT_ROOT"
    )
    if [[ "${ROB319_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
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
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
} > "$SUMMARY_FILE"

if [[ "${ROB319_CALLBACK_ONLY:-0}" == "1" ]]; then
  echo "callback-only dry run requested"
  exit 0
fi

cd "$REPO_DIR"
if [[ -n "${ROB319_EXTRA_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${ROB319_EXTRA_PYTHONPATH}:$PWD${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
fi

runner_args=(
  --config "$CONFIG"
  --run-id "$RUN_ID"
  --artifact-root "$ARTIFACT_ROOT"
  --checkpoint-root "$CHECKPOINT_ROOT"
  --earnings-root "$EARNINGS_ROOT"
  --devices "$DEVICES"
)

if [[ -n "${ROB319_NUM_PAIRS:-}" ]]; then
  runner_args+=(--num-pairs "$ROB319_NUM_PAIRS")
fi
if [[ -n "${ROB319_RANK:-}" ]]; then
  runner_args+=(--rank "$ROB319_RANK")
fi
if [[ -n "${ROB319_SIGMA:-}" ]]; then
  runner_args+=(--sigma "$ROB319_SIGMA")
fi
if [[ -n "${ROB319_ETA:-}" ]]; then
  runner_args+=(--eta "$ROB319_ETA")
fi
if [[ -n "${ROB319_EVALUATION_MODE:-}" ]]; then
  runner_args+=(--evaluation-mode "$ROB319_EVALUATION_MODE")
fi
if [[ -n "${ROB319_WINDOWED_DECODE_STRATEGY:-}" ]]; then
  runner_args+=(--windowed-decode-strategy "$ROB319_WINDOWED_DECODE_STRATEGY")
fi
if [[ -n "${ROB319_AUTOCAST_DTYPE:-}" ]]; then
  runner_args+=(--autocast-dtype "$ROB319_AUTOCAST_DTYPE")
fi
if [[ -n "${ROB319_MAX_AUDIO_FRAMES:-}" ]]; then
  runner_args+=(--max-audio-frames "$ROB319_MAX_AUDIO_FRAMES")
fi
if [[ -n "${ROB319_MAX_SEARCH_BLOCKS:-}" ]]; then
  runner_args+=(--max-search-blocks "$ROB319_MAX_SEARCH_BLOCKS")
fi
if [[ -n "${ROB319_MAX_VALIDATION_BLOCKS:-}" ]]; then
  runner_args+=(--max-validation-blocks "$ROB319_MAX_VALIDATION_BLOCKS")
fi
if [[ -n "${ROB319_WANDB_MODE:-}" ]]; then
  runner_args+=(--wandb-mode "$ROB319_WANDB_MODE")
fi
if [[ "${ROB319_DISABLE_WANDB:-0}" == "1" ]]; then
  runner_args+=(--disable-wandb)
fi
if [[ "${ROB319_LIST_TARGETS:-0}" == "1" ]]; then
  runner_args+=(--list-targets)
fi
if [[ "${ROB319_DETERMINISM_CHECK:-0}" == "1" ]]; then
  runner_args+=(--determinism-check)
fi
if [[ "${ROB319_RUN_AFTER_LISTING:-0}" == "1" ]]; then
  runner_args+=(--run-after-listing)
fi
if [[ "${ROB319_SKIP_VALIDATION:-0}" == "1" ]]; then
  runner_args+=(--skip-validation)
fi
if [[ "${ROB319_SAVE_COMBINED_CHECKPOINTS:-0}" == "1" ]]; then
  runner_args+=(--save-combined-checkpoints)
fi
if [[ "${ROB319_SMOKE:-0}" == "1" ]]; then
  runner_args+=(
    --num-pairs "${ROB319_SMOKE_NUM_PAIRS:-2}"
    --max-search-blocks "${ROB319_SMOKE_SEARCH_BLOCKS:-1}"
    --max-validation-blocks "${ROB319_SMOKE_VALIDATION_BLOCKS:-1}"
    --evaluation-mode "${ROB319_SMOKE_EVALUATION_MODE:-averaged_moving_window}"
    --windowed-decode-strategy "${ROB319_SMOKE_WINDOWED_DECODE_STRATEGY:-model_context}"
    --max-audio-frames "${ROB319_SMOKE_MAX_AUDIO_FRAMES:-1024}"
  )
fi

echo "runner_args=${runner_args[*]}"
python symphony/scripts/rob319_eggroll_deletion_search.py "${runner_args[@]}"
