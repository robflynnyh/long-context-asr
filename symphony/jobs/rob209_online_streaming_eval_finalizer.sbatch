#!/bin/bash
#SBATCH --job-name=rob209-online-finalize
#SBATCH --partition=sheffield
#SBATCH --time=00:10:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-209/rob209-online-finalizer-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-209/rob209-online-finalizer-%j.err

set -Eeuo pipefail

REPO_DIR="${ROB209_REPO_DIR:-/users/acp21rjf/long-context-asr}"
REMOTE_BRANCH="${ROB209_REMOTE_BRANCH:-symphony/ROB-209-ordered-floras-bounded-history}"
TARGET_COMMIT="${ROB209_TARGET_COMMIT:-}"
EVAL_JOB_ID="${ROB209_EVAL_JOB_ID:?ROB209_EVAL_JOB_ID must be exported}"
RUN_ID="${ROB209_EVAL_RUN_ID:?ROB209_EVAL_RUN_ID must be exported}"
ARTIFACT_ROOT="${ROB209_ARTIFACT_ROOT:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-209}"
RUN_DIR="${ROB209_EVAL_RUN_DIR:-$ARTIFACT_ROOT/$RUN_ID}"
LOG_PREFIX="${ROB209_EVAL_LOG_PREFIX:-rob209-online-eval}"
OUT_LOG="${ROB209_EVAL_OUT_LOG:-$ARTIFACT_ROOT/${LOG_PREFIX}-${EVAL_JOB_ID}.out}"
ERR_LOG="${ROB209_EVAL_ERR_LOG:-$ARTIFACT_ROOT/${LOG_PREFIX}-${EVAL_JOB_ID}.err}"
SUMMARY_FILE="${ROB209_EVAL_SUMMARY_FILE:-$RUN_DIR/${RUN_ID}.summary.txt}"
EVAL_CONFIG="${ROB209_EVAL_CONFIG:-$RUN_DIR/rob209_online_streaming_eval.yaml}"
RESULT_CSV="${ROB209_EVAL_RESULT_CSV:-$RUN_DIR/rob209_online_streaming_eval.csv}"
CHECKPOINT="${ROB209_EVAL_CHECKPOINT:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-209/checkpoints/rob209-ordered-floras-debuggen-20260608T2215Z/step_36425.pt}"
CHECKPOINT_STEP="${ROB209_EVAL_CHECKPOINT_STEP:-}"
DATASET="${ROB209_EVAL_DATASET:-tedlium}"
SPLIT="${ROB209_EVAL_SPLIT:-test}"
FLORAS50_ROOT="${ROB209_FLORAS50_ROOT:-/mnt/parscratch/users/acp21rjf/floras50_eval}"
TEDLIUM_ROOT="${ROB209_TEDLIUM_ROOT:-/mnt/parscratch/users/acp21rjf/TEDLIUM_release1}"
DECODE_MODE="${ROB209_EVAL_DECODE_MODE:-greedy}"
CHUNK_SIZE="${ROB209_EVAL_CHUNK_SIZE:-2048}"
SUBSAMPLING_HISTORY_FRAMES="${ROB209_EVAL_SUBSAMPLING_HISTORY_FRAMES:-2048}"
DECODER_HISTORY_FRAMES="${ROB209_EVAL_DECODER_HISTORY_FRAMES:-2048}"
FINAL_FLUSH_FRAMES="${ROB209_EVAL_FINAL_FLUSH_FRAMES:-75}"
MAX_SEQUENCE_LENGTH="${ROB209_EVAL_MAX_SEQUENCE_LENGTH:--1}"
MAX_OUTPUT_FRAMES="${ROB209_EVAL_MAX_OUTPUT_FRAMES:-0}"
BREAK_EVAL="${ROB209_EVAL_BREAK:-0}"
CALLBACK_PYTHON="${ROB209_CALLBACK_PYTHON:-python3}"
LINEAR_KEY_FILE="${ROB209_LINEAR_KEY_FILE:-$ARTIFACT_ROOT/.linear_api_key}"
LINEAR_ENV_FILE="${ROB209_LINEAR_ENV_FILE:-}"

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
        return 0
      fi
    fi
  done
  return 1
}

cd "$REPO_DIR"
if [[ -z "$TARGET_COMMIT" ]]; then
  TARGET_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
fi
if [[ -z "$CHECKPOINT_STEP" ]]; then
  checkpoint_name="$(basename "$CHECKPOINT")"
  CHECKPOINT_STEP="${checkpoint_name#step_}"
  CHECKPOINT_STEP="${CHECKPOINT_STEP%.pt}"
fi

mkdir -p "$RUN_DIR"
{
  echo "finalizer_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "finalizer_slurm_job_id=${SLURM_JOB_ID:-manual}"
  echo "eval_job_id=${EVAL_JOB_ID}"
  echo "run_id=${RUN_ID}"
  echo "repo_dir=${REPO_DIR}"
  echo "target_commit=${TARGET_COMMIT}"
} >> "$SUMMARY_FILE"

if [[ -f "$SUMMARY_FILE" ]] && grep -q '^callback_exit_code=0$' "$SUMMARY_FILE"; then
  echo "wrapper_callback_already_succeeded=1" >> "$SUMMARY_FILE"
  echo "Wrapper callback already succeeded; finalizer exiting without posting."
  exit 0
fi

SACCT_LINE="$(
  sacct -j "$EVAL_JOB_ID" --format=JobID,State,ExitCode,Elapsed -P -n \
    | awk -F'|' -v id="$EVAL_JOB_ID" '$1 == id { print; exit }'
)"
SLURM_STATE="$(printf '%s' "$SACCT_LINE" | awk -F'|' '{print $2}')"
SLURM_EXIT="$(printf '%s' "$SACCT_LINE" | awk -F'|' '{print $3}')"
if [[ -z "$SLURM_STATE" ]]; then
  SLURM_STATE="UNKNOWN"
fi
if [[ "$SLURM_STATE" == COMPLETED && "$SLURM_EXIT" == 0:0 ]]; then
  STATUS="success"
  EXIT_CODE="0"
else
  STATUS="failure"
  EXIT_CODE="${SLURM_EXIT%%:*}"
  if [[ -z "$EXIT_CODE" || "$EXIT_CODE" == "0" ]]; then
    EXIT_CODE="1"
  fi
fi

{
  echo "finalizer_sacct=${SACCT_LINE:-missing}"
  echo "finalizer_status=${STATUS}"
  echo "finalizer_exit_code=${EXIT_CODE}"
} >> "$SUMMARY_FILE"

load_linear_api_key || true
callback_args=(
  "$REPO_DIR/symphony/scripts/linear_stanage_callback.py"
  --issue-id ROB-209
  --state-name Todo
  --slurm-job-id "$EVAL_JOB_ID"
  --job-id "$EVAL_JOB_ID"
  --exit-code "$EXIT_CODE"
  --status "$STATUS"
  --log-out "$OUT_LOG"
  --log-err "$ERR_LOG"
  --artifact-path "$RUN_DIR"
  --summary-file "$SUMMARY_FILE"
  --title "ROB-209 Stanage online streaming eval finalizer"
  --job-label "Stanage online streaming eval"
  --summary-text "Finalizer posted because the wrapper callback did not record callback_exit_code=0. sacct=${SACCT_LINE:-missing}"
  --metadata "branch=${REMOTE_BRANCH}"
  --metadata "commit=${TARGET_COMMIT}"
  --metadata "checkpoint=${CHECKPOINT}"
  --metadata "checkpoint_step=${CHECKPOINT_STEP}"
  --metadata "dataset=${DATASET}"
  --metadata "split=${SPLIT}"
  --metadata "floras50_root=${FLORAS50_ROOT}"
  --metadata "tedlium_root=${TEDLIUM_ROOT}"
  --metadata "eval_config=${EVAL_CONFIG}"
  --metadata "result_csv=${RESULT_CSV}"
  --metadata "decode_mode=${DECODE_MODE}"
  --metadata "chunked_kv_cache=true"
  --metadata "chunk_size=${CHUNK_SIZE}"
  --metadata "decoder_history_frames=${DECODER_HISTORY_FRAMES}"
  --metadata "subsampling_history_frames=${SUBSAMPLING_HISTORY_FRAMES}"
  --metadata "final_flush_frames=${FINAL_FLUSH_FRAMES}"
  --metadata "max_sequence_length=${MAX_SEQUENCE_LENGTH}"
  --metadata "max_output_frames=${MAX_OUTPUT_FRAMES}"
  --metadata "break_eval=${BREAK_EVAL}"
)
if [[ "${ROB209_EVAL_FINALIZER_DRY_RUN:-0}" == "1" ]]; then
  callback_args+=(--dry-run)
fi

"$CALLBACK_PYTHON" "${callback_args[@]}" >> "$SUMMARY_FILE" 2>&1
callback_status=$?
echo "finalizer_callback_exit_code=${callback_status}" >> "$SUMMARY_FILE"
exit "$callback_status"
