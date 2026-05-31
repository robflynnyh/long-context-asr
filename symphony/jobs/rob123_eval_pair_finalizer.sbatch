#!/bin/bash
#SBATCH --job-name=rob123-eval-final
#SBATCH --partition=interactive
#SBATCH --time=00:20:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval/rob123-eval-final-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval/rob123-eval-final-%j.err

set -Eeuo pipefail

REPO_DIR="${ROB123_REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-123}"
REMOTE_BRANCH="${ROB123_REMOTE_BRANCH:-symphony/ROB-123-rope-full-spotify-stanage}"
TARGET_COMMIT="${ROB123_TARGET_COMMIT:?ROB123_TARGET_COMMIT must pin the code revision}"
ARTIFACT_ROOT="${ROB123_ARTIFACT_ROOT:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval}"
RUN_ID="${ROB123_RUN_ID:-rob123-cont2-eval-finalizer-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${ROB123_RUN_DIR:-$ARTIFACT_ROOT/$RUN_ID}"
OUT_LOG="${ROB123_OUT_LOG:-$ARTIFACT_ROOT/rob123-eval-final-${SLURM_JOB_ID:-manual}.out}"
ERR_LOG="${ROB123_ERR_LOG:-$ARTIFACT_ROOT/rob123-eval-final-${SLURM_JOB_ID:-manual}.err}"
SUMMARY_FILE="${ROB123_SUMMARY_FILE:-$RUN_DIR/${RUN_ID}.summary.txt}"
EXECUTED_SCRIPT="${ROB123_EXECUTED_SCRIPT:-$RUN_DIR/$(basename "$0")}"
CALLBACK_SCRIPT="${ROB123_CALLBACK_SCRIPT:-$RUN_DIR/linear_stanage_callback.py}"
CALLBACK_PYTHON="${ROB123_CALLBACK_PYTHON:-python3}"
LINEAR_KEY_FILE="${ROB123_LINEAR_KEY_FILE:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/.linear_api_key}"

CHECKPOINT="${ROB123_CHECKPOINT:?ROB123_CHECKPOINT must point at the evaluated checkpoint}"
TED_JOB_ID="${ROB123_TED_JOB_ID:?ROB123_TED_JOB_ID is required}"
EARN_JOB_ID="${ROB123_EARN_JOB_ID:?ROB123_EARN_JOB_ID is required}"
TED_RUN_ID="${ROB123_TED_RUN_ID:?ROB123_TED_RUN_ID is required}"
EARN_RUN_ID="${ROB123_EARN_RUN_ID:?ROB123_EARN_RUN_ID is required}"
TED_SUMMARY="${ROB123_TED_SUMMARY:-$ARTIFACT_ROOT/$TED_RUN_ID/${TED_RUN_ID}.summary.txt}"
EARN_SUMMARY="${ROB123_EARN_SUMMARY:-$ARTIFACT_ROOT/$EARN_RUN_ID/${EARN_RUN_ID}.summary.txt}"
TED_RESULT_CSV="${ROB123_TED_RESULT_CSV:-$ARTIFACT_ROOT/$TED_RUN_ID/rob123_tedlium_eval.csv}"
EARN_RESULT_CSV="${ROB123_EARN_RESULT_CSV:-$ARTIFACT_ROOT/$EARN_RUN_ID/rob123_earnings22_eval.csv}"

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
  export ROB123_EXECUTED_SCRIPT="$EXECUTED_SCRIPT"
  export ROB123_CALLBACK_SCRIPT="$CALLBACK_SCRIPT"
  exec bash "$EXECUTED_SCRIPT" "$@"
fi

mkdir -p "$RUN_DIR"

job_status() {
  sacct -j "$1" --format=JobID,State,ExitCode,Elapsed -P -n | awk -F'|' -v id="$1" '$1 == id { print $2 "|" $3 "|" $4; exit }'
}

append_summary_tail() {
  local label="$1"
  local path="$2"
  {
    echo "${label}_summary=${path}"
    if [[ -f "$path" ]]; then
      echo "${label}_summary_tail_begin"
      tail -80 "$path"
      echo "${label}_summary_tail_end"
    else
      echo "${label}_summary_missing=1"
    fi
  } >> "$SUMMARY_FILE"
}

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
    echo "ted_job_id=${TED_JOB_ID}"
    echo "earn_job_id=${EARN_JOB_ID}"
    echo "ted_result_csv=${TED_RESULT_CSV}"
    echo "earnings22_result_csv=${EARN_RESULT_CSV}"
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
      --artifact-path "$ARTIFACT_ROOT/$TED_RUN_ID"
      --artifact-path "$ARTIFACT_ROOT/$EARN_RUN_ID"
      --checkpoint-path "$CHECKPOINT"
      --summary-file "$SUMMARY_FILE"
      --title "ROB-123 post-continuation eval pair finished"
      --job-label "Stanage eval finalizer"
      --metadata "branch=${REMOTE_BRANCH}"
      --metadata "commit=${TARGET_COMMIT}"
      --metadata "checkpoint=${CHECKPOINT}"
      --metadata "tedlium_job=${TED_JOB_ID}"
      --metadata "earnings22_job=${EARN_JOB_ID}"
      --metadata "tedlium_result_csv=${TED_RESULT_CSV}"
      --metadata "earnings22_result_csv=${EARN_RESULT_CSV}"
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
  echo "checkpoint=${CHECKPOINT}"
  echo "ted_job_id=${TED_JOB_ID}"
  echo "earn_job_id=${EARN_JOB_ID}"
  echo "ted_summary=${TED_SUMMARY}"
  echo "earnings22_summary=${EARN_SUMMARY}"
} > "$SUMMARY_FILE"

if [[ "${ROB123_CALLBACK_ONLY:-0}" == "1" ]]; then
  echo "callback-only dry run requested"
  exit 0
fi

ted_status="$(job_status "$TED_JOB_ID")"
earn_status="$(job_status "$EARN_JOB_ID")"
{
  echo "ted_job_status=${ted_status}"
  echo "earnings22_job_status=${earn_status}"
} >> "$SUMMARY_FILE"

append_summary_tail "tedlium" "$TED_SUMMARY"
append_summary_tail "earnings22" "$EARN_SUMMARY"

fail=0
[[ "$ted_status" == COMPLETED\|0:0\|* ]] || fail=1
[[ "$earn_status" == COMPLETED\|0:0\|* ]] || fail=1
grep -q '^tedlium_test_wer=' "$TED_SUMMARY" || fail=1
grep -q '^earnings22_test_wer=' "$EARN_SUMMARY" || fail=1
[[ -f "$TED_RESULT_CSV" ]] || fail=1
[[ -f "$EARN_RESULT_CSV" ]] || fail=1

if [[ "$fail" != "0" ]]; then
  echo "eval_pair_status=failure" >> "$SUMMARY_FILE"
  exit 1
fi

echo "eval_pair_status=success" >> "$SUMMARY_FILE"
