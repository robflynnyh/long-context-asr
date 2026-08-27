#!/bin/bash
#SBATCH --job-name=rob327-ctc-rev16
#SBATCH --time=24:00:00
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=16
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-327/slurm/%x-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-327/slurm/%x-%j.err

set -euo pipefail

ISSUE_ID="ROB-327"
ARTIFACT_ROOT="/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/${ISSUE_ID}"
REPO_DIR="${ARTIFACT_ROOT}/code/long-context-asr-ctc"
CHECKPOINT="${ROB327_CHECKPOINT:-/mnt/parscratch/users/acp21rjf/floras50/windowed/checkpoints_seq_scheduler_rb/f50_w2048_1e3_rp_1/step_182892.pt}"
RUN_KIND="${ROB327_RUN_KIND:-full}"
RUN_LABEL="${ROB327_RUN_LABEL:-${RUN_KIND}}"
REPO_BRANCH="${ROB327_REPO_BRANCH:-symphony/ROB-327-floras-rev16-failure}"
REPO_COMMIT="${ROB327_REPO_COMMIT:-unknown}"
REV16_SPLIT="${ROB327_SPLIT:-test}"
EVAL_MODE="${ROB327_EVAL_MODE:-buffered}"
SEQ_LEN="${ROB327_SEQ_LEN:-32768}"
OVERLAP="${ROB327_OVERLAP:-28672}"
WINDOW_SIZE="${ROB327_WINDOW_SIZE:-2048}"
OVERRIDE_WINDOW_SIZE="${ROB327_OVERRIDE_WINDOW_SIZE:--1}"
MAX_SEQUENCE_LENGTH="${ROB327_MAX_SEQUENCE_LENGTH:-3600000}"
RUN_DIR="${ARTIFACT_ROOT}/runs/ctc-${RUN_LABEL}-${SLURM_JOB_ID:-manual}"
STDOUT_LOG="${ARTIFACT_ROOT}/slurm/${SLURM_JOB_NAME:-rob327-ctc-rev16}-${SLURM_JOB_ID:-manual}.out"
STDERR_LOG="${ARTIFACT_ROOT}/slurm/${SLURM_JOB_NAME:-rob327-ctc-rev16}-${SLURM_JOB_ID:-manual}.err"
EVAL_LOG="${RUN_DIR}/eval.log"
SUMMARY_FILE="${RUN_DIR}/summary.json"
MAX_RECORDINGS="${ROB327_MAX_RECORDINGS:-}"

mkdir -p "$RUN_DIR" "$RUN_DIR/tmp" "$RUN_DIR/cache" "$RUN_DIR/matplotlib"

load_linear_env() {
  for env_file in \
    "${ARTIFACT_ROOT}/linear.env" \
    "${HOME}/.config/long-context-asr/linear.env" \
    "${HOME}/.config/symphony/linear.env" \
    "${HOME}/.linear.env"
  do
    if [ -f "$env_file" ]; then
      set -a
      # shellcheck disable=SC1090
      . "$env_file"
      set +a
      return 0
    fi
  done
  return 0
}

post_callback() {
  status="$1"
  load_linear_env
  if [ "${CALLBACK_DRY_RUN:-0}" = "1" ]; then
    python "$REPO_DIR/symphony/scripts/linear_stanage_callback.py" \
      --issue-id "$ISSUE_ID" \
      --state-name Todo \
      --exit-code "$status" \
      --log-out "$STDOUT_LOG" \
      --log-err "$STDERR_LOG" \
      --log "$EVAL_LOG" \
      --artifact-path "$RUN_DIR" \
      --checkpoint-path "$CHECKPOINT" \
      --summary-file "$SUMMARY_FILE" \
      --title "ROB-327 Floras-only CTC Rev16 eval finished" \
      --metadata "run_kind=${RUN_KIND}" \
      --metadata "condition=floras50_ctc_final f50_w2048 repeat1 step_182892 rev16_${REV16_SPLIT} mode=${EVAL_MODE} window_size=${WINDOW_SIZE} override_window_size=${OVERRIDE_WINDOW_SIZE} seq_len=${SEQ_LEN} overlap=${OVERLAP}" \
      --metadata "repo_branch=${REPO_BRANCH}" \
      --metadata "repo_commit=${REPO_COMMIT}" \
      --max-log-chars 5000 \
      --max-body-chars 12000 \
      --dry-run
  else
    python "$REPO_DIR/symphony/scripts/linear_stanage_callback.py" \
      --issue-id "$ISSUE_ID" \
      --state-name Todo \
      --exit-code "$status" \
      --log-out "$STDOUT_LOG" \
      --log-err "$STDERR_LOG" \
      --log "$EVAL_LOG" \
      --artifact-path "$RUN_DIR" \
      --checkpoint-path "$CHECKPOINT" \
      --summary-file "$SUMMARY_FILE" \
      --title "ROB-327 Floras-only CTC Rev16 eval finished" \
      --metadata "run_kind=${RUN_KIND}" \
      --metadata "condition=floras50_ctc_final f50_w2048 repeat1 step_182892 rev16_${REV16_SPLIT} mode=${EVAL_MODE} window_size=${WINDOW_SIZE} override_window_size=${OVERRIDE_WINDOW_SIZE} seq_len=${SEQ_LEN} overlap=${OVERLAP}" \
      --metadata "repo_branch=${REPO_BRANCH}" \
      --metadata "repo_commit=${REPO_COMMIT}" \
      --max-log-chars 5000 \
      --max-body-chars 12000
  fi
}

on_exit() {
  status=$?
  set +e
  if [ "$status" -eq 0 ]; then
    touch "${RUN_DIR}/RUN_COMPLETE"
  else
    printf "%s\n" "$status" > "${RUN_DIR}/RUN_FAILED"
  fi
  post_callback "$status"
  exit "$status"
}
trap on_exit EXIT

{
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting ROB-327 Floras-only CTC Rev16 eval"
  echo "host=$(hostname)"
  echo "job_id=${SLURM_JOB_ID:-manual}"
  echo "run_kind=${RUN_KIND}"
  echo "run_dir=${RUN_DIR}"
  echo "repo_dir=${REPO_DIR}"
  echo "checkpoint=${CHECKPOINT}"
  echo "split=${REV16_SPLIT}"
  echo "eval_mode=${EVAL_MODE}"
  echo "seq_len=${SEQ_LEN}"
  echo "overlap=${OVERLAP}"
  echo "window_size=${WINDOW_SIZE}"
  echo "override_window_size=${OVERRIDE_WINDOW_SIZE}"
  echo "max_recordings=${MAX_RECORDINGS:-all}"
} | tee "$EVAL_LOG"

if [ "${ROB327_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "callback-only validation requested" | tee -a "$EVAL_LOG"
  exit 0
fi

test -f "$CHECKPOINT"
test -f "${REPO_DIR}/symphony/scripts/rob327_rev16_floras_ctc_eval.py"
test -f "${REPO_DIR}/lcasr/artifacts/floras50/tokenizer.model"
test -f "/mnt/parscratch/users/acp21rjf/rev_benchmark/test.txt"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TMPDIR="${RUN_DIR}/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export MPLCONFIGDIR="${RUN_DIR}/matplotlib"
export XDG_CACHE_HOME="${RUN_DIR}/cache"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/eval:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""

cd "$REPO_DIR"

if [ -n "$MAX_RECORDINGS" ]; then
  python symphony/scripts/rob327_rev16_floras_ctc_eval.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$RUN_DIR" \
    --split "$REV16_SPLIT" \
    --seq-len "$SEQ_LEN" \
    --overlap "$OVERLAP" \
    --window-size "$WINDOW_SIZE" \
    --override-window-size "$OVERRIDE_WINDOW_SIZE" \
    --max-sequence-length "$MAX_SEQUENCE_LENGTH" \
    --evaluation-mode "$EVAL_MODE" \
    --model-class SCConformerXL \
    --max-recordings "$MAX_RECORDINGS" \
    2>&1 | tee -a "$EVAL_LOG"
else
  python symphony/scripts/rob327_rev16_floras_ctc_eval.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$RUN_DIR" \
    --split "$REV16_SPLIT" \
    --seq-len "$SEQ_LEN" \
    --overlap "$OVERLAP" \
    --window-size "$WINDOW_SIZE" \
    --override-window-size "$OVERRIDE_WINDOW_SIZE" \
    --max-sequence-length "$MAX_SEQUENCE_LENGTH" \
    --evaluation-mode "$EVAL_MODE" \
    --model-class SCConformerXL \
    2>&1 | tee -a "$EVAL_LOG"
fi
