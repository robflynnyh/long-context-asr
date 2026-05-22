#!/bin/bash
#SBATCH --job-name=rob100-bestrq-paper
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=36:00:00
#SBATCH --mem=150GB
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-100/gpu-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-100/gpu-%j.err

set -uo pipefail

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-100}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-100}"
CONFIG_PATH="${ROB100_CONFIG:-exp/configs/ssl/bestrq_6l_2048_spotify.yaml}"
CHECKPOINT_DIR="/mnt/parscratch/users/acp21rjf/spotify/bestrq_ssl/rob100_papermask_p012_l4_sc_off_20260520"
WANDB_NAME="rob100_bestrq_6l_2048_papermask_p012_l4_sc_off_20260520"
TRAIN_LOG="${ARTIFACT_DIR}/train-${SLURM_JOB_ID:-manual}.log"
SUMMARY_FILE="${ARTIFACT_DIR}/summary-${SLURM_JOB_ID:-manual}.txt"
STDOUT_LOG="${ARTIFACT_DIR}/gpu-${SLURM_JOB_ID:-manual}.out"
STDERR_LOG="${ARTIFACT_DIR}/gpu-${SLURM_JOB_ID:-manual}.err"
COMMAND="python exp/train_bestRQ.py -config ${CONFIG_PATH} --remove_scheduler --reset_step --no_resume --num_workers ${ROB100_NUM_WORKERS:-0} --prefetch_factor ${ROB100_PREFETCH_FACTOR:-1}"
LINEAR_ENV_FILE="${LINEAR_ENV_FILE:-}"

if [[ -z "$LINEAR_ENV_FILE" ]]; then
  if [[ -f "$HOME/.config/long-context-asr/linear.env" ]]; then
    LINEAR_ENV_FILE="$HOME/.config/long-context-asr/linear.env"
  elif [[ -f "$HOME/.config/sap-longcontext/linear.env" ]]; then
    LINEAR_ENV_FILE="$HOME/.config/sap-longcontext/linear.env"
  fi
fi

write_summary() {
  {
    echo "ROB-100 BEST-RQ paper-mask training"
    echo "Slurm job: ${SLURM_JOB_ID:-manual}"
    echo "Exit code: $1"
    echo "Repo: $REPO_DIR"
    echo "Config: $CONFIG_PATH"
    echo "Command: $COMMAND"
    echo "Checkpoint dir: $CHECKPOINT_DIR"
    echo "W&B run name: $WANDB_NAME"
    echo "Train log: $TRAIN_LOG"
    echo
    echo "Recent train log tail:"
    tail -n 80 "$TRAIN_LOG" 2>/dev/null || true
  } > "$SUMMARY_FILE"
}

on_exit() {
  status=$?
  set +e
  mkdir -p "$ARTIFACT_DIR"
  write_summary "$status"

  callback_args=(
    "$REPO_DIR/symphony/scripts/linear_job_callback.py"
    --issue-id ROB-100
    --state-name Todo
    --job-id "${SLURM_JOB_ID:-manual}"
    --exit-code "$status"
    --log-out "$STDOUT_LOG"
    --log-err "$STDERR_LOG"
    --output-path "$CHECKPOINT_DIR"
    --summary-file "$SUMMARY_FILE"
    --title "ROB-100 BEST-RQ paper-mask training finished"
    --job-label "Stanage GPU job"
  )
  if [[ "${ROB100_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
    callback_args+=(--dry-run)
  fi

  python "${callback_args[@]}"
  callback_status=$?
  if [[ "$callback_status" -ne 0 ]]; then
    echo "ROB-100 Linear callback failed with exit status ${callback_status}" >&2
  fi
  exit "$status"
}
trap on_exit EXIT
trap 'trap - TERM INT; exit 143' TERM
trap 'trap - INT TERM; exit 130' INT

set -e

cd "$REPO_DIR"
mkdir -p "$ARTIFACT_DIR" "$CHECKPOINT_DIR"

if [[ -n "$LINEAR_ENV_FILE" && -f "$LINEAR_ENV_FILE" ]]; then
  set -a
  source "$LINEAR_ENV_FILE"
  set +a
fi

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

export PYTHONPATH="$REPO_DIR"
export WANDB_DIR="/mnt/parscratch/users/acp21rjf/wandb_log"

if [[ "${ROB100_CALLBACK_ONLY:-0}" == "1" ]]; then
  echo "ROB-100 callback-only validation"
  exit 0
fi

set -o pipefail
$COMMAND 2>&1 | tee "$TRAIN_LOG"
