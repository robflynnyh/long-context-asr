#!/bin/bash
#SBATCH --job-name=rob207-bestrq-fixmask
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=72:00:00
#SBATCH --mem=150GB
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-207/gpu-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-207/gpu-%j.err

set -uo pipefail

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-207}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-207}"
CONFIG_PATH="${ROB207_CONFIG:-exp/configs/ssl/bestrq_6l_2048_spotify_rob207_resume4epoch.yaml}"
SOURCE_CHECKPOINT="${ROB207_SOURCE_CHECKPOINT:-/mnt/parscratch/users/acp21rjf/spotify/bestrq_ssl/rob100_papermask_p012_l4_sc_off_20260520/step_105360.pt}"
CHECKPOINT_DIR="${ROB207_CHECKPOINT_DIR:-/mnt/parscratch/users/acp21rjf/spotify/bestrq_ssl/rob207_papermask_p012_l4_sc_off_resume4epoch_lr3e4_peritemmask_20260607}"
WANDB_NAME="rob207_bestrq_6l_2048_papermask_resume4epoch_lr3e4_peritemmask_20260607"
TRAIN_LOG="${ARTIFACT_DIR}/train-${SLURM_JOB_ID:-manual}.log"
SUMMARY_FILE="${ARTIFACT_DIR}/summary-${SLURM_JOB_ID:-manual}.txt"
STDOUT_LOG="${ARTIFACT_DIR}/gpu-${SLURM_JOB_ID:-manual}.out"
STDERR_LOG="${ARTIFACT_DIR}/gpu-${SLURM_JOB_ID:-manual}.err"
COMMAND="python exp/train_bestRQ.py -config ${CONFIG_PATH} --remove_scheduler --num_workers ${ROB207_NUM_WORKERS:-0} --prefetch_factor ${ROB207_PREFETCH_FACTOR:-1}"
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
    echo "ROB-207 BEST-RQ four-epoch continuation with per-sample mask selection"
    echo "Slurm job: ${SLURM_JOB_ID:-manual}"
    echo "Exit code: $1"
    echo "Repo: $REPO_DIR"
    echo "Config: $CONFIG_PATH"
    echo "Command: $COMMAND"
    echo "Source checkpoint: $SOURCE_CHECKPOINT"
    echo "Checkpoint dir: $CHECKPOINT_DIR"
    echo "Checkpoint cadence: save_every_n_epochs=1"
    echo "Learning rate: 3e-4"
    echo "Mask sampling: SpeechBrain-style blocks sampled independently per batch item from each valid stacked length"
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
    "$REPO_DIR/symphony/scripts/linear_stanage_callback.py"
    --issue-id ROB-207
    --state-name Todo
    --exit-code "$status"
    --log-out "$STDOUT_LOG"
    --log-err "$STDERR_LOG"
    --log "$TRAIN_LOG"
    --checkpoint-path "$CHECKPOINT_DIR"
    --artifact-path "$ARTIFACT_DIR"
    --summary-file "$SUMMARY_FILE"
    --title "ROB-207 BEST-RQ corrected-mask continuation finished"
    --job-label "Stanage GPU job"
    --metadata "branch=symphony/ROB-207-more-bestrq-training"
    --metadata "config=$CONFIG_PATH"
    --metadata "source_checkpoint=$SOURCE_CHECKPOINT"
    --metadata "learning_rate=3e-4"
    --metadata "checkpoint_cadence=1 per epoch"
    --metadata "mask_sampling=per-sample valid stacked length"
  )
  if [[ "${ROB207_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
    callback_args+=(--dry-run)
  fi

  python "${callback_args[@]}"
  callback_status=$?
  if [[ "$callback_status" -ne 0 ]]; then
    echo "ROB-207 Linear callback failed with exit status ${callback_status}" >&2
  fi
  exit "$status"
}
trap on_exit EXIT
trap 'trap - TERM INT; exit 143' TERM
trap 'trap - INT TERM; exit 130' INT

set -e

cd "$REPO_DIR"
mkdir -p "$ARTIFACT_DIR" "$CHECKPOINT_DIR"
if [[ ! -f "$CHECKPOINT_DIR/step_105360.pt" ]]; then
  cp "$SOURCE_CHECKPOINT" "$CHECKPOINT_DIR/step_105360.pt"
fi

if [[ -n "$LINEAR_ENV_FILE" && -f "$LINEAR_ENV_FILE" ]]; then
  set -a
  source "$LINEAR_ENV_FILE"
  set +a
fi

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

export PYTHONPATH="$REPO_DIR"
export WANDB_DIR="/mnt/parscratch/users/acp21rjf/wandb_log"

if [[ "${ROB207_CALLBACK_ONLY:-0}" == "1" ]]; then
  echo "ROB-207 callback-only validation"
  exit 0
fi

set -o pipefail
$COMMAND 2>&1 | tee "$TRAIN_LOG"
