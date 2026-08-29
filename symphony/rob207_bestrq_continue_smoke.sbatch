#!/bin/bash
#SBATCH --job-name=rob207-bestrq-smoke
#SBATCH --time=00:20:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=2
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-207/smoke-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-207/smoke-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-207}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-207}"
CONFIG_PATH="${ROB207_SMOKE_CONFIG:-exp/configs/ssl/bestrq_6l_2048_spotify_rob207_resume_smoke.yaml}"
SOURCE_CHECKPOINT="${ROB207_SOURCE_CHECKPOINT:-/mnt/parscratch/users/acp21rjf/spotify/bestrq_ssl/rob100_papermask_p012_l4_sc_off_20260520/step_105360.pt}"
CHECKPOINT_DIR="${ROB207_SMOKE_CHECKPOINT_DIR:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-207/smoke-checkpoints-peritemmask-20260607}"
TRAIN_LOG="${ARTIFACT_DIR}/smoke-train-${SLURM_JOB_ID:-manual}.log"

cd "$REPO_DIR"
mkdir -p "$ARTIFACT_DIR" "$CHECKPOINT_DIR"
if [[ ! -f "$CHECKPOINT_DIR/step_105360.pt" ]]; then
  cp "$SOURCE_CHECKPOINT" "$CHECKPOINT_DIR/step_105360.pt"
fi

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

export PYTHONPATH="$REPO_DIR"
python exp/train_bestRQ.py -config "$CONFIG_PATH" --remove_scheduler --num_workers 0 --prefetch_factor 1 2>&1 | tee "$TRAIN_LOG"
