#!/bin/bash
#SBATCH --job-name=rob207-codebook
#SBATCH --time=00:30:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-207/codebook-util-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-207/codebook-util-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-207}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-207/codebook-util-logmel-${SLURM_JOB_ID:-manual}}"
PAIRS="${ROB207_PAIRS:-/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json}"
CHECKPOINT="${ROB207_SOURCE_CHECKPOINT:-/mnt/parscratch/users/acp21rjf/spotify/bestrq_ssl/rob100_papermask_p012_l4_sc_off_20260520/step_105360.pt}"

cd "$REPO_DIR"
mkdir -p "$ARTIFACT_DIR"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

export PYTHONPATH="$REPO_DIR"

python symphony/scripts/rob207_codebook_util_sanity.py \
  --pairs "$PAIRS" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$ARTIFACT_DIR" \
  --max-records "${ROB207_CODEBOOK_MAX_RECORDS:-24}" \
  --min-duration "${ROB207_CODEBOOK_MIN_DURATION:-30}" \
  --max-duration "${ROB207_CODEBOOK_MAX_DURATION:-180}" \
  --max-feature-frames "${ROB207_CODEBOOK_MAX_FEATURE_FRAMES:-12000}" \
  --seed "${ROB207_CODEBOOK_SEED:-207}"
