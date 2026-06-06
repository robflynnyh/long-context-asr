#!/bin/bash
#SBATCH --job-name=rob192-ted-chunks-cpu
#SBATCH --partition=interactive
#SBATCH --time=08:00:00
#SBATCH --mem=96GB
#SBATCH --cpus-per-task=16
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/eval/rob192-ted-chunks-cpu-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/eval/rob192-ted-chunks-cpu-%j.err

set -euo pipefail

REPO_DIR="${ROB192_REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-192}"
cd "$REPO_DIR"

export ROB192_REPO_DIR="$REPO_DIR"
export ROB192_TED_CHUNK_RUN_ID="${ROB192_TED_CHUNK_RUN_ID:-rob192-tedlium-independent-chunks-cpu-${SLURM_JOB_ID}}"
export ROB192_TED_CHUNK_OUT_LOG="${ROB192_TED_CHUNK_OUT_LOG:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/eval/rob192-ted-chunks-cpu-${SLURM_JOB_ID}.out}"
export ROB192_TED_CHUNK_ERR_LOG="${ROB192_TED_CHUNK_ERR_LOG:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/eval/rob192-ted-chunks-cpu-${SLURM_JOB_ID}.err}"
export ROB192_TED_CHUNK_FORCE_CPU=1
export ROB192_OMP_THREADS="${ROB192_OMP_THREADS:-16}"

bash symphony/scripts/run_rob192_tedlium_independent_chunks_stanage.sh
