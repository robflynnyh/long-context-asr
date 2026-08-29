#!/bin/bash
#SBATCH --job-name=rob192-ted-chunks
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=08:00:00
#SBATCH --mem=96GB
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/eval/rob192-ted-chunks-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/eval/rob192-ted-chunks-%j.err

set -euo pipefail

REPO_DIR="${ROB192_REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-192}"
cd "$REPO_DIR"

export ROB192_REPO_DIR="$REPO_DIR"
export ROB192_TED_CHUNK_RUN_ID="${ROB192_TED_CHUNK_RUN_ID:-rob192-tedlium-independent-chunks-${SLURM_JOB_ID}}"
export ROB192_TED_CHUNK_OUT_LOG="${ROB192_TED_CHUNK_OUT_LOG:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/eval/rob192-ted-chunks-${SLURM_JOB_ID}.out}"
export ROB192_TED_CHUNK_ERR_LOG="${ROB192_TED_CHUNK_ERR_LOG:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/eval/rob192-ted-chunks-${SLURM_JOB_ID}.err}"

bash symphony/scripts/run_rob192_tedlium_independent_chunks_stanage.sh
