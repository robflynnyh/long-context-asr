#!/bin/bash
#SBATCH --job-name=rob192-ft-floras12
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=80:00:00
#SBATCH --mem=130GB
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/rob192-floras-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/rob192-floras-%j.err

set -euo pipefail

REPO_DIR="${ROB192_REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-192}"
cd "$REPO_DIR"

export ROB192_REPO_DIR="$REPO_DIR"
export ROB192_OUT_LOG="${ROB192_OUT_LOG:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/rob192-floras-${SLURM_JOB_ID}.out}"
export ROB192_ERR_LOG="${ROB192_ERR_LOG:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/rob192-floras-${SLURM_JOB_ID}.err}"
export ROB192_RUN_ID="${ROB192_RUN_ID:-rob192-rope-streaming-floras50-lr5e-5-12ep-${SLURM_JOB_ID}}"

bash symphony/scripts/run_rob192_streaming_decoder_asr_floras_stanage.sh
