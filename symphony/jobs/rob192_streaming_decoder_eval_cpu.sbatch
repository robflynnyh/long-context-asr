#!/bin/bash
#SBATCH --job-name=rob192-eval-cpu
#SBATCH --partition=cpu
#SBATCH --time=06:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/eval/rob192-eval-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/eval/rob192-eval-%j.err

set -euo pipefail

REPO_DIR="${ROB192_REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-192}"
cd "$REPO_DIR"

export ROB192_REPO_DIR="$REPO_DIR"
export ROB192_EVAL_RUN_ID="${ROB192_EVAL_RUN_ID:-rob192-final-streaming-eval-${SLURM_JOB_ID}}"
export ROB192_EVAL_OUT_LOG="${ROB192_EVAL_OUT_LOG:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/eval/rob192-eval-${SLURM_JOB_ID}.out}"
export ROB192_EVAL_ERR_LOG="${ROB192_EVAL_ERR_LOG:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-192/eval/rob192-eval-${SLURM_JOB_ID}.err}"

bash symphony/scripts/run_rob192_streaming_decoder_eval_stanage.sh
