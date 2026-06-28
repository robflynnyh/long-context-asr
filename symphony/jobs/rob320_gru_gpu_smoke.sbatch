#!/bin/bash
#SBATCH --job-name=ROB-320-gru-gpu-smoke
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-h100-nvl
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --mem=130GB
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-320/logs/gpu_smoke-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-320/logs/gpu_smoke-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/users/acp21rjf/long-context-asr}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-320}"

mkdir -p "${ARTIFACT_ROOT}/logs" "${ARTIFACT_ROOT}/cuda_final_smoke" "${ARTIFACT_ROOT}/checkpoints"
cd "${REPO_DIR}"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
python "${REPO_DIR}/symphony/scripts/rob320_gru_cpu_smoke.py" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --device cuda \
  --model-size final \
  --max-frames 1024
