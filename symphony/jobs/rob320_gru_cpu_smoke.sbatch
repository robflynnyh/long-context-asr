#!/bin/bash
#SBATCH --job-name=ROB-320-gru-cpu-smoke
#SBATCH --time=00:30:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=2
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-320/logs/cpu_smoke-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-320/logs/cpu_smoke-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/users/acp21rjf/long-context-asr}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-320}"

mkdir -p "${ARTIFACT_ROOT}/logs" "${ARTIFACT_ROOT}/cpu_smoke" "${ARTIFACT_ROOT}/checkpoints"
cd "${REPO_DIR}"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
python "${REPO_DIR}/symphony/scripts/rob320_gru_cpu_smoke.py" \
  --artifact-root "${ARTIFACT_ROOT}"
