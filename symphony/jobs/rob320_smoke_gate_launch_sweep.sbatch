#!/bin/bash
#SBATCH --job-name=ROB-320-smoke-gate
#SBATCH --time=00:30:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-320/logs/smoke_gate-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-320/logs/smoke_gate-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/users/acp21rjf/long-context-asr}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-320}"
SMOKE_JOB_ID="${SMOKE_JOB_ID:?SMOKE_JOB_ID must be set}"
BRANCH="${BRANCH:-$(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD)}"
COMMIT="${COMMIT:-$(git -C "${REPO_DIR}" rev-parse HEAD)}"

mkdir -p "${ARTIFACT_ROOT}/logs" "${ARTIFACT_ROOT}/finalizer"
cd "${REPO_DIR}"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

for env_file in \
  "${REPO_DIR}/symphony/.env" \
  "/users/acp21rjf/long-context-asr/symphony/.env" \
  "${HOME}/.config/long-context-asr/linear.env" \
  "${HOME}/.config/sap-longcontext/linear.env"; do
  if [[ -f "${env_file}" ]]; then
    set -a
    source "${env_file}"
    set +a
    break
  fi
done

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
python "${REPO_DIR}/symphony/scripts/rob320_smoke_gate_launch_sweep.py" \
  --issue-id ROB-320 \
  --repo-dir "${REPO_DIR}" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --smoke-job-id "${SMOKE_JOB_ID}" \
  --branch "${BRANCH}" \
  --commit "${COMMIT}" \
  --state-name-queued Backlog \
  --state-name-failed Todo
