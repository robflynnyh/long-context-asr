#!/bin/bash
#SBATCH --job-name=rob78-finalize
#SBATCH --time=00:10:00
#SBATCH --mem=4GB
#SBATCH --partition=interactive
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-78/finalizer-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-78/finalizer-%j.err

set -euo pipefail

REPO_DIR="${ROB78_REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-78}"
ARRAY_JOB_ID="${ROB78_ARRAY_JOB_ID:?ROB78_ARRAY_JOB_ID is required}"
EXPECTED_COMMIT="${ROB78_GIT_COMMIT:-}"

cd "${REPO_DIR}"

if [ -n "${EXPECTED_COMMIT}" ] && [ "$(git rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]; then
  echo "remote checkout is not at expected commit ${EXPECTED_COMMIT}" >&2
  git rev-parse HEAD >&2
  exit 2
fi

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
python symphony/rob78_finalize.py --array-job-id "${ARRAY_JOB_ID}" --state Todo
