#!/bin/bash
#SBATCH --job-name=rob78-smoke
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --partition=interactive
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-78/smoke-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-78/smoke-%j.err

set -euo pipefail

REPO_DIR="${ROB78_REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-78}"
ARTIFACT_ROOT="${ROB78_ARTIFACT_ROOT:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-78}"
EXPECTED_COMMIT="${ROB78_GIT_COMMIT:-}"

mkdir -p "${ARTIFACT_ROOT}"
cd "${REPO_DIR}"

if [ -n "${EXPECTED_COMMIT}" ] && [ "$(git rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]; then
  echo "remote checkout is not at expected commit ${EXPECTED_COMMIT}" >&2
  git rev-parse HEAD >&2
  exit 2
fi

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""

python symphony/rob78_prepare_configs.py --mode smoke --artifact-root "${ARTIFACT_ROOT}"
CONFIG="$(sed -n '1p' "${ARTIFACT_ROOT}/smoke_configs.txt")"
python exp/train.py -config "${CONFIG}" --remove_scheduler --reset_step -num_workers 0
