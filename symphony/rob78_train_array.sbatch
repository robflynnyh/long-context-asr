#!/bin/bash
#SBATCH --job-name=rob78-train
#SBATCH --time=90:00:00
#SBATCH --mem=130GB
#SBATCH --partition=gpu-h100-nvl
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --array=0-14
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-78/gpu-%A_%a.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-78/gpu-%A_%a.err

set -euo pipefail

REPO_DIR="${ROB78_REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-78}"
ARTIFACT_ROOT="${ROB78_ARTIFACT_ROOT:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-78}"
EXPECTED_COMMIT="${ROB78_GIT_COMMIT:-}"
MANIFEST="${ARTIFACT_ROOT}/train_configs.txt"

cd "${REPO_DIR}"

if [ -n "${EXPECTED_COMMIT}" ] && [ "$(git rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]; then
  echo "remote checkout is not at expected commit ${EXPECTED_COMMIT}" >&2
  git rev-parse HEAD >&2
  exit 2
fi

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

if [ ! -s "${MANIFEST}" ]; then
  python symphony/rob78_prepare_configs.py --mode train --artifact-root "${ARTIFACT_ROOT}"
fi

CONFIG="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST}")"
if [ -z "${CONFIG}" ]; then
  echo "no config found for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 3
fi

echo "ROB-78 task ${SLURM_ARRAY_TASK_ID} config: ${CONFIG}"
python exp/train.py -config "${CONFIG}" --remove_scheduler --reset_step -num_workers 4 -pin_memory -prefetch 2
