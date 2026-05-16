#!/bin/bash
#SBATCH --job-name=rob81-ft-finalize
#SBATCH --partition=interactive
#SBATCH --time=00:10:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/finalizer-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/finalizer-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/users/acp21rjf/long-context-asr}"
ARTIFACT_DIR=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81
GPU_JOB_ID="${ROB81_GPU_JOB_ID:?ROB81_GPU_JOB_ID must be exported}"
OUT_LOG="${ARTIFACT_DIR}/floras12-${GPU_JOB_ID}.out"
ERR_LOG="${ARTIFACT_DIR}/floras12-${GPU_JOB_ID}.err"
if [[ -f "${ARTIFACT_DIR}/floras12-b22-${GPU_JOB_ID}.out" ]]; then
  OUT_LOG="${ARTIFACT_DIR}/floras12-b22-${GPU_JOB_ID}.out"
  ERR_LOG="${ARTIFACT_DIR}/floras12-b22-${GPU_JOB_ID}.err"
fi
CHECKPOINT_DIR="${ARTIFACT_DIR}/checkpoints/supervised_floras50_spotifytok_safe_norm_drop_oov_lr1e-4_12ep_b22"

cd "${REPO_DIR}"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

STATE="$(sacct -j "${GPU_JOB_ID}" --format=State,ExitCode -n -P | head -n 1 || true)"
if [[ "${STATE}" == COMPLETED*0:0* ]]; then
  STATUS=success
else
  STATUS=failure
fi

python symphony/rob81_linear_callback.py \
  --gpu-job-id "${GPU_JOB_ID}" \
  --status "${STATUS}" \
  --out-log "${OUT_LOG}" \
  --err-log "${ERR_LOG}" \
  --checkpoint-dir "${CHECKPOINT_DIR}"
