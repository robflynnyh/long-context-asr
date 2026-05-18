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
LOG_PREFIX="${ROB81_LOG_PREFIX:-floras12-nw0}"
OUT_LOG="${ARTIFACT_DIR}/${LOG_PREFIX}-${GPU_JOB_ID}.out"
ERR_LOG="${ARTIFACT_DIR}/${LOG_PREFIX}-${GPU_JOB_ID}.err"
CHECKPOINT_DIR="${ROB81_CHECKPOINT_DIR:-${ARTIFACT_DIR}/checkpoints/supervised_floras50_spotifytok_safe_norm_drop_oov_lr1e-4_12ep_nw0}"

cd "${REPO_DIR}"

if [[ -z "${LINEAR_API_KEY:-}" && -f symphony/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source symphony/.env
  set +a
fi

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

STATE="$(sacct -j "${GPU_JOB_ID}" --format=State,ExitCode -n -P | head -n 1 || true)"
if [[ "${STATE}" == COMPLETED*0:0* ]]; then
  STATUS=success
else
  STATUS=failure
fi

DRY_RUN_ARGS=()
if [[ "${ROB81_FINALIZER_DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN_ARGS=(--dry-run)
fi

python symphony/rob81_linear_callback.py \
  "${DRY_RUN_ARGS[@]}" \
  --gpu-job-id "${GPU_JOB_ID}" \
  --status "${STATUS}" \
  --out-log "${OUT_LOG}" \
  --err-log "${ERR_LOG}" \
  --checkpoint-dir "${CHECKPOINT_DIR}"
