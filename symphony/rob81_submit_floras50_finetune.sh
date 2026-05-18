#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/users/acp21rjf/long-context-asr}"
cd "${REPO_DIR}"

if [[ -z "${LINEAR_API_KEY:-}" && -f symphony/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source symphony/.env
  set +a
fi
if [[ -z "${LINEAR_API_KEY:-}" ]]; then
  echo "LINEAR_API_KEY is required in the submit environment or symphony/.env" >&2
  exit 1
fi

CHECKPOINT_DIR="/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/checkpoints/supervised_floras50_spotifytok_safe_norm_drop_oov_lr1e-4_12ep_nw0"
LOG_PREFIX="floras12-nw0"

GPU_JOB_ID="$(sbatch --parsable symphony/rob81_floras50_finetune_gpu.sbatch)"
FINALIZER_JOB_ID="$(
  sbatch --parsable \
    --dependency=afterany:${GPU_JOB_ID} \
    --export=ALL,ROB81_GPU_JOB_ID=${GPU_JOB_ID},ROB81_LOG_PREFIX=${LOG_PREFIX},ROB81_CHECKPOINT_DIR=${CHECKPOINT_DIR} \
    symphony/rob81_floras50_finetune_finalizer.sbatch
)"

echo "gpu_job_id=${GPU_JOB_ID}"
echo "finalizer_job_id=${FINALIZER_JOB_ID}"
echo "status_cmd=squeue -j ${GPU_JOB_ID},${FINALIZER_JOB_ID} -o '%i|%j|%T|%R|%S|%M|%l|%P'"
