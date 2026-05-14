#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/users/acp21rjf/long-context-asr}"
cd "${REPO_DIR}"

GPU_JOB_ID="$(sbatch --parsable symphony/rob81_floras50_finetune_gpu.sbatch)"
FINALIZER_JOB_ID="$(
  sbatch --parsable \
    --dependency=afterany:${GPU_JOB_ID} \
    --export=ALL,ROB81_GPU_JOB_ID=${GPU_JOB_ID} \
    symphony/rob81_floras50_finetune_finalizer.sbatch
)"

echo "gpu_job_id=${GPU_JOB_ID}"
echo "finalizer_job_id=${FINALIZER_JOB_ID}"
echo "status_cmd=squeue -j ${GPU_JOB_ID},${FINALIZER_JOB_ID} -o '%i|%j|%T|%R|%S|%M|%l|%P'"
