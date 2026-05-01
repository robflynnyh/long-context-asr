#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=4
#SBATCH --job-name=lcasr-hf-upload
#SBATCH --output=/mnt/parscratch/users/acp21rjf/hf_lcasr_upload_work/slurm-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/hf_lcasr_upload_work/slurm-%j.err

set -euo pipefail

REPO_ROOT=/users/acp21rjf/long-context-asr
WORK_DIR=/mnt/parscratch/users/acp21rjf/hf_lcasr_upload_work
CONDA_ENV=/mnt/parscratch/users/acp21rjf/conda/main

mkdir -p "$WORK_DIR"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export HF_HUB_CACHE="${HF_HUB_CACHE:-$WORK_DIR/hf_cache}"
export HF_XET_CACHE="${HF_XET_CACHE:-$WORK_DIR/xet_cache}"

echo "started: $(date)"
echo "host: $(hostname)"
echo "repo root: $REPO_ROOT"
echo "work dir: $WORK_DIR"
echo "hf hub cache: $HF_HUB_CACHE"
echo "hf xet cache: $HF_XET_CACHE"

if command -v module >/dev/null 2>&1; then
    module load Anaconda3/2022.10 || true
fi

if [[ -f /opt/apps/testapps/common/software/staging/Anaconda3/2022.10/etc/profile.d/conda.sh ]]; then
    source /opt/apps/testapps/common/software/staging/Anaconda3/2022.10/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
else
    export PATH="$CONDA_ENV/bin:$PATH"
fi

GROUP_ARGS=()
if [[ -n "${UPLOAD_GROUPS:-}" ]]; then
    IFS=',' read -ra UPLOAD_GROUP_ARRAY <<< "$UPLOAD_GROUPS"
    for group in "${UPLOAD_GROUP_ARRAY[@]}"; do
        GROUP_ARGS+=(--group "$group")
    done
fi

EXTRA_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--dry-run)
fi
if [[ "${PRIVATE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--private)
fi

echo "groups: ${UPLOAD_GROUPS:-all}"
echo "dry run: ${DRY_RUN:-0}"
echo "private: ${PRIVATE:-0}"

python -c "from huggingface_hub import HfApi; print('hf user:', HfApi().whoami()['name'])"

CMD=(python scripts/upload_lcasr_checkpoints_to_hf.py)
if [[ -n "${UPLOAD_GROUPS:-}" ]]; then
    CMD+=("${GROUP_ARGS[@]}")
fi
if [[ "${DRY_RUN:-0}" == "1" || "${PRIVATE:-0}" == "1" ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}"

echo "finished: $(date)"
