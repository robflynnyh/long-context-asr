#!/bin/bash
#SBATCH --job-name=rob81-ft-smoke
#SBATCH --partition=interactive
#SBATCH --time=01:00:00
#SBATCH --mem=130GB
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/finetune-smoke-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/finetune-smoke-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/users/acp21rjf/long-context-asr}"
ARTIFACT_DIR=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81
MANIFEST_DIR="${ARTIFACT_DIR}/manifests"
NORMALIZED_TXT_DIR="${MANIFEST_DIR}/normalized_txt"
FILTERED_MANIFEST="${MANIFEST_DIR}/floras50_safe_norm_drop_oov.json"
SUMMARY_JSON="${MANIFEST_DIR}/floras50_safe_norm_drop_oov_summary.json"

cd "${REPO_DIR}"
mkdir -p "${ARTIFACT_DIR}" "${MANIFEST_DIR}" "${NORMALIZED_TXT_DIR}"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

python -m py_compile \
  exp/train_files/train_enc_dec.py \
  symphony/rob81_prepare_floras_manifest.py \
  symphony/rob81_finetune_smoke.py \
  symphony/rob81_linear_callback.py

python symphony/rob81_prepare_floras_manifest.py \
  --mapping /users/acp21rjf/align_floras50/tmp/mapping.json \
  --tokenizer lcasr/artifacts/tokenizer.model \
  --output "${FILTERED_MANIFEST}" \
  --text-output-dir "${NORMALIZED_TXT_DIR}" \
  --summary-json "${SUMMARY_JSON}"

python symphony/rob81_finetune_smoke.py \
  --config exp/configs/enc_dec/rob81_floras50_supervised_12ep_lr1e-4_nw0.yaml \
  --max-records 88 \
  --num-workers 0 \
  --prefetch-factor 1

python symphony/rob81_linear_callback.py \
  --dry-run \
  --gpu-job-id smoke \
  --status success \
  --out-log "${ARTIFACT_DIR}/finetune-smoke-${SLURM_JOB_ID}.out" \
  --err-log "${ARTIFACT_DIR}/finetune-smoke-${SLURM_JOB_ID}.err" \
  --checkpoint-dir /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/checkpoints/supervised_floras50_spotifytok_safe_norm_drop_oov_lr1e-4_12ep_nw0
