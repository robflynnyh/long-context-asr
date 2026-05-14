#!/bin/bash
#SBATCH --job-name=rob81-oov
#SBATCH --partition=interactive
#SBATCH --time=00:30:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/oov-audit-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/oov-audit-%j.err

set -euo pipefail

ARTIFACT_DIR=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81
SCRIPT=${ARTIFACT_DIR}/rob81_floras_oov_audit_normalized_v3.py
OUTPUT_JSON=${ARTIFACT_DIR}/oov-audit-normalized-v3-full.json

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

python "${SCRIPT}" \
  --mapping /users/acp21rjf/align_floras50/tmp/mapping.json \
  --tokenizer spotify_default /users/acp21rjf/long-context-asr/lcasr/artifacts/tokenizer.model \
  --tokenizer floras50 /users/acp21rjf/long-context-asr/lcasr/artifacts/floras50/tokenizer.model \
  --normalization safe \
  --executor thread \
  --workers "${SLURM_CPUS_PER_TASK:-8}" \
  --output-json "${OUTPUT_JSON}"
