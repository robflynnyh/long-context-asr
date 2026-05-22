#!/bin/bash
#SBATCH --job-name=rob81-ted-eval
#SBATCH --partition=interactive
#SBATCH --time=04:00:00
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/eval/tedlium-eval-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/eval/tedlium-eval-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/users/acp21rjf/long-context-asr}"
CONFIG="${CONFIG:-eval_configs/rob81_floras_finetune_tedlium.yaml}"
ARTIFACT_DIR=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/eval

mkdir -p "${ARTIFACT_DIR}"

cd "${REPO_DIR}/eval"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export LCASR_TEDLIUM_ROOT="${LCASR_TEDLIUM_ROOT:-/mnt/parscratch/users/acp21rjf/TEDLIUM_release1}"

echo "Running ROB-81 TEDLIUM eval"
echo "repo=${REPO_DIR}"
echo "config=${CONFIG}"
echo "LCASR_TEDLIUM_ROOT=${LCASR_TEDLIUM_ROOT}"

python eval_manager.py -config "${CONFIG}"
