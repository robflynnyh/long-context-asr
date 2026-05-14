#!/bin/bash
#SBATCH --job-name=rob81-ft-floras12
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=80:00:00
#SBATCH --mem=82GB
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/floras12-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/floras12-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/users/acp21rjf/long-context-asr}"
ARTIFACT_DIR=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81
CHECKPOINT_DIR="${ARTIFACT_DIR}/checkpoints/supervised_floras50_spotifytok_safe_norm_drop_oov_lr1e-4_12ep"

cd "${REPO_DIR}"
mkdir -p "${ARTIFACT_DIR}" "${CHECKPOINT_DIR}" "${ARTIFACT_DIR}/wandb"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

python exp/train_files/train_enc_dec.py \
  --config exp/configs/enc_dec/rob81_floras50_supervised_12ep_lr1e-4.yaml \
  --reset_step \
  --remove_scheduler \
  --num_workers 4 \
  --pin_memory \
  --prefetch_factor 2
