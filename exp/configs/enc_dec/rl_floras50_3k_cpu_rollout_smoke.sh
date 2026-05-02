#!/bin/bash
#SBATCH --job-name=ROB-26-rl-smoke
#SBATCH --time=00:45:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/cpu-rollout-smoke-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/cpu-rollout-smoke-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces/ROB-26}"
cd "${REPO_DIR}"
mkdir -p /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

python exp/train_files/train_enc_dec_rl.py \
  --config exp/configs/enc_dec/rl_floras50_3k.yaml \
  --smoke_rollout \
  --smoke_max_generate 4 \
  --smoke_num_rollouts 2
