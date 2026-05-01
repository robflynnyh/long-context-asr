#!/bin/bash
#SBATCH --job-name=ROB-26-rl-cpu
#SBATCH --time=00:20:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=2
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/cpu-debug-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/cpu-debug-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces/ROB-26}"
cd "${REPO_DIR}"
mkdir -p /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

python -m compileall exp/train_files/train_enc_dec_rl.py
python exp/train_files/train_enc_dec_rl.py --self_test
python exp/train_files/train_enc_dec_rl.py \
  --config exp/configs/enc_dec/rl_floras50_3k.yaml \
  --validate_config_only \
  --validate_load_model
