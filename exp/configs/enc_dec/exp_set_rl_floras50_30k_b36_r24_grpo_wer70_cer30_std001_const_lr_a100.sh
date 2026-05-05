#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/users/acp21rjf/long-context-asr}"
cd "${REPO_DIR}/exp"

python run_launcher.py \
  --template configs/enc_dec/exp_set_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr.yaml \
  --mode a100 \
  --launch train_files/train_enc_dec_rl.py
