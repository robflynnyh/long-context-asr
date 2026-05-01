#!/bin/bash
#SBATCH --job-name=rob28-eval
#SBATCH --time=20:00:00
#SBATCH --mem=80GB
#SBATCH --partition=gpu-h100-nvl
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-28/logs/eval-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-28/logs/eval-%j.err

set -euo pipefail

REPO=/mnt/parscratch/users/acp21rjf/symphony-workspaces/ROB-28
ARTIFACT_ROOT=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-28
FT_DIR="${ARTIFACT_ROOT}/checkpoints/floras50_finetune_ctx2048"
RUNTIME_CONFIG="${ARTIFACT_ROOT}/configs/eval_earnings22_full.yaml"
RESULT_CSV="${REPO}/eval/results/rob28_earnings22_full_ssl_finetuned.csv"

mkdir -p "${ARTIFACT_ROOT}/logs" "${ARTIFACT_ROOT}/configs" "${REPO}/eval/results"
cd "${REPO}"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

CHECKPOINT=$(/mnt/parscratch/users/acp21rjf/conda/main/bin/python - <<'PY'
import os
import re

checkpoint_dir = "/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-28/checkpoints/floras50_finetune_ctx2048"
checkpoints = [name for name in os.listdir(checkpoint_dir) if re.match(r"step_\d+\.pt$", name)]
if not checkpoints:
    raise SystemExit(f"No fine-tuned checkpoint found in {checkpoint_dir}")
latest = max(checkpoints, key=lambda name: int(name.split("_", 1)[1].split(".", 1)[0]))
print(os.path.join(checkpoint_dir, latest))
PY
)

cat > "${RUNTIME_CONFIG}" <<YAML
models:
  - name: rob28_best_rq_floras50_ctx2048
    repeat: 1
    seq_len: 2048
    overlap_ratio: 0.875
    path: ${CHECKPOINT}
    description: ROB-28 BEST-RQ Spotify SSL checkpoint fine-tuned on floras-50 for 1 epoch.

args:
  verbose: true
  model_class: SCConformerXL
  save_dataframe_path: ${RESULT_CSV}
  evaluation_mode: windowed_attention
  max_sequence_length: 3600000
  tokenizer_path: ${REPO}/lcasr/artifacts/floras50/tokenizer.model

datasets:
  - name: earnings22_full
    splits: [test]
YAML

cd eval
python eval_manager.py -config "${RUNTIME_CONFIG}"
