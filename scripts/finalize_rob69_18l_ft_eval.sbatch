#!/bin/bash
#SBATCH --job-name=rob69-finalize
#SBATCH --time=00:20:00
#SBATCH --partition=interactive
#SBATCH --mem=12GB
#SBATCH --cpus-per-task=2
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-69/finalize-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-69/finalize-%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-69}
ARTIFACT_DIR=${ARTIFACT_DIR:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-69}
RESULT_PATH="$REPO_ROOT/eval/results/thesis/rob69_18l_long_context_finetune_vs_baseline.csv"
SUMMARY_PATH="$ARTIFACT_DIR/rob69-summary.md"
STATUS=failed

cd "$REPO_ROOT"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main/

if [ -f symphony/.env ]; then
  set -a
  source symphony/.env
  set +a
fi

if python scripts/summarize_rob69_results.py --csv "$RESULT_PATH" --markdown-out "$SUMMARY_PATH"; then
  STATUS=completed
fi

python scripts/linear_experiment_callback.py \
  --issue ROB-69 \
  --state Todo \
  --status "$STATUS" \
  --summary-file "$SUMMARY_PATH" \
  --log-path "$ARTIFACT_DIR/eval-${EVAL_JOB_ID:-unknown}.out" \
  --result-path "$RESULT_PATH" \
  --job-id "${EVAL_JOB_ID:-unknown}"

if [ "$STATUS" != completed ]; then
  exit 1
fi
