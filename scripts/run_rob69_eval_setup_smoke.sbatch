#!/bin/bash
#SBATCH --job-name=rob69-smoke
#SBATCH --time=00:30:00
#SBATCH --partition=interactive
#SBATCH --mem=48GB
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-69/smoke-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-69/smoke-%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-69}
cd "$REPO_ROOT"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main/

python scripts/prepare_rob69_seed_results.py --force
python scripts/check_rob69_eval_setup.py --repo-root "$REPO_ROOT"
