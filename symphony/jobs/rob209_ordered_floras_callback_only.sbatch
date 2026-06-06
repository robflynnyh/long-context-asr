#!/bin/bash
#SBATCH --job-name=rob209-callback
#SBATCH --time=00:10:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-209/rob209-callback-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-209/rob209-callback-%j.err

set -Eeuo pipefail

export ROB209_RUN_ID="${ROB209_RUN_ID:-rob209-callback-only-$(date -u +%Y%m%dT%H%M%SZ)}"
export ROB209_CALLBACK_ONLY=1
export ROB209_CALLBACK_DRY_RUN=1

bash "${ROB209_REPO_DIR:-/users/acp21rjf/long-context-asr}/symphony/scripts/run_rob209_ordered_floras_stanage.sh"
