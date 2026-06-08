#!/bin/bash
#SBATCH --job-name=rob209-callback
#SBATCH --time=00:10:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-209/rob209-callback-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-209/rob209-callback-%j.err

set -Eeuo pipefail

export ROB209_REPO_DIR="${ROB209_REPO_DIR:-/users/acp21rjf/long-context-asr}"
export ROB209_REMOTE_BRANCH="${ROB209_REMOTE_BRANCH:-symphony/ROB-209-ordered-floras-bounded-history}"

resolve_rob209_target_commit() {
  local target="${ROB209_TARGET_COMMIT:-}"
  if [[ -z "$target" ]]; then
    target="$(git -C "$ROB209_REPO_DIR" rev-parse "origin/${ROB209_REMOTE_BRANCH}" 2>/dev/null || true)"
  fi
  if [[ -z "$target" ]]; then
    target="$(git -C "$ROB209_REPO_DIR" rev-parse HEAD 2>/dev/null || true)"
  fi
  if [[ -z "$target" ]]; then
    target="$(git -C "$ROB209_REPO_DIR" ls-remote origin "refs/heads/${ROB209_REMOTE_BRANCH}" 2>/dev/null | awk '{print $1}' || true)"
  fi
  if [[ -z "$target" ]]; then
    echo "Unable to resolve ROB209_TARGET_COMMIT for ${ROB209_REMOTE_BRANCH}" >&2
    exit 2
  fi
  printf '%s\n' "$target"
}

export ROB209_TARGET_COMMIT="$(resolve_rob209_target_commit)"
export ROB209_RUN_ID="${ROB209_RUN_ID:-rob209-callback-only-$(date -u +%Y%m%dT%H%M%SZ)}"
export ROB209_CALLBACK_ONLY=1
export ROB209_CALLBACK_DRY_RUN="${ROB209_CALLBACK_DRY_RUN:-1}"

bash "$ROB209_REPO_DIR/symphony/scripts/run_rob209_ordered_floras_stanage.sh"
