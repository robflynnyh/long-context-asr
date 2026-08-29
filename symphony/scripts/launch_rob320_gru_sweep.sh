#!/usr/bin/env bash
set -euo pipefail

ISSUE_ID="${ISSUE_ID:-ROB-320}"
REPO_DIR="${REPO_DIR:-/users/acp21rjf/long-context-asr}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-320}"
SWEEP_NAME="${SWEEP_NAME:-gru_conformer_seq_sweep_3epoch}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ARTIFACT_ROOT}/checkpoints/${SWEEP_NAME}}"
TEMPLATE="${TEMPLATE:-${REPO_DIR}/exp/configs/paper_templates/exp_set_seq_rotary_base_18l_gru_3epoch.yaml}"
GENERATED_DIR="${GENERATED_DIR:-${ARTIFACT_ROOT}/generated/${SWEEP_NAME}}"
LOG_DIR="${LOG_DIR:-${ARTIFACT_ROOT}/logs/${SWEEP_NAME}}"
JOB_IDS_FILE="${JOB_IDS_FILE:-${ARTIFACT_ROOT}/${SWEEP_NAME}_job_ids.tsv}"
BRANCH="${BRANCH:-$(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD)}"
COMMIT="${COMMIT:-$(git -C "${REPO_DIR}" rev-parse HEAD)}"

mkdir -p "${ARTIFACT_ROOT}" "${CHECKPOINT_ROOT}" "${GENERATED_DIR}" "${LOG_DIR}"

available_partitions="$(sinfo -h -o '%P' | tr ',' '\n' | sed 's/*//g' | sort -u)"
candidate_partitions=(gpu-h100-nvl gpu-h100 gpu)
valid_partitions=()
for partition in "${candidate_partitions[@]}"; do
  if grep -qx "${partition}" <<< "${available_partitions}"; then
    valid_partitions+=("${partition}")
  fi
done

if [[ "${#valid_partitions[@]}" -eq 0 ]]; then
  echo "No valid GPU partitions found among: ${candidate_partitions[*]}" >&2
  exit 1
fi

partition_arg="$(IFS=,; echo "${valid_partitions[*]}")"
gres_arg="${ROB320_GRES:-gpu:1}"

cd "${REPO_DIR}/exp"
python run_launcher.py \
  --template "${TEMPLATE}" \
  --mode h100nvl \
  --launch train.py \
  --save_dir "${GENERATED_DIR}" \
  --log_dir "${LOG_DIR}" \
  --job_ids_out "${JOB_IDS_FILE}" \
  --job_name_prefix "${ISSUE_ID}-gru" \
  --deterministic_names \
  --partition "${partition_arg}" \
  --gres "${gres_arg}" \
  --mem 130GB \
  --cpus_per_task 8

dependency_jobs="$(cut -f1 "${JOB_IDS_FILE}" | paste -sd: -)"
if [[ -z "${dependency_jobs}" ]]; then
  echo "No sweep job ids were recorded in ${JOB_IDS_FILE}" >&2
  exit 1
fi

finalizer_sbatch="${ARTIFACT_ROOT}/${SWEEP_NAME}_finalizer.sbatch"
cat > "${finalizer_sbatch}" <<SBATCH
#!/bin/bash
#SBATCH --job-name=${ISSUE_ID}-gru-finalizer
#SBATCH --time=00:20:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --output=${ARTIFACT_ROOT}/logs/${SWEEP_NAME}_finalizer-%j.out
#SBATCH --error=${ARTIFACT_ROOT}/logs/${SWEEP_NAME}_finalizer-%j.err

set -euo pipefail
cd "${REPO_DIR}"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

for env_file in \
  "${REPO_DIR}/symphony/.env" \
  "/users/acp21rjf/long-context-asr/symphony/.env" \
  "\${HOME}/.config/long-context-asr/linear.env" \
  "\${HOME}/.config/sap-longcontext/linear.env"; do
  if [[ -f "\${env_file}" ]]; then
    set -a
    source "\${env_file}"
    set +a
    break
  fi
done

export PYTHONPATH="${REPO_DIR}:\${PYTHONPATH:-}"
python "${REPO_DIR}/symphony/scripts/rob320_sweep_finalizer.py" \
  --issue-id "${ISSUE_ID}" \
  --state-name Todo \
  --repo-dir "${REPO_DIR}" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --jobs-file "${JOB_IDS_FILE}" \
  --log-dir "${LOG_DIR}" \
  --config-template "${TEMPLATE}" \
  --branch "${BRANCH}" \
  --commit "${COMMIT}" \
  --expected-jobs 27
SBATCH

finalizer_output="$(sbatch --dependency=afterany:${dependency_jobs} "${finalizer_sbatch}")"
echo "${finalizer_output}"
echo "sweep_job_ids=${dependency_jobs}"
echo "job_ids_file=${JOB_IDS_FILE}"
echo "finalizer_sbatch=${finalizer_sbatch}"
