#!/usr/bin/env bash
set -euo pipefail

ISSUE_ID="${ROB128_ISSUE_ID:-ROB-128}"
RUN_ID="${ROB128_RUN_ID:-rob128-overfit-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ROB128_ARTIFACT_ROOT:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-128}"
RUN_DIR="${ARTIFACT_ROOT}/${RUN_ID}"
CHECKPOINT_DIR="${ARTIFACT_ROOT}/checkpoints/${RUN_ID}"
LOG_OUT="${RUN_DIR}/${RUN_ID}.out"
LOG_ERR="${RUN_DIR}/${RUN_ID}.err"
SUMMARY_PATH="${RUN_DIR}/OUTCOME.md"
RUN_MANIFEST="${RUN_DIR}/run_manifest.json"
CONFIG_PATH="${RUN_DIR}/rob128_${RUN_ID}_config.yaml"
EVAL_JSON="${RUN_DIR}/overfit_eval.json"

mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}"
exec > >(tee -a "${LOG_OUT}") 2> >(tee -a "${LOG_ERR}" >&2)

if [[ -f symphony/.env ]]; then
  set -a
  source symphony/.env
  set +a
fi

callback() {
  local exit_code="$1"
  if [[ "${ROB128_ENABLE_CALLBACK:-0}" != "1" ]]; then
    return 0
  fi
  local dry_args=()
  if [[ "${ROB128_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
    dry_args+=(--dry-run)
  fi
  python3 symphony/scripts/linear_mimas_callback.py \
    --issue-id "${ISSUE_ID}" \
    --state-name "${ROB128_CALLBACK_STATE:-Todo}" \
    --exit-code "${exit_code}" \
    --job-id "${RUN_ID}" \
    --screen-session "${STY:-}" \
    --log-out "${LOG_OUT}" \
    --log-err "${LOG_ERR}" \
    --output-path "${RUN_DIR}" \
    --artifact-path "${SUMMARY_PATH}" \
    --checkpoint-path "${CHECKPOINT_DIR}" \
    --summary-file "${SUMMARY_PATH}" \
    --title "ROB-128 supervised-feature probe overfit finished" \
    --metadata "Run=${RUN_ID}" \
    "${dry_args[@]}"
}
trap 'status=$?; callback "$status"; exit "$status"' EXIT

echo "ROB-128 run id: ${RUN_ID}"
echo "Run dir: ${RUN_DIR}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Git branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Git commit: $(git rev-parse HEAD)"

prepare_args=()
if [[ -n "${ROB128_RECORD_ID:-}" ]]; then
  prepare_args+=(--record-id "${ROB128_RECORD_ID}")
fi

python3 symphony/scripts/prepare_rob128_overfit_config.py \
  --run-dir "${RUN_DIR}" \
  --config-out "${CONFIG_PATH}" \
  --run-manifest-out "${RUN_MANIFEST}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --head "${ROB128_HEAD:-bilstm}" \
  --max-epochs "${ROB128_MAX_EPOCHS:-60}" \
  --learning-rate "${ROB128_LR:-0.001}" \
  --batch-size "${ROB128_BATCH_SIZE:-1}" \
  --seq-len "${ROB128_SEQ_LEN:-2048}" \
  "${prepare_args[@]}"

if [[ "${ROB128_DRY_RUN:-0}" == "1" ]]; then
  echo "ROB128_DRY_RUN=1; prepared config only."
  cat > "${SUMMARY_PATH}" <<EOF
# ROB-128 Dry Run

Prepared config: \`${CONFIG_PATH}\`
Run manifest: \`${RUN_MANIFEST}\`
EOF
  exit 0
fi

python3 exp/train.py \
  -config "${CONFIG_PATH}" \
  --remove_scheduler \
  --reset_step \
  --no_resume \
  --disable_wandb \
  --num_workers 0 \
  --prefetch_factor 1

python3 symphony/scripts/eval_rob128_manifest_probe.py \
  --run-manifest "${RUN_MANIFEST}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --output-json "${EVAL_JSON}" \
  --disable-flash-attention

python3 symphony/scripts/summarize_rob128_overfit.py \
  --run-manifest "${RUN_MANIFEST}" \
  --eval-json "${EVAL_JSON}" \
  --summary-out "${SUMMARY_PATH}"

cat "${SUMMARY_PATH}"
