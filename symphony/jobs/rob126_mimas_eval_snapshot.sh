#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${ROB126_REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
ARTIFACT_ROOT="${ROB126_ARTIFACT_ROOT:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-126}"
SOURCE_RUN_ID="${ROB126_SOURCE_RUN_ID:-rob126-full-cleanprep-top2-b32-4epoch-20260523T160938Z}"
SOURCE_RUN_DIR="${ROB126_SOURCE_RUN_DIR:-$ARTIFACT_ROOT/$SOURCE_RUN_ID}"
EVAL_RUN_ID="${ROB126_EVAL_RUN_ID:-${SOURCE_RUN_ID}-snapshot-eval-$(date -u +%Y%m%dT%H%M%SZ)}"
EVAL_RUN_DIR="${ROB126_EVAL_RUN_DIR:-$ARTIFACT_ROOT/$EVAL_RUN_ID}"
OUT_LOG="${ROB126_EVAL_OUT_LOG:-$EVAL_RUN_DIR/${EVAL_RUN_ID}.out}"
ERR_LOG="${ROB126_EVAL_ERR_LOG:-$EVAL_RUN_DIR/${EVAL_RUN_ID}.err}"
SUMMARY_FILE="${ROB126_EVAL_SUMMARY_FILE:-$EVAL_RUN_DIR/${EVAL_RUN_ID}.summary.txt}"
RESULT_SUMMARY="${ROB126_EVAL_RESULT_SUMMARY:-$EVAL_RUN_DIR/OUTCOME.md}"
SOURCE_MANIFEST="${ROB126_SOURCE_MANIFEST:-$SOURCE_RUN_DIR/run_manifest.json}"
SNAPSHOT_MANIFEST="${ROB126_SNAPSHOT_MANIFEST:-$EVAL_RUN_DIR/run_manifest.snapshot.json}"
EVAL_CONFIG="${ROB126_EVAL_CONFIG:-$EVAL_RUN_DIR/rob126_snapshot_eval.yaml}"
RESULT_CSV="${ROB126_RESULT_CSV:-$EVAL_RUN_DIR/tedlium_test_results.csv}"
WITH_GPU="${ROB126_WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"
GPU_POOL="${ROB126_GPU_POOL:-1,2}"
GPU_IDLE_SECONDS="${ROB126_GPU_IDLE_SECONDS:-30}"
GPU_STAGE="${ROB126_EVAL_GPU_STAGE:-0}"
CALLBACK_SCRIPT="${ROB126_CALLBACK_SCRIPT:-$REPO_DIR/symphony/scripts/linear_mimas_callback.py}"
CALLBACK_PYTHON="${ROB126_CALLBACK_PYTHON:-python3}"
LINEAR_KEY_FILE="${ROB126_LINEAR_KEY_FILE:-$ARTIFACT_ROOT/.linear_api_key}"
LINEAR_FALLBACK_KEY_FILE="${ROB126_LINEAR_FALLBACK_KEY_FILE:-/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-119/.linear_api_key}"

mkdir -p "$EVAL_RUN_DIR" "$EVAL_RUN_DIR/tmp" "$EVAL_RUN_DIR/xdg-cache"

if [[ "${ROB126_EVAL_LOGGING_CONFIGURED:-0}" != "1" ]]; then
  export ROB126_EVAL_LOGGING_CONFIGURED=1
  exec > >(tee -a "$OUT_LOG") 2> >(tee -a "$ERR_LOG" >&2)
fi

on_exit() {
  local status=$?
  set +e
  {
    echo "ended_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_code=${status}"
    echo "eval_run_id=${EVAL_RUN_ID}"
    echo "source_run_id=${SOURCE_RUN_ID}"
    echo "source_manifest=${SOURCE_MANIFEST}"
    echo "snapshot_manifest=${SNAPSHOT_MANIFEST}"
    echo "eval_config=${EVAL_CONFIG}"
    echo "result_csv=${RESULT_CSV}"
    echo "result_summary=${RESULT_SUMMARY}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "gpu_stage=${GPU_STAGE}"
  } >> "$SUMMARY_FILE"
  if [[ "${ROB126_ENABLE_CALLBACK:-1}" == "1" ]]; then
    if [[ -z "${LINEAR_API_KEY:-}" ]]; then
      if [[ -f "$LINEAR_KEY_FILE" ]]; then
        export LINEAR_API_KEY
        LINEAR_API_KEY="$(cat "$LINEAR_KEY_FILE")"
      elif [[ -f "$LINEAR_FALLBACK_KEY_FILE" ]]; then
        export LINEAR_API_KEY
        LINEAR_API_KEY="$(cat "$LINEAR_FALLBACK_KEY_FILE")"
      fi
    fi
    callback_args=(
      --issue-id ROB-126
      --state-name Todo
      --job-id "$EVAL_RUN_ID"
      --job-label "Mimas snapshot eval"
      --exit-code "$status"
      --log-out "$OUT_LOG"
      --log-err "$ERR_LOG"
      --output-path "$EVAL_RUN_DIR"
      --summary-file "$RESULT_SUMMARY"
      --title "ROB-126 interrupted top-layer probe snapshot eval finished"
    )
    if [[ "${ROB126_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
      callback_args+=(--dry-run)
    fi
    "$CALLBACK_PYTHON" "$CALLBACK_SCRIPT" "${callback_args[@]}" >> "$SUMMARY_FILE" 2>&1
  fi
  return "$status"
}
trap on_exit EXIT
trap 'trap - TERM INT; exit 143' TERM
trap 'trap - INT; exit 130' INT

{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "eval_run_id=${EVAL_RUN_ID}"
  echo "source_run_id=${SOURCE_RUN_ID}"
  echo "repo_dir=${REPO_DIR}"
  echo "commit=$(cd "$REPO_DIR" && git rev-parse HEAD)"
  echo "source_manifest=${SOURCE_MANIFEST}"
  echo "snapshot_manifest=${SNAPSHOT_MANIFEST}"
  echo "gpu_pool=${GPU_POOL}"
  echo "gpu_stage=${GPU_STAGE}"
} > "$SUMMARY_FILE"

if [[ "${ROB126_CALLBACK_ONLY:-0}" == "1" ]]; then
  echo "callback-only dry run requested"
  exit 0
fi

cd "$REPO_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TMPDIR="${TMPDIR:-$EVAL_RUN_DIR/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$EVAL_RUN_DIR/xdg-cache}"

if [[ "$GPU_STAGE" != "1" ]]; then
  echo "Acquiring GPU through with-gpu pool ${GPU_POOL} for snapshot eval."
  export ROB126_EVAL_GPU_STAGE=1
  ROB126_ENABLE_CALLBACK=0 "$WITH_GPU" "$GPU_POOL" --num 1 --idle-seconds "$GPU_IDLE_SECONDS" -- bash "${BASH_SOURCE[0]}"
  exit $?
fi

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "1" && "${CUDA_VISIBLE_DEVICES:-}" != "2" ]]; then
  echo "ROB-126 snapshot eval must run on Mimas GPU 1 or 2; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 2
fi

python - "$SOURCE_MANIFEST" "$SNAPSHOT_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
manifest = json.loads(source.read_text(encoding="utf-8"))
for run in manifest["runs"]:
    checkpoint_dir = Path(run["checkpoint_dir"])
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
    if not checkpoints:
        raise SystemExit(f"no checkpoints found in {checkpoint_dir}")
    latest = checkpoints[-1]
    step = int(latest.stem.split("_")[1])
    run["max_epochs"] = f"interrupted epoch 0, step {step}"
    run["snapshot_checkpoint"] = str(latest)
manifest["max_epochs"] = "interrupted snapshot"
manifest["snapshot_note"] = "Human-stopped ROB-126 full run before epoch completion; eval uses the latest retained checkpoint."
target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(target)
PY

python symphony/scripts/make_rob126_eval_config.py \
  --run-manifest "$SNAPSHOT_MANIFEST" \
  --eval-config-out "$EVAL_CONFIG" \
  --csv-out "$RESULT_CSV" \
  --include-per-recording

(
  cd eval
  python eval_manager.py -config "$EVAL_CONFIG"
)

python symphony/scripts/summarize_rob126_results.py \
  --run-manifest "$SNAPSHOT_MANIFEST" \
  --csv "$RESULT_CSV" \
  --summary-out "$RESULT_SUMMARY"

{
  echo ""
  echo "Snapshot note: this is an interrupted checkpoint evaluation, not a completed 4-epoch run."
  echo "Retained checkpoint: $(python - "$SNAPSHOT_MANIFEST" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1]))
print(manifest["runs"][0].get("snapshot_checkpoint", "unknown"))
PY
)"
} >> "$RESULT_SUMMARY"

cat "$RESULT_SUMMARY"
