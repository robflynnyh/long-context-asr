#!/bin/bash
#SBATCH --job-name=rob123-rope-f4
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=80:00:00
#SBATCH --mem=82GB
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/rob123-rope-f4-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/rob123-rope-f4-%j.err

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${ROB123_REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-123}"
REMOTE_BRANCH="${ROB123_REMOTE_BRANCH:-symphony/ROB-123-rope-full-spotify-stanage}"
TARGET_COMMIT="${ROB123_TARGET_COMMIT:?ROB123_TARGET_COMMIT must pin the code revision}"
RUN_ID="${ROB123_RUN_ID:-rob123-rope-full-spotify-factor4-2epoch-delay0p5-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ROB123_ARTIFACT_ROOT:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123}"
RUN_DIR="${ROB123_RUN_DIR:-$ARTIFACT_ROOT/$RUN_ID}"
OUT_LOG="${ROB123_OUT_LOG:-$ARTIFACT_ROOT/rob123-rope-f4-${SLURM_JOB_ID:-manual}.out}"
ERR_LOG="${ROB123_ERR_LOG:-$ARTIFACT_ROOT/rob123-rope-f4-${SLURM_JOB_ID:-manual}.err}"
SUMMARY_FILE="${ROB123_SUMMARY_FILE:-$RUN_DIR/${RUN_ID}.summary.txt}"
PREP_SUMMARY="${ROB123_PREP_SUMMARY:-$RUN_DIR/${RUN_ID}.prep.txt}"
EXECUTED_SCRIPT="${ROB123_EXECUTED_SCRIPT:-$RUN_DIR/$(basename "$0")}"
CALLBACK_SCRIPT="${ROB123_CALLBACK_SCRIPT:-$RUN_DIR/linear_stanage_callback.py}"
BASE_CONFIG="${ROB123_BASE_CONFIG:-exp/configs/streaming_decoder_asr_100m.yaml}"
SOURCE_PAIRS="${ROB123_SOURCE_PAIRS:-/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json}"
RUNTIME_CONFIG="${ROB123_RUNTIME_CONFIG:-$RUN_DIR/streaming_decoder_asr_100m_rope_full_spotify_factor4_2epoch_delay0p5.yaml}"
CHECKPOINT_DIR="${ROB123_CHECKPOINT_DIR:-/mnt/parscratch/users/acp21rjf/spotify/streaming_decoder_asr_100m_rope_rob123_full_spotify_factor4_2epoch_delay0p5_${RUN_ID}}"
PRETRAINED_CHECKPOINT="${ROB123_PRETRAINED_CHECKPOINT:-}"
WANDB_DIR="${ROB123_WANDB_DIR:-$ARTIFACT_ROOT/wandb}"
WANDB_NAME="${ROB123_WANDB_NAME:-rob123_rope_streaming_decoder_full_spotify_factor4_2epoch}"
BATCH_SIZE="${ROB123_BATCH_SIZE:-88}"
MAX_EPOCHS="${ROB123_MAX_EPOCHS:-2}"
LEARNING_RATE="${ROB123_LEARNING_RATE:-3e-4}"
SAVE_EVERY_N_STEPS="${ROB123_SAVE_EVERY_N_STEPS:-50000}"
SUBSAMPLING_FACTOR="${ROB123_SUBSAMPLING_FACTOR:-4}"
ROTARY_BASE_FREQ="${ROB123_ROTARY_BASE_FREQ:-1500000}"
DELAY_SECONDS="${ROB123_DELAY_SECONDS:-0.5}"
DEBUG_GENERATE_EVERY_RECORDS="${ROB123_DEBUG_GENERATE_EVERY_RECORDS:-500}"
DEBUG_GENERATE_MAX_FRAMES="${ROB123_DEBUG_GENERATE_MAX_FRAMES:-0}"
VALIDATE_PATH_LIMIT="${ROB123_VALIDATE_PATH_LIMIT:-50}"
VALIDATE_ALL_PATHS="${ROB123_VALIDATE_ALL_PATHS:-0}"
NUM_WORKERS="${ROB123_NUM_WORKERS:-0}"
PREFETCH="${ROB123_PREFETCH:-1}"
PIN_MEMORY="${ROB123_PIN_MEMORY:-0}"
RESET_STEP="${ROB123_RESET_STEP:-0}"
SKIP_GIT_UPDATE="${ROB123_SKIP_GIT_UPDATE:-0}"
CALLBACK_PYTHON="${ROB123_CALLBACK_PYTHON:-python3}"
LINEAR_KEY_FILE="${ROB123_LINEAR_KEY_FILE:-$ARTIFACT_ROOT/.linear_api_key}"

if [[ "${ROB123_RUN_DIR_EXEC:-0}" != "1" ]]; then
  mkdir -p "$RUN_DIR"
  cp "${BASH_SOURCE[0]}" "$EXECUTED_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_stanage_callback.py" "$CALLBACK_SCRIPT"
  cp "$REPO_DIR/symphony/scripts/linear_job_callback.py" "$RUN_DIR/linear_job_callback.py"
  chmod +x "$EXECUTED_SCRIPT" "$CALLBACK_SCRIPT"
  export ROB123_RUN_DIR_EXEC=1
  export ROB123_REPO_DIR="$REPO_DIR"
  export ROB123_REMOTE_BRANCH="$REMOTE_BRANCH"
  export ROB123_TARGET_COMMIT="$TARGET_COMMIT"
  export ROB123_RUN_ID="$RUN_ID"
  export ROB123_ARTIFACT_ROOT="$ARTIFACT_ROOT"
  export ROB123_RUN_DIR="$RUN_DIR"
  export ROB123_OUT_LOG="$OUT_LOG"
  export ROB123_ERR_LOG="$ERR_LOG"
  export ROB123_SUMMARY_FILE="$SUMMARY_FILE"
  export ROB123_PREP_SUMMARY="$PREP_SUMMARY"
  export ROB123_EXECUTED_SCRIPT="$EXECUTED_SCRIPT"
  export ROB123_CALLBACK_SCRIPT="$CALLBACK_SCRIPT"
  exec bash "$EXECUTED_SCRIPT" "$@"
fi

mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR" "$WANDB_DIR"

on_exit() {
  local status=$?
  set +e
  {
    echo "ended_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_code=${status}"
    echo "slurm_job_id=${SLURM_JOB_ID:-manual}"
    echo "run_id=${RUN_ID}"
    echo "host=$(hostname)"
    echo "repo_dir=${REPO_DIR}"
    echo "remote_branch=${REMOTE_BRANCH}"
    echo "target_commit=${TARGET_COMMIT}"
    echo "runtime_config=${RUNTIME_CONFIG}"
    echo "source_pairs=${SOURCE_PAIRS}"
    echo "checkpoint_dir=${CHECKPOINT_DIR}"
    echo "pretrained_checkpoint=${PRETRAINED_CHECKPOINT}"
    echo "wandb_dir=${WANDB_DIR}"
    echo "wandb_name=${WANDB_NAME}"
    echo "batch_size=${BATCH_SIZE}"
    echo "max_epochs=${MAX_EPOCHS}"
    echo "learning_rate=${LEARNING_RATE}"
    echo "save_every_n_steps=${SAVE_EVERY_N_STEPS}"
    echo "subsampling_factor=${SUBSAMPLING_FACTOR}"
    echo "use_rotary=true"
    echo "rotary_base_freq=${ROTARY_BASE_FREQ}"
    echo "delay_seconds=${DELAY_SECONDS}"
    echo "reset_step=${RESET_STEP}"
    echo "skip_git_update=${SKIP_GIT_UPDATE}"
    echo "slurm_partition=${SLURM_JOB_PARTITION:-unset}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  } >> "$SUMMARY_FILE"
  if [[ "${ROB123_ENABLE_CALLBACK:-1}" == "1" ]]; then
    if [[ -z "${LINEAR_API_KEY:-}" && -f "$LINEAR_KEY_FILE" ]]; then
      export LINEAR_API_KEY
      LINEAR_API_KEY="$(cat "$LINEAR_KEY_FILE")"
    fi
    callback_args=(
      "$CALLBACK_SCRIPT"
      --issue-id ROB-123
      --state-name Todo
      --exit-code "$status"
      --log-out "$OUT_LOG"
      --log-err "$ERR_LOG"
      --artifact-path "$RUN_DIR"
      --checkpoint-path "$CHECKPOINT_DIR"
      --summary-file "$SUMMARY_FILE"
      --title "ROB-123 Stanage full-Spotify RoPE factor-4 training finished"
      --job-label "Stanage GPU training"
      --metadata "branch=${REMOTE_BRANCH}"
      --metadata "commit=${TARGET_COMMIT}"
      --metadata "runtime_config=${RUNTIME_CONFIG}"
      --metadata "source_pairs=${SOURCE_PAIRS}"
      --metadata "pretrained_checkpoint=${PRETRAINED_CHECKPOINT}"
      --metadata "wandb_name=${WANDB_NAME}"
      --metadata "subsampling_factor=${SUBSAMPLING_FACTOR}"
      --metadata "delay_seconds=${DELAY_SECONDS}"
    )
    if [[ "${ROB123_CALLBACK_DRY_RUN:-0}" == "1" ]]; then
      callback_args+=(--dry-run)
    fi
    "$CALLBACK_PYTHON" "${callback_args[@]}" >> "$SUMMARY_FILE" 2>&1
  fi
  return "$status"
}
trap on_exit EXIT
trap 'trap - TERM INT; exit 143' TERM
trap 'trap - TERM INT; exit 130' INT

{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "slurm_job_id=${SLURM_JOB_ID:-manual}"
  echo "run_id=${RUN_ID}"
  echo "host=$(hostname)"
  echo "executed_script=${EXECUTED_SCRIPT}"
  echo "callback_script=${CALLBACK_SCRIPT}"
} > "$SUMMARY_FILE"

if [[ "${ROB123_CALLBACK_ONLY:-0}" == "1" ]]; then
  echo "callback-only dry run requested"
  exit 0
fi

cd "$REPO_DIR"
if [[ "$SKIP_GIT_UPDATE" == "1" ]]; then
  current_commit="$(git rev-parse HEAD)"
  if [[ "$current_commit" != "$TARGET_COMMIT" ]]; then
    echo "ROB123_SKIP_GIT_UPDATE=1 but repo is at $current_commit, expected $TARGET_COMMIT" >&2
    exit 2
  fi
else
  git fetch origin "$REMOTE_BRANCH"
  git checkout --detach "$TARGET_COMMIT"
fi
export PYTHONPATH="$PWD"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

prep_args=(
  --base-config "$BASE_CONFIG"
  --source-pairs "$SOURCE_PAIRS"
  --config-out "$RUNTIME_CONFIG"
  --summary-out "$PREP_SUMMARY"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --wandb-dir "$WANDB_DIR"
  --wandb-name "$WANDB_NAME"
  --batch-size "$BATCH_SIZE"
  --max-epochs "$MAX_EPOCHS"
  --learning-rate "$LEARNING_RATE"
  --save-every-n-steps "$SAVE_EVERY_N_STEPS"
  --subsampling-factor "$SUBSAMPLING_FACTOR"
  --rotary-base-freq "$ROTARY_BASE_FREQ"
  --delay-seconds "$DELAY_SECONDS"
  --debug-generate-every-records "$DEBUG_GENERATE_EVERY_RECORDS"
  --debug-generate-max-frames "$DEBUG_GENERATE_MAX_FRAMES"
  --validate-path-limit "$VALIDATE_PATH_LIMIT"
)
if [[ -n "$PRETRAINED_CHECKPOINT" ]]; then
  prep_args+=(--pretrained-checkpoint "$PRETRAINED_CHECKPOINT")
fi
if [[ "$VALIDATE_ALL_PATHS" == "1" ]]; then
  prep_args+=(--validate-all-paths)
fi
python symphony/scripts/prepare_rob123_stanage_rope_full_spotify.py "${prep_args[@]}"

cat "$PREP_SUMMARY" >> "$SUMMARY_FILE"

train_args=(
  -config "$RUNTIME_CONFIG"
  -num_workers "$NUM_WORKERS"
  -prefetch "$PREFETCH"
)
if [[ "$RESET_STEP" == "1" ]]; then
  train_args+=(-reset_step)
fi
if [[ "$PIN_MEMORY" == "1" ]]; then
  train_args+=(-pin_memory)
fi
if [[ -n "${ROB123_MAX_RECORDS:-}" ]]; then
  train_args+=(-max_records "$ROB123_MAX_RECORDS")
fi
if [[ -n "${ROB123_MAX_STEPS:-}" ]]; then
  train_args+=(-max_steps "$ROB123_MAX_STEPS")
fi
if [[ "${ROB123_DISABLE_WANDB:-0}" == "1" ]]; then
  train_args+=(-disable_wandb)
fi

python exp/train_streaming_decoder_asr.py "${train_args[@]}"
