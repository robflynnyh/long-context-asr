# Slurm Notes

Use Slurm for training, ASR evaluation, preprocessing, large scans, metric extraction over large files, and anything likely to run for more than a few minutes. Repository inspection, patching, and small syntax/metadata checks are fine on the login node.

## Observed Launch Patterns

Training launch strings are embedded in `exp/run_launcher.py`:

- A100: `#SBATCH --partition=gpu`, `#SBATCH --gres=gpu:1`, `#SBATCH --qos=gpu`, `#SBATCH --mem=82GB`, `#SBATCH --cpus-per-task=8`.
- H100: `#SBATCH --partition=gpu-h100`, `#SBATCH --gres=gpu:1`, `#SBATCH --qos=gpu`, `#SBATCH --mem=100GB`, `#SBATCH --cpus-per-task=16`.
- H100 NVL: `#SBATCH --partition=gpu-h100-nvl`, `#SBATCH --gres=gpu:h100:1`, `#SBATCH --qos=gpu`, `#SBATCH --mem=130GB`, `#SBATCH --cpus-per-task=8`.

Evaluation launchers:

- `eval/run_eval_a100.sh`: `gpu`, one GPU, 80 GB, 4 CPUs, 90 hours.
- `eval/run_eval_h100.sh`: `gpu-h100-nvl`, one H100, 80 GB, 4 CPUs, 20 hours.
- `eval/run_eval_cpu.sh`: CPU-only, 60 GB, 16 CPUs, 90 hours.

Preprocessing launchers under `job_scripts/preprocess/` are CPU jobs that call `python -m lcasr.utils.preprocess --ogg_path <path> --stage 0`.

## ROB-26 GPU Queue Triage

For the ROB-26 encoder-decoder RL run, `gpu-h100-nvl` was the best valid placement observed before launch on 2026-05-01. `sbatch --test-only` checks showed:

- `hp-h100-nvl`, `hp-h100`, and `hp-a100` failed with `Invalid account or account/partition combination specified`.
- `gpu-h100` was valid but estimated later than the existing `gpu-h100-nvl` job.
- General `gpu` / A100 placement was valid but estimated much later.

Do not submit duplicate training jobs to the same checkpoint output directory. If changing placement, cancel the pending job first and record the old/new job IDs in Linear and the diary.

On a 2026-05-01 21:32 BST resume check, queued job `10094511` had scheduler estimate `2026-05-02T08:58:11`. Fresh `sbatch --test-only --time=06:00:00` checks estimated `gpu-h100` at `2026-05-02T07:58:19`, `gpu-h100-nvl` at `2026-05-03T14:03:22`, and general `gpu` at `2026-05-27T04:03:37`. The possible `gpu-h100` improvement was only about one hour and estimate volatility made cancel/requeue unattractive, so keep `10094511` queued unless a materially better valid placement appears.

On a 2026-05-01 22:03 BST follow-up, direct submission to `gpu-h100` with the 24h limit produced job `10097296`, but its actual scheduler estimate was `2026-05-02T19:25:00`, later than `10094511`. Job `10097296` was immediately canceled before start (`CANCELLED`, elapsed `00:00:00`), so `10094511` remains the active ROB-26 training job.

On a 2026-05-01 22:11 BST follow-up, `10094511` still estimated `2026-05-02T08:58:11`. Fresh `sbatch --test-only` checks for 24h, 12h, and 6h jobs estimated `gpu-h100-nvl` at `2026-05-12T05:03:22` and `gpu-h100` at `2026-05-03T03:58:19`, both worse than the existing active job. Keep `10094511` queued.

## Minimal Symphony Job Template

Put custom job scripts and logs under parscratch for issue work:

```bash
#!/bin/bash
#SBATCH --job-name=rob24-check
#SBATCH --time=00:10:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=2
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-24-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-24-%j.err

set -euo pipefail
cd /mnt/parscratch/users/acp21rjf/symphony-workspaces/ROB-24

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

python -m compileall lcasr exp eval
```

Submit and monitor:

```bash
mkdir -p /mnt/parscratch/users/acp21rjf/symphony-job-artifacts
sbatch /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-24-check.sbatch
squeue -j <job_id>
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed
```

After completion, inspect stdout/stderr. Treat nonzero exit codes, failed/cancelled/timeout states, tracebacks, uncaught exceptions, and obvious error lines as validation failures.

## Log And Artifact Rules

- Do not commit Slurm output, checkpoints, generated CSV sweeps, datasets, caches, or WandB/local environment files.
- Keep issue-specific job logs under `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts`.
- Keep short-lived temp files under `/mnt/parscratch/users/acp21rjf/symphony-tmp` and remove them before handoff unless they are explicitly needed as validation evidence.
- Record job ID, script path, log path, command purpose, and outcome in the Linear workpad and `symphony/RESEARCH_DIARY.md`.
