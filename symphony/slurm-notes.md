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

For long training or evaluation jobs, do not poll more often than needed. If the next action is not likely to be unblocked immediately, wait several minutes between `squeue` checks and use `sacct` after the job leaves the queue.

Keep log reads bounded by default:

```bash
tail -n 80 /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/<job>.out
tail -n 80 /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/<job>.err
grep -n -E "error|traceback|failed|exception|epoch|loss|wer" /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/<job>.out | tail -n 40
```

After completion, inspect stdout/stderr with bounded reads and targeted searches first. Treat nonzero exit codes, failed/cancelled/timeout states, tracebacks, uncaught exceptions, and obvious error lines as validation failures. Read larger log sections only when the bounded output points to a specific failure or metric location.

## Log And Artifact Rules

- Do not commit Slurm output, checkpoints, generated CSV sweeps, datasets, caches, or WandB/local environment files.
- Keep issue-specific job logs under `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts`.
- Keep short-lived temp files under `/mnt/parscratch/users/acp21rjf/symphony-tmp` and remove them before handoff unless they are explicitly needed as validation evidence.
- Record job ID, script path, log path, command purpose, and outcome in the Linear workpad and `symphony/RESEARCH_DIARY.md`.
