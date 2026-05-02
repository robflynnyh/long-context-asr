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

## GPU Queue Guidance

For ROB-26 encoder-decoder RL work, these GPU placement constraints were observed:

- `hp-h100-nvl`, `hp-h100`, and `hp-a100` failed with `Invalid account or account/partition combination specified`.
- `gpu-h100-nvl`, `gpu-h100`, and general `gpu` were valid, but queue estimates varied substantially.

`sbatch --test-only` can be materially optimistic for GPU placement. Treat it as advisory only. If testing a replacement placement is worthwhile, submit one candidate, immediately compare the real `squeue`/`scontrol` estimate against the active job, and cancel the worse pending job before either can start.

Do not submit duplicate training jobs to the same checkpoint output directory. If changing placement, cancel or isolate the pending run first, then record the decision and final job IDs in the Linear workpad. The research diary should summarize the outcome, not every queue probe.

Prefer launchers that make the code revision explicit. Either run from a stable issue-specific worktree or checkout the intended branch/commit inside the batch script before Python starts. At minimum, export `PYTHONPATH` to the intended repo root so jobs do not import an installed `lcasr` package from another checkout.

Avoid preserving raw queue history in repo docs. Keep exact queue probes, transient estimates, and canceled candidate IDs in the Linear workpad only when they explain a live decision.

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
squeue -j <job_id> -o '%i|%j|%T|%R|%S|%M|%l|%P'
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed
```

After completion, inspect stdout/stderr. Treat nonzero exit codes, failed/cancelled/timeout states, tracebacks, uncaught exceptions, and obvious error lines as validation failures.

Keep Slurm inspection output bounded. Use `tail`, `rg`, or narrow `sed -n` slices for logs, and request only the `squeue`/`sacct` fields needed for the decision. Do not stream full logs or broad queue listings into Codex context.

## Log And Artifact Rules

- Do not commit Slurm output, checkpoints, generated CSV sweeps, datasets, caches, or WandB/local environment files.
- Keep issue-specific job logs under `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts`.
- Keep short-lived temp files under `/mnt/parscratch/users/acp21rjf/symphony-tmp` and remove them before handoff unless they are explicitly needed as validation evidence.
- Record job ID, script path, log path, command purpose, and outcome in the Linear workpad. Add only concise outcome summaries to `symphony/RESEARCH_DIARY.md`.
