# Agent Notes

This repo is research code for long-context ASR experiments. Most operational knowledge is encoded in Python scripts, YAML templates, and Slurm launchers rather than in prose docs. Start by reading the specific script you plan to run.

## Repo Map

- `lcasr/`: importable library used by training and evaluation code. Key areas are `models/`, `components/`, `eval/`, `decoding/`, and `utils/`.
- `exp/`: training entrypoints and experiment templates.
  - `exp/train.py` is the main acoustic-model training script for `SCConformerXL`.
  - `exp/train_files/train_enc_dec.py` is the main encoder-decoder training script.
  - `exp/run_launcher.py` expands YAML templates into run configs and submits Slurm jobs.
  - `exp/run_restarter.py` resubmits generated runs from `.tmp/`.
- `eval/`: evaluation entrypoints, configs, and historical result CSV/PDF files.
  - `eval/eval_manager.py` runs model/dataset grids from YAML.
  - `eval/run.py` is the generic evaluation module used by `eval/eval_manager.py`.
  - Dataset-specific modules live under `eval/<dataset>/run.py`.
- `job_scripts/preprocess/`: Spotify OGG-to-mel preprocessing array script.
- `symphony/`: Symphony workflow and future-agent notes. Do not put large artifacts here.

## Environment

The Symphony workflow starts Codex with:

```bash
module load conda_alma9_container/v1
eval "$(conda shell.bash hook)"
conda activate /mnt/parscratch/users/acp21rjf/conda/main
export TMPDIR=/mnt/parscratch/users/acp21rjf/symphony-tmp
```

Older repo Slurm scripts usually use:

```bash
module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main
```

Some legacy scripts use `source activate a100`; prefer the full conda path from `symphony/WORKFLOW.md` when creating new Symphony-specific jobs.

## Local Install And Imports

The package is defined by `setup.py` and imports as `lcasr`. For Python checks in a fresh environment, install from the repo root:

```bash
pip install -e .
```

`requirements.txt` is minimal. The code also imports packages that may be supplied by the cluster environment, including `torchaudio`, `sentencepiece`, `librosa`, `wandb`, `whisper`, `pyctcdecode`, `pandas`, and `omegaconf`.

## Data And Paths

Many scripts contain Rob-specific parscratch defaults. Before running training or evaluation, inspect the target config and dataset module for absolute paths.

- Training configs usually read Spotify pairs from `/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json`.
- Checkpoint configs commonly write under `/mnt/parscratch/users/acp21rjf/spotify/checkpoints*`.
- `eval/paths.yaml` is gitignored. Start from `eval/paths_template.yaml` for Earnings-22-style keys, but inspect the target dataset module because the template is not exhaustive.
- Several dataset modules still use hard-coded defaults directly. Check `eval/<dataset>/run.py` before assuming `eval/paths.yaml` is honored.

## Symphony Handoff Hygiene

`symphony/WORKFLOW.md` is the authoritative workflow. Repo-specific reminders:

- Use `/mnt/parscratch/users/acp21rjf/symphony-tmp` for temporary files and `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts` for job logs/artifacts. Keep large outputs, checkpoints, downloads, logs, and validation artifacts out of the repo.
- Keep command output bounded with targeted `rg`, `head`, `tail`, `sed -n`, filtered Slurm fields, and `git diff --stat`. Summarize large artifacts and reference paths instead of loading raw output into context.
- Use `symphony/RESEARCH_DIARY.md` for concise outcome summaries only. Put routine queue checks, repeated resume observations, and detailed troubleshooting trails in the Linear workpad when needed.
- After launching long-running GPU training or evaluation, hand off with job IDs, log paths, expected outputs, and follow-up commands, then stop.
- For docs-only edits, lightweight validation such as `git diff --check` is usually enough.
