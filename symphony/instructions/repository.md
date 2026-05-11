# Repository

`lcasr/` is the importable Python package containing model code, decoding,
evaluation helpers, data loading, and general utilities.

`exp/` is the training harness. Important entry points include `exp/train.py`,
`exp/train_files/train_enc_dec.py`, `exp/train_files/train_enc_dec_rl.py`,
`exp/run_launcher.py`, and `exp/run_restarter.py`.

`exp/configs/` contains training configs and templates. `exp/run_launcher.py`
expands templates into concrete configs and submits Slurm jobs.

`eval/` is the evaluation harness. `eval/eval_manager.py` runs grids from YAML,
`eval/run.py` is the generic evaluation module, and dataset-specific modules
live under `eval/<dataset>/run.py`.

`job_scripts/preprocess/` contains preprocessing launchers and array scripts.
Read the relevant launcher before changing a run or starting a long job.

`symphony/` contains the Symphony workflow and future-agent notes. Keep
Symphony-specific instructions and runtime config there. `symphony/.env` is
local-only and must not be committed.

`symphony/slurm-notes.md`, `symphony/training-notes.md`, `symphony/eval-notes.md`,
and `symphony/agent-notes.md` record reusable repo procedures. Read the relevant
note before launching or editing that area.

Many configs and scripts contain Rob-specific absolute paths. Before running
training or evaluation, inspect the target config, launcher, and dataset module
for checkpoint, data, output, and `PYTHONPATH` assumptions.

Keep credentials, raw data, large checkpoints, Slurm logs, W&B output, caches,
and bulky temporary files out of Git.

Commit small, meaningful result artifacts when they are part of the requested
deliverable and are reasonable for Git. For large generated artifacts, commit a
small index or summary recording the external path, size, generation command,
and why the artifact was not committed.

Append concise dated entries to `symphony/RESEARCH_DIARY.md` for meaningful
project changes, experiment launches, completed runs, fixes, validation
outcomes, PR handoffs, and interpretation updates. Do not add repetitive launch
bookkeeping that will not help a future agent.
