# Symphony

This directory contains Symphony runner configuration and lightweight operational notes for the long-context-asr Linear project.

- `WORKFLOW.md`: Symphony workflow used to launch agents for Linear issues.
- `instructions/`: binding per-agent instructions loaded by `WORKFLOW.md`.
- `RESEARCH_DIARY.md`: concise dated notes from Symphony-managed work.
- `agent-notes.md`: quick repo map, setup expectations, and handoff hygiene.
- `training-notes.md`: training config/template launch flow and checkpoint caveats.
- `eval-notes.md`: evaluation entrypoints, config shape, dataset path handling, and outputs.
- `slurm-notes.md`: cluster launch patterns observed in repo scripts.
- Additional focused notes may be added here when agents discover non-obvious repo procedures, Slurm patterns, environment setup, data/checkpoint locations, evaluation commands, or recurring failure modes.

Do not store credentials, raw data, checkpoints, large logs, or generated artifacts here. `symphony/.env` is local-only and ignored by Git. Put large non-commit files under parscratch and reference their paths.
