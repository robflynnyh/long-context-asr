# Symphony Research Diary

This diary is for concise, durable notes from Symphony-managed work on this repository.

Agents should append dated entries for meaningful implementation changes, experiment launches, completed Slurm jobs, failed runs, fixes, validation outcomes, PR handoffs, and interpretation updates. Keep large logs and artifacts in parscratch and reference their paths here.

## 2026-05-01

- ROB-24 on branch `symphony/ROB-24-agent-docs`: identified sparse/scattered operational docs, added focused future-agent notes under `symphony/` for repo orientation, training, evaluation, and Slurm patterns. No Slurm jobs launched.
- ROB-24 PR handoff: opened GitHub PR https://github.com/robflynnyh/long-context-asr/pull/2 against `dev` after docs validation.
- ROB-25 on branch `symphony/ROB-25-benchmark-encdec`: added `eval/run.py` support for config-driven encoder-decoder `transcribe_kwargs`, fixed `EncDecSconformerV2.transcribe` single-sequence unwrapping, and added TEDLIUM test config `eval/eval_configs/rob25_enc_dec_tedlium.yaml` with `overlap_ratio: 0`.
- ROB-25 Slurm validation: environment check job `10094097` completed successfully in 56s; logs confirm conda torch `2.5.1+cu118`, 7 TEDLIUM benchmark models, `overlap_ratio` `[0.0]`, and no missing checkpoint paths.
- ROB-25 Slurm validation: interactive CPU smoke job `10094080` completed successfully in 2m59s using `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-25/smoke_enc_dec_tedlium.yaml`; smoke CSV at `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-25/smoke_enc_dec_tedlium.csv`.
- ROB-25 Slurm cleanup: cancelled superseded pending/incomplete smoke jobs `10093975`, `10094010`, `10094019`, `10094027`, and `10094081` after the environment check and interactive smoke passed.
- ROB-25 Slurm launch: submitted full TEDLIUM-test H100 benchmark job `10094147` (`rob25-full`) using `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-25/run_eval_manager.sbatch`; output CSV target `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-25/results/enc_dec_tedlium_test.csv`, logs `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-25/rob25-full-10094147.out` and `.err`.
- ROB-25 docs: updated `symphony/eval-notes.md` with encoder-decoder `transcribe_kwargs` usage and the `overlap_ratio: 0` caveat for encoder-decoder benchmarks.
- ROB-25 Slurm cleanup: attempted interactive full benchmark job `10094177` (`rob25-full-int`) was cancelled by Slurm before start due `QOSMaxMemoryPerJob`; the active full benchmark remains H100 job `10094147`.
