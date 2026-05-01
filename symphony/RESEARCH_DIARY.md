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
- ROB-25 Slurm cleanup: attempted interactive full benchmark job `10094177` (`rob25-full-int`) was cancelled by Slurm before start due `QOSMaxMemoryPerJob`; pending H100 full job `10094147` was later canceled to avoid duplicating the active interactive benchmark.
- ROB-25 Slurm launch: submitted active full TEDLIUM-test benchmark job `10094215` (`rob25-full-int`) on the interactive partition with 60GB; logs `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-25/rob25-full-int-10094215.out` and `.err`, output CSV `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-25/results/enc_dec_tedlium_test.csv`.
- ROB-25 Slurm validation: full TEDLIUM-test benchmark job `10094215` completed successfully in 1:34:24 with exit code `0:0`; parsed CSV has 84 data rows and 7 aggregate rows, all `overlap_ratio: 0`, with no failure lines in envcheck/smoke/full logs.
- ROB-25 results: added `symphony/rob-25-encdec-benchmark-results.md`; best aggregate TEDLIUM test WER was `enc_dec_3l_no_anorm_v2` at `0.09803296119085593`.
- ROB-25 PR handoff: rebased branch `symphony/ROB-25-benchmark-encdec` onto `origin/dev` `67d5b2c`, force-updated the issue branch after confirming the remote branch SHA because local Git lacks `--force-with-lease`, and opened draft PR https://github.com/robflynnyh/long-context-asr/pull/4.
