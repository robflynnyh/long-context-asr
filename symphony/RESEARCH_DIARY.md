# Symphony Research Diary

This diary is for concise, durable notes from Symphony-managed work on this repository.

Agents should append dated entries for meaningful implementation changes, experiment launches, completed Slurm jobs, failed runs, fixes, validation outcomes, PR handoffs, and interpretation updates. Keep large logs and artifacts in parscratch and reference their paths here.

## 2026-05-01

- ROB-24 on branch `symphony/ROB-24-agent-docs`: identified sparse/scattered operational docs, added focused future-agent notes under `symphony/` for repo orientation, training, evaluation, and Slurm patterns. No Slurm jobs launched.
- ROB-24 PR handoff: opened GitHub PR https://github.com/robflynnyh/long-context-asr/pull/2 against `dev` after docs validation.
- ROB-27 on branch `symphony/ROB-27-benchmark-18l-long-context`: recovered incomplete workspace checkout, verified clean `dev` at `origin/dev` `40ba9a0`, and identified `exp/configs/paper_templates/exp_set_spotify_L_FT.yaml` as the 18L long-only finetune launcher. Added evaluation config for `/mnt/parscratch/users/acp21rjf/spotify/long_only/FT_3epoch_18L/*/step_23763.pt`.
- ROB-27 Slurm launch: submitted GPU job `10094264` (`ROB27-18L-eval`) from `eval/` with config `eval/eval_configs_for_thesis/eval_config_pre_windowed_spotify_FT_long_only_18L.yaml`; logs under `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-27/`, output CSV `eval/results/thesis/long_only_FT_3epoch_18L_5e5.csv`.
- ROB-27 queue check: job `10094264` remained pending for `Priority`; `squeue --start` estimated `2026-05-03T05:20:00` on `gpu33`. A100 and alternate H100 test-only submissions were later than the existing job, so the original pending job was kept.
- ROB-27 queue update: lowered pending job `10094264` walltime from 90h to 48h with `scontrol update JobId=10094264 TimeLimit=48:00:00`; estimated start remained `2026-05-03T05:20:00`. Slurm later reported the pending reason as `Priority`.
- ROB-27 queue probe: 4h/8h/12h H100 test-only jobs all estimated `2026-05-03T07:50:50`, later than existing job `10094264`, so no split jobs were submitted. Pushed branch `symphony/ROB-27-benchmark-18l-long-context` to origin without opening a PR because the required CSV is still pending.
