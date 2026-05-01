# Symphony Research Diary

This diary is for concise, durable notes from Symphony-managed work on this repository.

Agents should append dated entries for meaningful implementation changes, experiment launches, completed Slurm jobs, failed runs, fixes, validation outcomes, PR handoffs, and interpretation updates. Keep large logs and artifacts in parscratch and reference their paths here.

## 2026-05-01

- ROB-24 on branch `symphony/ROB-24-agent-docs`: identified sparse/scattered operational docs, added focused future-agent notes under `symphony/` for repo orientation, training, evaluation, and Slurm patterns. No Slurm jobs launched.
- ROB-24 PR handoff: opened GitHub PR https://github.com/robflynnyh/long-context-asr/pull/2 against `dev` after docs validation.
- ROB-26 on branch `symphony/ROB-26-rl-post-training`: found archived encoder-decoder RL code was not wired into active training; added an on-policy MaxRL/GRPO post-training entrypoint, floras-50 3K-step config, Slurm launchers, Tedlium eval config, and `symphony/rl-post-training-notes.md`.
- ROB-26 CPU validation launched: `sbatch exp/configs/enc_dec/rl_floras50_3k_cpu_debug.sh` -> job `10094324`; logs under `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/cpu-debug-10094324.{out,err}`.
- ROB-26 CPU validation completed: job `10094324` finished `COMPLETED 0:0`; compile, helper self-test, and config validation passed. Strengthened self-test to cover rollout sampling/log-prob gradients and launched second CPU validation job `10094337`.
- ROB-26 second CPU validation completed: job `10094337` finished `COMPLETED 0:0`; strengthened self-test and config validation passed, with only known module/deprecation warnings in stderr.
- ROB-26 GPU training launched: `sbatch exp/configs/enc_dec/rl_floras50_3k_gpu.sh` -> job `10094361`; MaxRL 3K-step floras-50 run from `baseline_rp_1`; logs `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/rl3k-10094361.{out,err}`.
- ROB-26 checkpoint-load CPU validation launched: added `--validate_load_model` to CPU debug script and submitted job `10094410` to instantiate `baseline_rp_1` on CPU and load its state dict before the queued GPU run starts.
- ROB-26 checkpoint-load CPU validation completed: job `10094410` finished `COMPLETED 0:0`; load succeeded with `strict=False` and reported 16 missing/16 unexpected keys. Added key-name logging and launched follow-up CPU validation job `10094426`.
- ROB-26 follow-up CPU validation completed: job `10094426` finished `COMPLETED 0:0`; mismatch was legacy decoder norm `.scale` keys vs current `.weight` keys. Added pretrained state-dict key remapping and launched CPU validation job `10094436`.
- ROB-26 remap CPU validation completed: job `10094436` finished `COMPLETED 0:0`; checkpoint load validation now reports `missing=0, unexpected=0`.
- ROB-26 draft PR opened: https://github.com/robflynnyh/long-context-asr/pull/3 from `symphony/ROB-26-rl-post-training` to `dev`; GPU job `10094361` remains queued pending post-training/eval results.
- ROB-26 checkpoint clarification: Linear comment requested `enc_dec_3l_no_anorm_v2` instead of `baseline_rp_1`; canceled queued baseline GPU job `10094361` before start and updated config/eval/docs to `/mnt/parscratch/users/acp21rjf/spotify/checkpoints/enc_dec/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt`.
- ROB-26 CPU validation relaunched for clarified checkpoint: `sbatch exp/configs/enc_dec/rl_floras50_3k_cpu_debug.sh` -> job `10094491`.
- ROB-26 clarified-checkpoint CPU validation completed: job `10094491` finished `COMPLETED 0:0`; compile, RL self-test, config validation, and checkpoint-load validation passed with `missing=0, unexpected=0`.
- ROB-26 replacement GPU training launched for `enc_dec_3l_no_anorm_v2`: `sbatch exp/configs/enc_dec/rl_floras50_3k_gpu.sh` -> job `10094511`; logs `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/rl3k-10094511.{out,err}`; queued with estimated start `2026-05-03T08:00:00`.
- ROB-26 continuation queue triage: job `10094511` still pending on `gpu-h100-nvl`; `sbatch --test-only` showed hp GPU partitions unavailable for this account and `gpu-h100`/general `gpu` estimates were later, so kept `10094511` queued and documented the check in `symphony/slurm-notes.md`.
- ROB-26 resume check after worker restart: job `10094511` still pending with scheduler estimate `2026-05-02T08:58:11`; recent CPU validation jobs rechecked as `COMPLETED 0:0`, and fresh non-submitting placement checks did not justify cancel/requeue. Updated `symphony/slurm-notes.md`.
