# Symphony Research Diary

This diary is for concise, durable notes from Symphony-managed work on this repository.

## Diary Guidelines

- Record outcomes, decisions, fixes, experiment launches, validation results, and artifact locations. Do not record every queue poll, resume check, or repeated status observation.
- Prefer one bullet per meaningful unit of work. Include the issue key, branch when relevant, important job IDs, final states, and paths needed for follow-up.
- Summarize repeated attempts as a single entry that says what was tried and what decision followed. Move detailed Slurm behavior, commands, or extraction snippets to focused notes such as `symphony/slurm-notes.md`, `symphony/training-notes.md`, or `symphony/eval-notes.md`.
- Keep credentials, raw data, checkpoints, large logs, generated CSVs, and bulky output out of the diary. Reference parscratch paths instead.

## 2026-05-11

- ROB-78 on branch `symphony/rob-78-spotify-f-long-6epoch`: added a 15-run Spotify long-only fine-tuning template for the 6-epoch base model (`w128`, `w512`, `w2048`, `w8192`, and full-context `w360000`, each with 3 repeats), plus deterministic config expansion, Stanage CPU-smoke, GPU-array, and Linear callback finalizer scripts. CPU smoke job `10160987` completed successfully after loading the 6-epoch full-context checkpoint and running one derived 512-frame smoke sample. Queued GPU array `10161003_[0-14]` on `gpu-h100-nvl` with finalizer `10161005`; large artifacts live under `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-78` and `/mnt/parscratch/users/acp21rjf/spotify/long_only/FT_3epoch_6epoch`.

## 2026-05-01

- ROB-24 on branch `symphony/ROB-24-agent-docs`: added focused future-agent notes under `symphony/` for repo orientation, training, evaluation, and Slurm patterns. Opened PR https://github.com/robflynnyh/long-context-asr/pull/2 against `dev`. No Slurm jobs launched.
- ROB-25 on branch `symphony/ROB-25-benchmark-encdec`: added encoder-decoder TEDLIUM benchmark support, config-driven `transcribe_kwargs`, and results notes. Validation jobs `10094097`, `10094080`, and `10094215` completed successfully; best aggregate TEDLIUM test WER was `enc_dec_3l_no_anorm_v2` at `0.09803296119085593`. Opened PR https://github.com/robflynnyh/long-context-asr/pull/4 against `dev`.
- ROB-26 on branch `symphony/ROB-26-rl-post-training`: found archived encoder-decoder RL code was not wired into active training. Added an on-policy MaxRL/GRPO post-training entrypoint, Floras-50 RL configs, Slurm launchers, Tedlium eval config, and `symphony/rl-post-training-notes.md`. Opened draft PR https://github.com/robflynnyh/long-context-asr/pull/3 against `dev`.
- ROB-26 validation: CPU compile/self-test/config/checkpoint-load validation passed across jobs `10094324`, `10094337`, `10094410`, `10094426`, `10094436`, `10094491`, `10097233`, `10097247`, `10097253`, `10097382`, `10097383`, `10097384`, and `10097385`. Key fixes from validation were remapping legacy decoder norm `.scale` checkpoint keys to current `.weight` keys, keeping rollout sampling and gradient log-prob recomputation in eval mode to avoid dropout mismatch, and filtering mixed-length RL chunks to avoid zero-length audio chunks.
- ROB-26 checkpoint clarification: Linear requested `enc_dec_3l_no_anorm_v2` instead of `baseline_rp_1`; canceled the queued baseline GPU run `10094361` before start and updated config/eval/docs to use `/mnt/parscratch/users/acp21rjf/spotify/checkpoints/enc_dec/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt`.
- ROB-26 3K GPU training launched from `enc_dec_3l_no_anorm_v2`: job `10094511` on `gpu-h100-nvl`, logs `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/rl3k-10094511.{out,err}`, checkpoints under `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/checkpoints/rl_floras50_3k_enc_dec_3l_no_anorm_v2/`. Dependent Tedlium eval job `10097287` was queued with `afterok:10094511`.
- ROB-26 Slurm handoff: multiple replacement probes showed `sbatch --test-only` was unreliable for this workload; worse real replacement submissions were canceled before start. Reusable queue-placement guidance and the shared-workspace commit-pinning caveat are documented in `symphony/slurm-notes.md`.
- ROB-26 eval handoff: added aggregate Tedlium WER extraction instructions to `symphony/eval-notes.md` for `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/eval/tedlium_rl_floras50_3k.csv`.

## 2026-05-03

- ROB-26 RL configs were scaled up for the next run: the 3K config now uses batch size 6 and 8 rollouts; the 30K constant-LR sweep configs were renamed to `exp/configs/enc_dec/rl_floras50_30k_b18_r48_const_lr_<lr>.yaml` and now use batch size 18 and 48 rollouts.
- Added parallel 30K GRPO LR-sweep configs under `exp/configs/enc_dec/rl_floras50_30k_b18_r48_grpo_const_lr_<lr>.yaml`; these keep `rl.reward_threshold: 0.8`, while MaxRL configs no longer include that GRPO-only threshold field.

## 2026-05-02

- ROB-26 3K GPU training job `10094511` completed successfully (`COMPLETED 0:0`). Final checkpoint exists at `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/checkpoints/rl_floras50_3k_enc_dec_3l_no_anorm_v2/step_3000.pt`. The dependent Tedlium eval job `10097287` was canceled before running.
- ROB-26 follow-up after 3K results: the completed run showed near-zero sparse rewards and LR decay to zero, so added `scheduler.name: constant` support plus six 30K-step configs under `exp/configs/enc_dec/rl_floras50_30k_b6_r24_const_lr_<lr>.yaml`. Each run used batch size 6, 24 rollouts, MaxRL, constant LR, and saves every 2K steps to a separate checkpoint directory under `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/checkpoints/`.
- ROB-26 constant-LR sweep launched on `gpu-h100-nvl`: `10101630` lr `3e-6`, `10101631` lr `6e-6`, `10101632` lr `1e-5`, `10101633` lr `2e-5`, `10101634` lr `4e-5`, and `10101635` lr `8e-5`. The launcher `exp/configs/enc_dec/rl_floras50_30k_b6_r24_const_lr_gpu.sh` exported `PYTHONPATH=/mnt/parscratch/users/acp21rjf/symphony-workspaces/ROB-26` so jobs used this issue checkout instead of an installed `lcasr` package from another workspace.
