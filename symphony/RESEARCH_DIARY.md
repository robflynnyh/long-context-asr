# Symphony Research Diary

This diary is for concise, durable notes from Symphony-managed work on this repository.

## Diary Guidelines

- Record outcomes, decisions, fixes, experiment launches, validation results, and artifact locations. Do not record every queue poll, resume check, or repeated status observation.
- Prefer one bullet per meaningful unit of work. Include the issue key, branch when relevant, important job IDs, final states, and paths needed for follow-up.
- Summarize repeated attempts as a single entry that says what was tried and what decision followed. Move detailed Slurm behavior, commands, or extraction snippets to focused notes such as `symphony/slurm-notes.md`, `symphony/training-notes.md`, or `symphony/eval-notes.md`.
- Keep credentials, raw data, checkpoints, large logs, generated CSVs, and bulky output out of the diary. Reference parscratch paths instead.

## 2026-05-18

- ROB-90 on branch `symphony/ROB-90-streaming-decoder-tedlium-eval`: wired bounded TEDLIUM utterance evaluation for `StreamingDecoderASR` through the generic `eval/run.py` path, updated greedy two-head inference to score text by `P(not silence) * P(token)`, and ran the ROB-76 two-head checkpoint `/store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_two_head_mimas_full_epoch_rob76-mimas-3epoch-b48-two-head-20260517T140145Z/step_82737.pt` on Mimas GPU 3. The one-utterance smoke passed, then the 8-utterance greedy test wrote JSONL/summary artifacts under `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/generic-eval-run-8utt/`; compact report committed at `symphony/reports/ROB-90-streaming-decoder-tedlium-eval.md`. After review feedback, ROB-90-specific artifact/report helpers were moved out of the widely used `eval/run.py` into `eval/streaming_decoder_eval.py`, and the one-utterance smoke was rerun under `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/smoke-helper-rework/`.
- ROB-76 PR cleanup: Linear selected the two-head streaming decoder, so the single-head CE and canceled soft-dilation ablations were removed from the PR. The retained Mimas run `rob76-mimas-3epoch-b48-two-head-20260517T140145Z` reached step `82737`; future long Mimas runs now execute immutable run-dir wrapper/callback copies to avoid live-script edits affecting exit traps.
- ROB-76 padding cleanup: the streaming decoder no longer passes a padding mask into causal self-attention because padding is right-tail only, valid queries cannot attend to future padded keys, and padded target positions are ignored by the loss. This also removes the shared `attention.py` padded-causal SDPA fallback from the PR.

## 2026-05-17

- ROB-76 design pivot: debugging showed teacher-forced predictions could learn non-silence while free-running greedy decode stayed blank, so the PR moved to a two-head decoder with separate silence and text heads, shuffled training chunks, and no scheduled sampling or loss weighting. The required Stanage CPU smoke job `10228966` passed from a clean clone with the 107.73M model and one optimizer step.
- ROB-76 run evidence: the two-head Mimas 3-epoch run `rob76-mimas-3epoch-b48-two-head-20260517T140145Z` was launched after wrapper and callback dry runs; earlier scheduled-feedback, soft-dilation, and single-head CE explorations were superseded and are not part of the final PR surface.

## 2026-05-13

- ROB-85 on branch `symphony/ROB-85-update-rules`: updated Symphony execution rules and agent notes to ban `/tmp` on Mimas. Future Mimas-local scratch should use repo-local `.tmp/` or another issue-specific user-owned path under `/exp/exp4/acp21rjf/`; Stanage scratch remains `/mnt/parscratch/users/acp21rjf/symphony-tmp`.

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

## 2026-05-11

- ROB-69 on branch `symphony/ROB-69-18l-long-context-benchmark`: added an 18L long-only finetune benchmark config comparing `FT_3epoch_18L` checkpoints against matched 18L baseline checkpoints. Stanage CPU smoke job `10156237` completed successfully after seeding 75 baseline rows and checking 30 model entries, five dataset loaders, output paths, and sampled checkpoint metadata. Queued H100 eval job `10156464` with finalizer/callback job `10156465`; remote result path is `/mnt/parscratch/users/acp21rjf/symphony-workspaces-long-context-asr/ROB-69/eval/results/thesis/rob69_18l_long_context_finetune_vs_baseline.csv`.
- ROB-76 on branch `symphony/ROB-76-streaming-decoder-asr`: added the initial decoder-only streaming ASR path with causal subsampling, causal shared attention, previous-label feedback, delayed frame-synchronous word targets, and an explicit silence class. The 100M config instantiated at 107.73M parameters; Stanage CPU smokes through job `10160756` validated the shared-attention version, including the padded causal SDPA fallback.

## 2026-05-15

- ROB-76 queue follow-up: updated the full config to 2048-frame chunks, batch 176, no accumulation, and three epochs; Stanage CPU smoke job `10221260` passed and the callback-backed GPU wrapper path was queued for `gpu-h100-nvl`.

## 2026-05-16

- ROB-76 Mimas/debug path: added the Mimas Spotify-10% manifest helper, W&B/callback wrapper, debug generation, checkpoint interval handling, and batch-48 wrapper defaults. Also fixed the cosine schedule horizon and causal-subsampling chunking after the Stanage `conv2d` indexing failure; bounded Mimas smokes validated these paths.

## 2026-05-12

- ROB-69 eval job `10156464` completed successfully, but finalizer job `10156465` failed because appended finetuned CSV rows included an extra pandas index field. Normalized the completed remote result into `eval/results/thesis/rob69_18l_long_context_finetune_vs_baseline.csv`, made ROB-69 summarization tolerate and normalize that output shape, and fixed eval manager CSV appends to write `index=False`. Mean WER was slightly worse for the long-only finetuned checkpoints on most dataset/window pairs, with small improvements only on `rev16` window 128, `tedlium` window 8192, and `this_american_life` windows 128 and 22500.

## 2026-05-02

- ROB-26 3K GPU training job `10094511` completed successfully (`COMPLETED 0:0`). Final checkpoint exists at `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/checkpoints/rl_floras50_3k_enc_dec_3l_no_anorm_v2/step_3000.pt`. The dependent Tedlium eval job `10097287` was canceled before running.
- ROB-26 follow-up after 3K results: the completed run showed near-zero sparse rewards and LR decay to zero, so added `scheduler.name: constant` support plus six 30K-step configs under `exp/configs/enc_dec/rl_floras50_30k_b6_r24_const_lr_<lr>.yaml`. Each run used batch size 6, 24 rollouts, MaxRL, constant LR, and saves every 2K steps to a separate checkpoint directory under `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/checkpoints/`.
- ROB-26 constant-LR sweep launched on `gpu-h100-nvl`: `10101630` lr `3e-6`, `10101631` lr `6e-6`, `10101632` lr `1e-5`, `10101633` lr `2e-5`, `10101634` lr `4e-5`, and `10101635` lr `8e-5`. The launcher `exp/configs/enc_dec/rl_floras50_30k_b6_r24_const_lr_gpu.sh` exported `PYTHONPATH=/mnt/parscratch/users/acp21rjf/symphony-workspaces/ROB-26` so jobs used this issue checkout instead of an installed `lcasr` package from another workspace.
