# ROB-319 Artifact Notes

The ROB-319 deletion-search runner now defaults to the 6L SAP_LCASR
sequence-scheduler checkpoint family after Rob's 2026-06-28 Linear correction
to use the SAP models rather than the earlier 18L family.

- Default artifact root:
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search`
- Default checkpoint root:
  `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR`
- Superseded Mimas-local copied 18L checkpoint root:
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/source_checkpoints_18l_1024D`
- Default Earnings root:
  `/store/store4/data/earnings-22`
- Default checkpoints:
  `n_seq_sched_1024_rp_1/step_105360.pt`,
  `n_seq_sched_8192_rp_1/step_105360.pt`,
  `n_seq_sched_16384_rp_1/step_105360.pt`
  exist under the SAP_LCASR default root; each repeat-1 file is about 1.45 GB.
- Early SAP setup-smoke artifact:
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search/rob319-targets-smoke-20260628T000000Z`
  was produced while checking `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR`.
  It was initially treated as the wrong family only because the issue text
  emphasized 18L checkpoints; after the latest Linear correction, SAP_LCASR is
  the intended family.
- SAP target smoke:
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search/rob319-targets-smoke-sap6l-20260628T1730Z`
  loaded repeat-1 SAP_LCASR `1024`, `8192`, and `16384` checkpoints as
  89.9M-parameter 6L models and matched 42 shared target tensors: 18
  convolution, 12 `ff1`, and 12 `ff2`.
- Callback dry-run artifact:
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search/rob319-callback-dryrun-20260628T000000Z`
  verifies the Mimas wrapper summary/callback path without launching GPUs or
  loading checkpoints.
- Exact 18L checkpoint copies:
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/source_checkpoints_18l_1024D`
  contains the copied repeat-1 `1024`, `8192`, and `16384` `step_105360.pt`
  files from Stanage. Local and remote byte sizes match at `7139994645` bytes
  per file.
- Exact-copy target smoke:
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search/rob319-targets-smoke-18l-copy-20260628T000000Z`
  loaded the copied checkpoints as 445.7M-parameter 18L models and matched 126
  shared target tensors: 54 convolution, 36 `ff1`, and 36 `ff2`.
- Failed windowed GPU smoke:
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search/rob319-gpu-smoke-18l-copy-20260628T000000Z`
  used the exact copied 18L checkpoints and failed before clean block
  completion because the current Mimas environment has `flash_attn==1.0.8`
  and does not expose the `flash_attn_qkvpacked_func` API required by the
  repo's local-window attention path. The model fell back to the non-flash
  branch, which asserts that windowed attention is unsupported there.
- Failed capped GPU smokes:
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search/rob319-gpu-smoke-18l-capped-20260628T000000Z`,
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search/rob319-gpu-smoke-18l-cap256-blocking-20260628T000000Z`,
  and
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search/rob319-gpu-smoke-18l-capped-ckptoff-20260628T000000Z`
  were intermediate Mimas debugging runs for averaged-moving-window capped
  smoke evaluation. They exposed CUDA device/allocator state issues before the
  runner was changed to set each model bundle's CUDA device explicitly.
- Successful capped 2-GPU smoke:
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search/rob319-gpu-smoke-18l-cap256-setdevice-20260628T000000Z`
  ran under `with-gpu any --num 2`, loaded the exact copied 18L checkpoints,
  evaluated 2 antithetic pairs on one search block and one validation block,
  and logged `context_score` diagnostics to W&B run
  `https://wandb.ai/wobrob101/long-context-asr/runs/d8sxxzd2`. This is a
  mechanics-only smoke: it used `evaluation_mode=averaged_moving_window` and
  `max_audio_frames=256`, so WERs near `0.9997` reflect cropped audio against
  full references and are not a scientific deletion-search result.
- Artifact-local FlashAttention v2 build:
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/flash_attn_v2_build_20260628T1445Z`
  contains a local build of `flash_attn-2.5.9.post1` for Python 3.9,
  PyTorch 2.0.1, CUDA 11.7, and RTX A4500 compute capability 8.6. The wheel
  and target `site` directory are kept outside the shared conda environment;
  pass the target site path through `ROB319_EXTRA_PYTHONPATH` for Mimas runs.
- Windowed FlashAttention v2 smokes:
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search/rob319-gpu-smoke-18l-windowed-fa2-20260628T000000Z`
  imported the artifact-local FlashAttention v2 build but failed on the first
  clean block with CUDA OOM in subsampling when the whole Earnings recording
  was decoded as one chunk. After changing the runner to decode windowed mode
  in each checkpoint's trained context-length chunks,
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search/rob319-gpu-smoke-18l-windowed-fa2-cap2048-20260628T000000Z`
  completed with `evaluation_mode=windowed_attention`, `max_audio_frames=2048`,
  one antithetic pair, one search block, and W&B run
  `https://wandb.ai/wobrob101/long-context-asr/runs/xl4hz23b`. This is also a
  mechanics-only smoke because the audio was capped against full references.
  The larger capped probe
  `rob319-gpu-smoke-18l-windowed-fa2-cap32768-20260628T000000Z` was manually
  interrupted after the high-overlap short-model decode proved too slow for a
  quick smoke.
- Successful uncapped full-recording fp16 smoke:
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search/rob319-gpu-smoke-18l-windowed-fa2-fp16-fullrec-1pair-20260628T000000Z`
  completed one full Earnings search block with one antithetic pair using
  `windowed_decode_strategy=full_recording` and `autocast_dtype=float16`, while
  leaving model weights in fp32 for the low-rank perturbations. W&B run:
  `https://wandb.ai/wobrob101/long-context-asr/runs/z4wabrbg`. The clean
  block WERs were `short=0.3219`, `medium=0.2845`, and `long=0.2810`, giving a
  clean long-context gain of `0.0408`; both candidate signs preserved nearly
  all of that gain (`context_score` about `0.996` to `0.998`).
- Superseded full 32-pair batch/offline run:
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search/rob319-full-18l-windowed-fa2-fp16-32pairs-20260628T1520Z`
  was launched in detached screen
  `3477642.rob319-full-fa2-fp16-20260628T1520Z` on Mimas with
  `with-gpu any --num 2`, exact copied checkpoints, artifact-local
  FlashAttention v2, `windowed_decode_strategy=full_recording`, and
  `autocast_dtype=float16`. W&B run:
  `https://wandb.ai/wobrob101/long-context-asr/runs/twnmunja`. Rob stopped
  this run after identifying that it used a batch/offline search shape: it
  evaluated temporary candidates from the clean/current weights and deferred
  the combined perturbation until all search blocks. Its partial block-0
  candidate files are retained for provenance only and are not a final
  ROB-319 result.
- Current blockwise runner behavior:
  `symphony/scripts/rob319_eggroll_deletion_search.py` now defaults to
  `search.update_mode: blockwise`. For each 5-recording search block it
  evaluates the 64 antithetic candidates around the current in-memory damaged
  weights, computes pair weights from that block's `context_score` only,
  permanently applies `W <- W + eta * combined_delta_block`, logs
  `block_update_metrics.jsonl` and `pair_weights.jsonl`, then writes
  `deletion_state_latest.json`. Resume uses `--resume-state` or
  `ROB319_RESUME_STATE` to replay the compact accumulated pair-weight state
  into freshly loaded clean checkpoint copies.
- Successful blockwise 2-GPU smoke:
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search/rob319-gpu-smoke-18l-blockwise-fa2-fp16-fullrec-1pair-20260628T1711Z`
  used the exact copied 18L checkpoints, artifact-local FlashAttention v2,
  `windowed_decode_strategy=full_recording`, `autocast_dtype=float16`, one
  search block, one antithetic pair, and skipped held-out validation. W&B run:
  `https://wandb.ai/wobrob101/long-context-asr/runs/fm01ghef`. It wrote one
  clean/current row, two candidate rows, one pair-weight row, one block-update
  row, and compact deletion state. The block update had nonzero
  `pair_weight=0.0018844221105529524`, `combined_delta_norm=35.68911983501585`,
  and `cumulative_delta_norm=0.0035689119835015846`.
- Successful resume check:
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search/rob319-gpu-smoke-18l-blockwise-resume-check-20260628T1721Z`
  loaded the smoke's `deletion_state_latest.json`, replayed the accumulated
  update into freshly loaded model copies, skipped completed search block 0,
  and exited with the same `cumulative_delta_norm=0.0035689119835015846`.
- Superseded 18L blockwise full launch:
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search/rob319-full-18l-blockwise-fa2-fp16-32pairs-20260628T1725Z`
  was launched after the blockwise fix but before the 2026-06-28 17:18 UTC
  Linear correction was incorporated. It was stopped with exit code 130 and is
  not a ROB-319 result. The corrected run should use the SAP_LCASR checkpoint
  root.
- Successful SAP blockwise 2-GPU smoke:
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search/rob319-gpu-smoke-sap6l-blockwise-fa2-fp16-fullrec-1pair-20260628T1731Z`
  used the SAP_LCASR repeat-1 checkpoints, artifact-local FlashAttention v2,
  `windowed_decode_strategy=full_recording`, `autocast_dtype=float16`, one
  search block, one antithetic pair, and skipped held-out validation. W&B run:
  `https://wandb.ai/wobrob101/long-context-asr/runs/lcg6xwbg`. Clean/current
  block WERs were `short=0.4007`, `medium=0.3145`, and `long=0.3108`, giving a
  clean long-context gain of `0.0899`. It wrote the expected candidate,
  pair-weight, block-update, and deletion-state artifacts.
- Successful SAP resume check:
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search/rob319-gpu-smoke-sap6l-blockwise-resume-check-20260628T1741Z`
  loaded the SAP smoke's `deletion_state_latest.json`, replayed the accumulated
  update into fresh SAP model copies, skipped completed search block 0, and
  exited with the same `cumulative_delta_norm=4.678250729025346e-05`.

Each run writes a run-local `ARTIFACT_INDEX.md`, resolved config, target tensor
list, block manifest, clean/current/candidate/validation metrics, pair weights,
block-update diagnostics, compact deletion state, and summary JSON. Source
checkpoints are loaded read-only; low-rank perturbations and persistent
blockwise deletion updates are applied only to in-memory model copies unless
`ROB319_SAVE_COMBINED_CHECKPOINTS=1` is explicitly set for the launcher.

Validation note: the earlier 18L checkpoint-copy and smoke artifacts remain
recorded for provenance, but current ROB-319 runs should use the default
`/store/store5/data/acp21rjf_checkpoints/SAP_LCASR` root unless Rob changes
that instruction again.
