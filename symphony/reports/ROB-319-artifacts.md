# ROB-319 Artifact Notes

The ROB-319 deletion-search runner is configured to require the issue's 18L
1024D checkpoint family by default. On the current Mimas host,
`/mnt/parscratch/users/acp21rjf/...` was not mounted; if an exact Mimas-local
mirror exists, pass it with `ROB319_CHECKPOINT_ROOT` or `--checkpoint-root`.

- Default artifact root:
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search`
- Default checkpoint root:
  `/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb_18l_1024D`
- Default Earnings root:
  `/store/store4/data/earnings-22`
- Default checkpoints:
  `n_seq_sched_1024_rp_1/step_105360.pt`,
  `n_seq_sched_8192_rp_1/step_105360.pt`,
  `n_seq_sched_16384_rp_1/step_105360.pt`
  were verified read-only on Stanage under the default checkpoint root.
- Rejected setup-smoke artifact:
  `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-319/eggroll_deletion_context_search/rob319-targets-smoke-20260628T000000Z`
  was produced while checking `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR`;
  it is evidence that this mirror is the wrong 6-layer 768D family, not a real
  ROB-319 search result.
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

Each run writes a run-local `ARTIFACT_INDEX.md`, resolved config, target tensor
list, block manifest, clean/candidate/validation metrics, pair weights, and
summary JSON. Source checkpoints are loaded read-only; low-rank perturbations
are applied in memory and restored unless `ROB319_SAVE_COMBINED_CHECKPOINTS=1`
is explicitly set for the launcher.

Validation note: `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR` contains
matching `n_seq_sched_*` paths but loaded as a 6-layer 768D family (~90M
parameters), so it is not a valid default for this issue's requested 18L 1024D
comparison.
