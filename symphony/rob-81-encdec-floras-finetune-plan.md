# ROB-81 Encoder-Decoder Floras Finetune Plan

Issue: ROB-81

## Instruction Context

The required Symphony instruction files were read before planning:

1. `symphony/instructions/linear-context.md`
2. `symphony/instructions/repository.md`
3. `symphony/instructions/work-loop.md`
4. `symphony/instructions/experiment-execution.md`
5. `symphony/instructions/validation-and-handoff.md`

Directly relevant constraints:

- ROB-81 had no recent Linear comments when the plan was prepared, so no later
  comment changed the issue scope.
- No `Branch/ref:` was supplied, so work should branch from `dev`.
- Stanage is the default execution target for training. Do not run the finetune
  as Mimas/local GPU work unless a later human comment explicitly asks for it.
- A Stanage CPU smoke test must pass against the same code path, config family,
  data paths, checkpoint path, import environment, and output assumptions before
  any GPU job is queued.
- A long GPU job needs a callback or finalizer path before submission. Record
  job IDs, scripts, logs, output paths, branch, and commit in Linear.

## Source Checkpoint

ROB-25 found the best benchmarked encoder-decoder checkpoint to be
`enc_dec_3l_no_anorm_v2`:

- TEDLIUM test aggregate WER: `0.09803296119085593`
- Benchmark config: `eval/eval_configs/rob25_enc_dec_tedlium.yaml`
- Result note: `symphony/rob-25-encdec-benchmark-results.md`
- Checkpoint:
  `/mnt/parscratch/users/acp21rjf/spotify/checkpoints/enc_dec/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt`

ROB-26 used the same checkpoint after Linear clarification for the RL
post-training setup, as documented in `symphony/rl-post-training-notes.md`.

## Original LR

The original training LR for this checkpoint was `2e-3` (`0.002`):

- Source template `exp/configs/enc_dec/enc_dec_test.yaml` sets
  `optimizer.args.lr: [2e-3]`.
- Generated run config `exp/configs/enc_dec/0_873621.yaml` sets
  `optimizer.args.lr: 0.002`.

The checkpoint directory name also contains `lr_2e3`, but the config files are
the authoritative evidence.

## Finetune LR Recommendation

Use `1e-4` as the first supervised Floras-50 finetune LR.

Rationale:

- `2e-3` was the original training LR for the source checkpoint, not a safe
  default for domain finetuning from an already-good model.
- `1e-4` is a 20x reduction from the source LR and is closer to the conservative
  scale used by existing Floras long-only finetune templates.
- ROB-26 RL configs used much smaller LRs such as `1e-5`, but those were for
  on-policy RL/post-training and should not be treated as the default for normal
  supervised finetuning.

If compute budget allows, make the first real run a small LR sweep:

- `5e-5`
- `1e-4`
- `2e-4`

Pick the final LR from a short held-out Floras/dev loss comparison or a small
TEDLIUM smoke evaluation before committing to a larger full run.

## Data Scope

ROB-81 asks for Floras-50 full dataset finetuning. Do not reuse
`lcasr/artifacts/floras50_long_only.json` unless a later human comment changes
the requested scope to long-only.

The existing full Floras templates point at:

```text
/users/acp21rjf/align_floras50/tmp/mapping.json
```

Verify this path on Stanage before preparing the final training config. The
path was not present from the current Mimas workspace during planning, but that
does not prove it is absent on Stanage.

## Implementation Plan

1. Add a supervised `EncDecSconformerV2` Floras-50 finetune config starting from
   the ROB-25 checkpoint.
2. Write all checkpoints, WandB files, logs, and run artifacts under:

   ```text
   /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/
   ```

3. Patch or extend `exp/train_files/train_enc_dec.py` before launch if needed so
   `checkpointing.pretrained` loads model weights from the source checkpoint but
   starts a fresh optimizer, scheduler, step count, and seen-id state for the
   finetune. The current normal encoder-decoder trainer loads from
   `checkpointing.dir`, so blindly using the source checkpoint directory risks
   restoring the original optimizer/scheduler state and ignoring the chosen
   finetune LR.
4. Add a Stanage CPU smoke launcher that validates:
   - imports and syntax,
   - full Floras manifest existence and readability,
   - source checkpoint existence and model load compatibility,
   - output directory permissions,
   - the smallest practical dataloader/model setup path.
5. Add a GPU Slurm launcher only after the CPU smoke passes.
6. Add and dry-run a callback or finalizer that posts success/failure evidence
   to Linear and moves the issue back to `Todo` for final inspection.
7. Queue either the selected `1e-4` run or the small LR sweep after explicit
   launch decision.
8. After completion, inspect logs and artifacts, run targeted evaluation or
   summary extraction, update `symphony/RESEARCH_DIARY.md`, push a PR against
   `dev`, and move the issue to `In Review` only after GitHub handoff succeeds.

## Pre-Launch Validation

Before any full GPU launch:

```bash
python -m py_compile exp/train_files/train_enc_dec.py
bash -n <rob81-cpu-smoke-launcher>
bash -n <rob81-gpu-launcher>
bash -n <rob81-callback-or-finalizer>
```

Then run the Stanage CPU smoke job and inspect its stdout/stderr for tracebacks,
missing paths, load failures, permission errors, or callback errors.
