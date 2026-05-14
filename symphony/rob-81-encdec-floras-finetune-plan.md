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

- ROB-81 had no recent Linear comments when the first plan was prepared. A
  later human comment on 2026-05-13 asked to check whether Floras has many OOV
  words for the Spotify-trained encoder-decoder model before proceeding. A
  follow-up human comment on 2026-05-14 asked for an on-the-fly normalization
  re-audit and an explicit check that the normalization is not harming the
  transcript. The latest human comment on 2026-05-14 asked to proceed with
  that normalization, drop samples that still have OOV words after
  normalization, use 12 Floras epochs, and provide the exact Stanage scheduler
  launch specification before queueing.
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

## Floras OOV Audit

After the later Linear comment, a bounded Stanage CPU audit checked the prepared
Floras labels against the Spotify-trained encoder-decoder tokenizer:

```text
script: symphony/rob81_floras_oov_audit.py
slurm wrapper: symphony/rob81_floras_oov_audit.sbatch
job: 10207037
stdout: /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/oov-audit-10207037.out
stderr: /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/oov-audit-10207037.err
json: /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/oov-audit-full.json
mapping: /users/acp21rjf/align_floras50/tmp/mapping.json
```

The audit reads label JSON only; it does not load audio tensors or train. It
counts a word occurrence as OOV if encoding that word emits tokenizer id `1`,
the SentencePiece `[UNK]` id. It also reports a word-wise subword UNK rate, so
the numbers are not a WER proxy.

Results on all 30,482 prepared Floras records:

| Tokenizer | Word occurrences with `[UNK]` | Word OOV rate | Unique OOV words | Records with `[UNK]` |
| --- | ---: | ---: | ---: | ---: |
| Spotify default tokenizer (`lcasr/artifacts/tokenizer.model`) | 714,172 / 87,774,712 | 0.8136% | 169,807 / 925,430 | 19,859 / 30,482 |
| Floras tokenizer (`lcasr/artifacts/floras50/tokenizer.model`) | 13 / 87,774,712 | 0.0000148% | 13 / 925,430 | 8 / 30,482 |

The Spotify-tokenizer OOVs are numerous by unique type and record coverage, but
the occurrence rate is under 1%. The most frequent raw-label causes are markup
or punctuation rather than ordinary lexical gaps:

```text
&gt;&gt; 23263
it’s 16254
I’m 12348
– 12246
don’t 11533
that’s 9399
you’re 7091
It’s 6971
we’re 5586
[ 5164
] 5152
[Music] 4626
```

Do not switch the finetune to the Floras tokenizer without a deliberate
checkpoint-compatibility decision. The source checkpoint's decoder embeddings
and output head were trained against the Spotify tokenizer id semantics; the
Floras tokenizer has the same vocabulary size but different pieces/ids.

After the 2026-05-14 follow-up comment, the audit was extended to compare raw
labels against a conservative on-the-fly normalization:

- recursively unescape HTML entities,
- normalize common Unicode apostrophes, quotes, dashes, and non-breaking
  spaces,
- remove standalone speaker arrows,
- strip bracket characters while preserving their contents,
- trim leading/trailing token punctuation,
- drop tokens with no alphanumeric content, and
- collapse whitespace.

The first normalized job (`10214935`) completed but exposed an overly noisy
transcript-harm metric around double-escaped HTML. The v2 job fixed that metric,
then the final v3 job also removed underscore-only placeholder tokens as
no-alphanumeric markup. It wrote:

```text
job: 10215028
stdout: /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/oov-audit-10215028.out
stderr: /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/oov-audit-10215028.err
json: /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/oov-audit-normalized-v3-full.json
elapsed: 00:07:14
state: COMPLETED 0:0
```

Normalized Spotify-tokenizer result on all 30,482 records:

| Mode | Word occurrences with `[UNK]` | Word OOV rate | Unique OOV words | Records with `[UNK]` |
| --- | ---: | ---: | ---: | ---: |
| Raw | 714,172 / 87,774,712 | 0.8136% | 169,807 / 925,430 | 19,859 / 30,482 |
| Safe normalized | 40,082 / 87,559,909 | 0.0458% | 18,196 / 511,370 | 3,525 / 30,482 |

Transcript-harm diagnostics for the safe normalization:

| Check | Result |
| --- | ---: |
| Records changed by normalization | 27,248 / 30,482 |
| Empty after normalization | 0 |
| Records with any alphanumeric-content delta | 4 / 30,482 |
| Records with >10% word-count drop | 17 / 30,482 |
| Raw word count | 87,774,712 |
| Normalized word count | 87,559,909 |

The sampled alphanumeric-delta and large-word-drop records were dominated by
punctuation, HTML/markup, bracketed stage directions, repeated dash separators,
or Unicode symbol normalization. Manual spot-checks did not show obvious
deletion of spoken transcript content, but the four alphanumeric-delta records
should be reviewed once more before wiring this into the training dataloader.

Remaining Spotify-tokenizer OOV after normalization is much smaller and mostly
non-English script, markup fragments, or special symbols. Top examples include
Arabic phrases such as `الله`, residual HTML fragments such as
`color="#000000`, `BLANK_AUDIO`, `Māori`, masked profanity such as `sh*t`, and
named entities with diacritics such as `Bartłomiej` / `Płotka`.

Recommendation before launch: keep the Spotify tokenizer for checkpoint
compatibility and use this conservative label normalization for Floras
supervised finetuning, subject to a final review of the four alphanumeric-delta
records. The normalized Spotify OOV rate is low enough that tokenizer mismatch
no longer looks like a blocker for a CPU smoke or a small supervised finetune
pilot. Do not switch to the Floras tokenizer without a deliberate
checkpoint-compatibility change.

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
   - label normalization/tokenization behavior,
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

## Proposed 12-Epoch Launch Specification

This is the concrete launch shape to use for the latest Linear request. Do not
queue it until the implementation patch, CPU smoke, and callback/finalizer dry
run have passed.

Training config:

```text
config: exp/configs/enc_dec/rob81_floras50_supervised_12ep_lr1e-4.yaml
trainer: exp/train_files/train_enc_dec.py
model_class: EncDecSconformerV2
source checkpoint: /mnt/parscratch/users/acp21rjf/spotify/checkpoints/enc_dec/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt
tokenizer: lcasr/artifacts/tokenizer.model
data source: /users/acp21rjf/align_floras50/tmp/mapping.json
filtered manifest: /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/manifests/floras50_safe_norm_drop_oov.json
checkpoint dir: /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/checkpoints/supervised_floras50_spotifytok_safe_norm_drop_oov_lr1e-4_12ep
wandb dir: /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/wandb
wandb project: floras50_enc_dec_supervised
wandb name: rob81-supervised-floras50-safe-norm-drop-oov-lr1e-4-12ep
optimizer: madgrad
lr: 1e-4
warmup_steps: 100
max_epochs: 12
audio_chunking.size: 2048
batch_size: 88
dtype: bfloat16
save_every_n_steps: 2000
random_seed: 552268
```

Data handling:

- Keep the Spotify tokenizer for checkpoint compatibility.
- Apply the audited safe normalization to labels.
- Drop any Floras record whose normalized label still emits tokenizer id `1`
  (`[UNK]`) for any word under the Spotify tokenizer. The latest audit implies
  this will drop `3,525 / 30,482` records and train on about `26,957` records.
- Materialize the filtered manifest and any normalized transcript artifacts
  under the ROB-81 artifact directory; do not mutate the original Floras data
  or checkpoint directories.

Required implementation before launch:

- Extend `exp/train_files/train_enc_dec.py` to honor
  `checkpointing.pretrained` as a model-weights source while keeping a fresh
  finetune optimizer, scheduler, step count, and seen-id state.
- Reuse the same safe normalization logic from
  `symphony/rob81_floras_oov_audit.py` for training labels.
- Add an issue-local manifest-prep/smoke path that verifies normalized-label
  filtering, source checkpoint load, output permissions, and a tiny dataloader
  or one-batch model path on Stanage CPU.
- Add a finalizer or `EXIT`-trap callback that posts job status, log paths,
  checkpoint path, and failure evidence to Linear, then moves ROB-81 back to
  `Todo` for result inspection.

CPU smoke submission:

```bash
ssh acp21rjf@stanage.shef.ac.uk 'cd /users/acp21rjf/long-context-asr && git fetch origin symphony/ROB-81-finetune-best-encdec-floras && git checkout -B symphony/ROB-81-finetune-best-encdec-floras FETCH_HEAD && mkdir -p /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81 && sbatch symphony/rob81_floras50_finetune_cpu_smoke.sbatch'
```

GPU Slurm request:

Use one job with a comma-separated partition list so Slurm can place it on any
of the requested GPU pools without running duplicate training jobs:

```bash
ssh acp21rjf@stanage.shef.ac.uk 'cd /users/acp21rjf/long-context-asr && git fetch origin symphony/ROB-81-finetune-best-encdec-floras && git checkout -B symphony/ROB-81-finetune-best-encdec-floras FETCH_HEAD && sbatch symphony/rob81_floras50_finetune_gpu.sbatch'
```

`symphony/rob81_floras50_finetune_gpu.sbatch` should use:

```bash
#SBATCH --job-name=rob81-ft-floras12
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=80:00:00
#SBATCH --mem=82GB
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/floras12-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/floras12-%j.err
```

The GPU command inside the script should be:

```bash
python exp/train_files/train_enc_dec.py \
  --config exp/configs/enc_dec/rob81_floras50_supervised_12ep_lr1e-4.yaml \
  --reset_step \
  --remove_scheduler \
  --num_workers 4 \
  --pin_memory \
  --prefetch_factor 2
```

Completion check:

```bash
sacct -j <gpu_job_id> --format=JobID,JobName,State,ExitCode,Elapsed
tail -n 80 /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/floras12-<gpu_job_id>.err
tail -n 120 /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/floras12-<gpu_job_id>.out
find /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/checkpoints/supervised_floras50_spotifytok_safe_norm_drop_oov_lr1e-4_12ep -maxdepth 1 -name "step_*.pt" | sort -V | tail
```

If `--partition=gpu,gpu-h100,gpu-h100-nvl` is rejected on Stanage, fall back to
the same script on `gpu-h100-nvl` first, because that is the validated partition
for recent ROB-26 encoder-decoder work, then record the rejection and the final
chosen partition in Linear.

## Queued Run

The prelaunch CPU smoke was run on Stanage as job `10215207` and completed
successfully in `00:46:01`. It materialized the filtered manifest at:

```text
/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/manifests/floras50_safe_norm_drop_oov.json
```

Manifest summary:

- input records: `30,482`
- kept records: `26,957`
- dropped normalized-OOV records: `3,525`
- dropped missing/empty records: `0`
- unique words checked: `511,370`

`sbatch --test-only` accepted the multi-partition request but estimated a
general `gpu` placement around 2026-06-10, while `gpu-h100` estimated
2026-05-23 and `gpu-h100-nvl` estimated 2026-06-22. The actual queued run uses
`gpu-h100`.

Queued jobs:

```text
GPU job: 10215377
Finalizer job: 10215378
Partition: gpu-h100
GPU log stdout: /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/floras12-10215377.out
GPU log stderr: /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/floras12-10215377.err
Completion check: squeue -j 10215377,10215378 -o '%i|%j|%T|%R|%S|%M|%l|%P'
```
