# ROB-87 Audio Codec Training Plan

Issue: ROB-87, "put together a plan for audio codec training"

Date: 2026-05-15

## Instruction And Context Check

Read the required instruction files before planning or editing, in order:

1. `symphony/instructions/linear-context.md`
2. `symphony/instructions/repository.md`
3. `symphony/instructions/work-loop.md`
4. `symphony/instructions/experiment-execution.md`
5. `symphony/instructions/validation-and-handoff.md`

Instructions that directly affect this issue:

- Recent Linear comments had to be fetched before planning. The first human
  correction changed the direction to "train the codecs not eval existing
  ones". Later PR comments further constrained the plan: do not use an existing
  codec library as the implementation base, although other setups can inform
  the recipe; and the codec will likely need explicit attention or equivalent
  long-sequence machinery in both encoder and decoder paths to benefit from
  long sequences. This plan therefore centers on a repo-local codec-training
  recipe under short-context and long-context conditions, with architecture
  controls that separate longer crops from real encoder/decoder context.
  Frozen-token prediction is only a diagnostic, not the main recommendation.
- No `Branch/ref` was supplied, so `dev` is the base branch.
- Long-running GPU work should not be launched unless the issue asks for a run.
  This issue asks for a preliminary plan/investigation, so this handoff is
  docs-only.
- Stanage is the default execution target for future compute. Any codec
  dependency, data-path, or config work must pass a smallest-practical Stanage
  CPU smoke before a GPU job is queued.
- Repository artifacts should stay small. Codec checkpoints, reconstructed
  audio, Slurm logs, W&B state, token dumps, and large metric tables should live
  under durable issue-specific parscratch paths, not Git.
- Documentation-only validation is `git diff --check` plus diff inspection.

## Corrected Question

Test whether neural audio codec training benefits from longer context.

The primary question is not "can a long-context ASR model predict codes from a
frozen codec?" The primary question is:

> If the codec model itself is trained with access to longer waveform context,
> does it learn a better codec than the same architecture trained on short
> crops?

This needs a matched training comparison, because audio codec quality is often
dominated by local waveform reconstruction, quantizer behavior, discriminator
losses, and bitrate. Longer training crops only answer the question if the
codec architecture or losses can actually use cross-window information.

## Key Design Point

Separate three notions of "long context":

1. Longer training crop: the codec sees longer waveform segments in each
   optimization step, but the architecture may still have a fixed convolutional
   receptive field.
2. Longer effective model context: the encoder, quantizer, decoder, or
   discriminator has an explicit path to use information beyond the local
   receptive field.
3. Longer evaluation context: the trained codec is run on long recordings and
   judged for reconstruction continuity, code stability, and downstream utility.

The first pass should include all three as separate controls. A plain crop-size
sweep is useful, but it is not enough by itself to prove long-context use.

For ROB-87, "long context" should ultimately mean more than longer waveform
crops. The codec needs an explicit way to use long-range information in the
encoder before quantization and in the decoder after quantization. Otherwise a
long crop can still reduce to independent local encoding and local waveform
reconstruction.

## Existing Repo Fit

Relevant local affordances:

- `exp/train.py` and the configs under `exp/configs/` already establish this
  repo's long-context training discipline: sequence-length sweeps, Slurm
  launchers, checkpoint directories, and small-to-large validation.
- `symphony/training-notes.md` records the existing command shapes and expected
  config fields for model training.
- `job_scripts/preprocess/` is the only local preprocessing area, and currently
  appears Spotify OGG-to-mel focused rather than codec-waveform focused.
- `lcasr/models/BestRQ.py` is relevant only as a masked discrete-target
  reference. It should not drive the main plan, because ROB-87 is about training
  codecs rather than training predictors over frozen codec tokens.

Main repo gap:

- There is no native waveform codec training harness in `lcasr`.
- There is no checked-in codec dataset manifest or training config.
- The expected Spotify manifest path
  `/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json` was not visible
  from this Mimas checkout during the investigation, so waveform availability
  must be verified from Stanage before implementation.

## Recommended In-Repo Codec Recipe

Do not base the implementation on an existing codec library. Build a minimal
repo-local training recipe, while using established codec setups only as design
references for architecture, losses, metrics, and sanity checks.

Proposed local components:

- `lcasr` model module: a small waveform codec with convolutional encoder,
  residual vector quantization, and convolutional decoder. Keep the first
  version deliberately simple enough to train and ablate before adding
  adversarial losses.
- `exp/train_files/` entry point: a codec-specific training script that follows
  the repo's config, checkpoint, WandB, dtype, and Slurm conventions where
  practical, but does not depend on transcript labels.
- `exp/configs/codec/` configs: short, medium, and long crop variants with
  explicit sample rate, crop seconds, bitrate, codebook count, RVQ depth, batch
  size, gradient accumulation, loss weights, and output paths.
- Issue-local manifest builder: read existing long-form waveform files without
  moving or rewriting data, and write only compact JSON manifests plus summary
  statistics.
- Metrics helper: compute reconstruction losses, SI-SNR or equivalent signal
  metrics, multi-scale STFT loss, codebook usage, perplexity, dead-code rate,
  token rate, and boundary-stability summaries.

Recipe references, not dependencies:

- DAC is a useful reference for the overall neural-codec shape: convolutional
  encoder/decoder, RVQ bottleneck, multi-scale reconstruction losses, optional
  adversarial training, and practical sample-rate/bitrate choices. Do not clone
  it as the implementation base for this plan.
- EnCodec/AudioCraft is a useful reference for SEANet-style encoder-decoder
  structure, residual vector quantization, discriminator/perceptual losses, and
  compression metrics. Treat its configs as recipe guidance, not as the training
  harness to run.
- Mimi is a useful reference for streaming speech-codec behavior and
  downstream token use, especially when designing later long-context or
  chunk-cached variants. It is not the first implementation target.

Initial scope:

- First implement only the non-adversarial autoencoding path if that makes the
  smoke test and crop-length sweep tractable. Add discriminator and
  feature-matching losses only after one-step training and reconstruction
  metrics are stable.
- Keep the first codec small. The goal is to expose whether longer context is
  helpful under controlled conditions, not to immediately match production
  codec quality.

## Encoder And Decoder Context Requirement

The PR clarification means the investigation should not stop at "train the same
local codec on longer segments". Longer segments are a useful baseline, but a
credible long-context codec variant should add context where the codec can use
it:

- Encoder side: add a temporal attention, state-space, or cached-context module
  over encoder latents before RVQ. This lets the discrete codes depend on
  neighboring and earlier speech beyond the convolutional receptive field.
- Quantizer side: keep the first RVQ implementation local unless there is a
  strong reason to complicate it. Measure codebook usage carefully, because a
  long-context encoder that improves reconstruction by reducing code diversity
  may be bad for downstream token modeling.
- Decoder side: add a temporal attention, state-space, or cached-context module
  after RVQ and before waveform upsampling. This tests whether reconstruction
  continuity improves when the decoder sees longer code histories, not only the
  current local code window.
- Streaming constraint: include a causal/cached variant if the target use case
  is streaming or chunked long recordings. A noncausal full-sequence attention
  variant can be useful as an upper bound, but should not be confused with a
  deployable long-context codec.

Recommended architecture ladder:

| Variant | Encoder Context | Decoder Context | Purpose |
| --- | --- | --- | --- |
| Local | convolutional receptive field only | convolutional receptive field only | baseline and crop-length control |
| Enc-long | attention/state/cached context before RVQ | local | tests whether better codes need long context |
| Dec-long | local | attention/state/cached context after RVQ | tests whether reconstruction continuity needs long code history |
| Enc+Dec-long | long-context encoder | long-context decoder | final long-context candidate if single-sided gains are plausible |

This ladder should be preferred over immediately adding every long-context
mechanism at once. It gives interpretable failure modes: if only decoder context
helps, the codes may be adequate but local reconstruction is limiting; if only
encoder context helps, the bottleneck is code assignment; if neither helps, the
dataset/bitrate/loss may not reward long-range information.

## Stage 0: Feasibility Audit

Goal: prove that codec training inputs, dependencies, and a tiny training step
are practical before designing a full sweep.

Tasks:

- On Stanage, locate the waveform source for Spotify or another long-recording
  speech dataset. Record whether the training source is OGG, WAV, FLAC, or only
  precomputed mel `.spec.pt` files.
- Build an issue-local manifest sample with 20 to 100 recordings spanning
  short, medium, and long durations. Do not move or rewrite existing data.
- Implement the smallest repo-local codec skeleton needed for a smoke test:
  dataset, model initialization, forward reconstruction, RVQ/codebook stats,
  loss computation, checkpoint path creation, and config parsing.
- Run the smallest CPU smoke that imports the local codec modules, reads one
  short waveform, builds the dataset, and initializes the model/config without
  starting meaningful training.
- Run a tiny GPU smoke only after the CPU smoke passes. The smoke should execute
  one or two optimizer steps and write logs/checkpoints under
  `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-87/`.
- Record sample rate, crop length, batch size, bitrate, codebook count,
  discriminator setting, command, commit, env, log path, and output path.

Stop criteria:

- Stop and report blocker if only mel spectrograms are available and original
  waveforms cannot be resolved.
- Stop and report blocker if the local codec dependencies cannot be satisfied
  on Stanage without invasive shared-environment changes.
- Stop and report blocker if a one-step training smoke fails before the model
  reaches forward/backward.

## Stage 1: Matched Crop-Length Codec Training

Goal: test whether longer training crops improve codec reconstruction when all
other major variables are fixed.

Train the same codec architecture at the same bitrate under matched budgets:

| Arm | Crop Length | Purpose |
| --- | ---: | --- |
| A | 1 to 2 s | very local baseline, cheap smoke and sanity check |
| B | 5 s | standard short speech-codec crop |
| C | 30 s | medium-context crop with realistic utterance continuity |
| D | 120 s | long-context crop for podcasts or long-form speech |

Budget matching:

- Match total audio hours seen, optimizer steps, effective batch size, learning
  rate schedule, codec bitrate, codebook count, sample rate, and train/valid
  split.
- If memory forces smaller batches for longer crops, keep the effective number
  of waveform samples or audio seconds per optimizer update explicit in the
  result table.
- Start with one seed and a small data subset. Only run repeats if the first
  sweep shows a plausible context effect.

Primary metrics:

- Validation reconstruction losses reported by the codec codebase.
- SI-SNR or equivalent signal metric.
- Multi-scale STFT reconstruction loss.
- Codebook usage, perplexity, dead-code rate, and commitment/codebook losses.
- Real-time factor and GPU memory, because a context benefit that is not
  operationally affordable may not be useful.

Qualitative outputs:

- Decode the same held-out long recordings for every arm.
- Keep only a small index in Git; store audio outputs under the ROB-87
  parscratch artifact directory.
- Inspect boundary continuity, speaker consistency, loudness drift, background
  stability, and long-range artifacts. Do not rely on qualitative audio alone
  for the decision.

Decision rule:

- Treat longer context as useful only if medium/long crop training improves
  held-out reconstruction or codebook stability at matched bitrate and budget,
  and the improvement is visible on long recordings rather than only on train
  loss.
- Treat it as weak if the gain disappears after matching audio hours, or if
  improvements are explained by larger effective batch/audio seconds rather
  than accessible context.

## Stage 2: Effective-Context Controls

Goal: distinguish "trained on longer crops" from "uses longer context".

Controls:

- Receptive-field audit: calculate or empirically probe the codec encoder and
  decoder receptive field. If the model has no path beyond a few seconds,
  expect crop-length effects to be limited.
- Encoder/decoder context ablation: compare Local, Enc-long, Dec-long, and
  Enc+Dec-long variants at matched bitrate and parameter scale. Do not conclude
  that audio codes lack long-context benefit until at least one variant gives
  both encoder and decoder paths access to longer sequences.
- Chunk-shuffled long crop: train on long crops whose subwindows are shuffled or
  replaced across recordings. A true long-context benefit should degrade.
- Long crop with local discriminator: keep long waveform input but restrict the
  discriminator/loss to local windows. This tests whether the discriminator is
  the part using long context.
- Local crop with long-eval decode: train short, decode long recordings
  end-to-end. This separates training context from inference continuity.
- Matched memory pressure: ensure the long-crop arm is not simply worse because
  it was forced into a much smaller effective batch or unstable discriminator
  update ratio.

If Stage 1 shows no gain and the receptive-field audit says the architecture is
local, the right next experiment is an explicitly long-context codec variant,
not more crop-size repeats.

## Stage 3: Long-Context Codec Variant

Goal: test an architecture that can use longer waveform context directly.

Start with the smallest modification that can be ablated cleanly:

- Add a bottleneck-level temporal module over encoder latents before RVQ, with
  short and long attention/window settings.
- Add the same class of temporal module over quantized decoder latents before
  waveform upsampling, again with short and long settings.
- Add a cached/streaming latent context path that conditions the current chunk
  on previous encoded chunks when the target evaluation is chunked long-form
  speech.
- Only treat a long-context discriminator/feature-matching branch as a later
  auxiliary variant. A discriminator-only long-context win would not prove that
  the codec encoder or decoder itself benefits from long sequences.

Keep the comparison controlled:

- Same base codec, bitrate, data, optimizer, crop sampler, and train/valid
  split.
- Compare local-context and long-context variants at the same parameter scale
  where possible.
- Report separate Local, Enc-long, Dec-long, and Enc+Dec-long rows if compute
  allows. If compute does not allow the full ladder, prioritize Local versus
  Enc+Dec-long and record the missing ablations as residual risk.
- Report both reconstruction quality and codebook behavior. A model that
  improves waveform metrics by collapsing or underusing RVQ streams is not a
  better codec for downstream token modeling.

## Stage 4: Downstream ASR-Relevant Evaluation

Goal: decide whether the trained codec is useful for the long-context ASR
program, not only whether it reconstructs audio.

Run after Stage 1 or Stage 3 has a credible codec winner:

- Encode held-out speech with each trained codec and measure token rate,
  codebook entropy, code stability over overlapping windows, and bitrate.
- Train a small, matched codec-token predictor only as a diagnostic of token
  learnability. This is not the primary ROB-87 question.
- Optionally train or probe an ASR head from codec latents/tokens on a bounded
  labeled subset and compare WER against existing mel/BEST-RQ baselines.
- Check whether long-context-trained codec tokens are more stable across
  chunk boundaries and long recordings.

## Suggested Follow-Up Issues

1. `ROB-87a`: Stanage waveform/dependency audit for repo-local codec training.
2. `ROB-87b`: minimal in-repo codec skeleton plus one-step training smoke with
   issue-local manifests and logs.
3. `ROB-87c`: matched crop-length in-repo codec training sweep on a small
   speech subset.
4. `ROB-87d`: metric and qualitative reconstruction summary across crop arms.
5. `ROB-87e`: effective-context controls, receptive-field audit, and
   encoder/decoder context ablations.
6. `ROB-87f`: long-context codec variant with explicit encoder and decoder
   sequence modules if crop length alone does not help.
7. `ROB-87g`: downstream ASR/token-stability probe for the best trained codec.

## Launch Discipline For Follow-Up Runs

Any future training issue should:

- Run a Stanage CPU smoke against the same code path, manifest, codec
  dependency, and output paths before submitting GPU jobs.
- Use issue-specific parscratch output paths, for example
  `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-87/`.
- Include an `EXIT` callback or dependent finalizer before queueing long jobs.
- Record job id, branch, commit, script path, log paths, expected outputs, and
  completion-check command in Linear.
- Keep checkpoints, decoded audio, and metric dumps out of Git. Commit only the
  local training code, small configs, compact manifests/summaries, and any
  small follow-up documentation.

## Preliminary Recommendation

Do not make frozen-codec token prediction the first experiment. The corrected
plan is to train the codec itself under matched short-vs-long context
conditions, using a small repo-local codec recipe informed by DAC, EnCodec, and
Mimi rather than using an existing codec library. Start with a Stanage waveform
and dependency audit, add the smallest in-repo one-step training smoke, then run
a matched crop-length sweep. Treat that sweep as a baseline, not the final
answer. If the local codec does not improve from longer crops, or if any gain is
ambiguous, move to an explicit long-context codec variant with attention or an
equivalent temporal module in both encoder and decoder paths before concluding
that audio codes cannot benefit from longer context.
