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
  the recipe; the codec will likely need explicit attention or equivalent
  long-sequence machinery in both encoder and decoder paths to benefit from
  long sequences; the current audio source is OGG and should be loaded on the
  fly to waveform with `torchaudio`; the first long-context test should be an
  encoder-plus-decoder context model; and the architecture plan needs two
  compression targets, one comparable to Mimi/EnCodec-style compression and
  one heavier-compression variant for faster LLM generation. The newest Linear
  comment asks for a casual setup that uses state-space or linear-attention
  sequence mixers and convolutional downsampling, with a small investigation of
  the cost of adding the mixer at different layers and sequence lengths. This
  plan therefore
  centers on a repo-local codec-training recipe under short-context and
  long-context conditions, with architecture controls that separate longer crops
  from real encoder/decoder context. Frozen-token prediction is only a
  diagnostic, not the main recommendation.
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
- The expected source audio is currently OGG. This is usable for codec training
  if the follow-up smoke confirms `torchaudio` can load it on the fly to
  waveform from the Stanage training environment. The first data-path audit
  should therefore test OGG decode, sample-rate normalization, duration
  extraction, and crop sampling rather than treating non-WAV storage as a
  blocker.

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

## Architecture Targets

Before writing the training harness, define two related codec targets. Both
should share as much implementation as possible so that changes in compression
level are explicit config choices rather than separate code paths.

| Target | Compression Goal | Intended Use | First Knobs |
| --- | --- | --- | --- |
| SpeechCodec-base | comparable to Mimi/EnCodec-style speech compression | quality-controlled baseline for ASR-relevant speech reconstruction and token stability | sample rate, encoder stride, latent frame rate, RVQ depth, codebook size, bitrate |
| SpeechCodec-llm | heavier compression than the base target | faster audio generation or prediction with an LLM over fewer tokens | lower latent frame rate, fewer RVQ streams, smaller codebooks, stronger intelligibility-weighted losses |

The base target should come first, because it is the sanity check that the
repo-local recipe can train a credible codec at a known compression regime. The
heavier target should reuse the same code after the base smoke is stable, then
trade reconstruction quality against token rate, code stability, intelligibility,
and downstream modeling cost.

Architecture choices to specify before the first GPU run:

- Input path: OGG loaded on the fly with `torchaudio`, resampled to a fixed
  codec sample rate, then cropped without materializing rewritten waveform
  copies.
- Encoder/decoder: causal or noncausal convolutional stack with a documented
  receptive field, plus optional bottleneck-level context modules.
- Quantizer: RVQ with explicit codebook count, codebook size, commitment weight,
  replacement policy for dead codes, and per-stream usage metrics.
- Losses: start with waveform L1/L2 plus multi-scale STFT. Add adversarial,
  feature-matching, or perceptual losses only after the non-adversarial smoke
  can train and decode reliably.
- Compression reporting: always report latent frame rate, RVQ streams, bits per
  second, tokens per second for LLM consumption, and reconstruction quality.

## Casual Conv-Plus-Mixer Setup

The first architecture should stay casual and cheap enough to debug:

- Use strided 1D convolutions to reduce waveform length before any global or
  semi-global sequence mixer. Do not put state-space or linear attention over
  raw waveform samples.
- Put a small temporal mixer at the compressed latent rate, before RVQ on the
  encoder side and after RVQ projection on the decoder side.
- Treat repo-local Mamba/state-space code as a candidate if the Stanage env has
  the required `mamba_ssm` and `causal_conv1d` kernels. If those kernels are
  not available, use a simple linear-attention or gated-conv fallback for the
  initial cost probe rather than blocking the plan on environment surgery.
- Keep the first mixer shallow: one block at the bottleneck before trying
  multiple insertion depths. The goal is to see whether a long-context path is
  operationally plausible, not to design a final codec architecture.

Concrete starting variant:

1. Waveform to conv encoder with cumulative stride chosen from the compression
   target, for example 160, 320, or 640 samples.
2. Optional 1 to 2 mixer blocks on encoder latents before RVQ.
3. RVQ bottleneck with the same codebook settings as the Local baseline.
4. Optional 1 to 2 mixer blocks on decoder latents after code lookup/projection.
5. Conv decoder back to waveform.

For SpeechCodec-base, the first long-context test should use the bottleneck
site only. For SpeechCodec-llm, also test one earlier encoder insertion if the
heavier compression makes the bottleneck too sparse to carry useful timing or
speaker context.

## Sequence-Mixer Cost Investigation

Before running a training sweep, benchmark the mixer cost independently from
codec quality. Use synthetic tensors and one real OGG crop, then report wall
time, peak GPU memory, activation memory under training, and output shapes.

Use this sequence-length table as the first cost grid. Values assume a 24 kHz
codec sample rate; if Stage 0 chooses 16 kHz, recompute the same table in the
architecture spec.

| Cumulative Stride | Latent Rate At 24 kHz | 5 s | 30 s | 120 s | Initial Interpretation |
| ---: | ---: | ---: | ---: | ---: | --- |
| 80 | 300 Hz | 1,500 | 9,000 | 36,000 | too early for global linear attention; possible only for shallow SSM or local windows |
| 160 | 150 Hz | 750 | 4,500 | 18,000 | plausible for SSM; linear attention needs careful memory check |
| 320 | 75 Hz | 375 | 2,250 | 9,000 | good first bottleneck cost point |
| 640 | 37.5 Hz | 188 | 1,125 | 4,500 | cheap enough for long-crop sweeps, but may discard too much local detail |

Benchmark variants:

| Site | Example Tensor | Mixer Options | Why |
| --- | --- | --- | --- |
| Early encoder | stride 80 or 160 latents | SSM only, or linear attention with short feature dim | finds the cost of making codes depend on pre-bottleneck context |
| Bottleneck encoder | stride 320 or 640 latents | SSM and linear attention | primary long-context encoder candidate before RVQ |
| Bottleneck decoder | same latent length as RVQ output | SSM and linear attention | primary long-context decoder candidate after codes |
| Two-sided | encoder plus decoder bottleneck mixers | selected cheapest working mixer | estimates the real Enc+Dec-long overhead |

Cost reporting should include:

- `d_model`, number of mixer blocks, state size or attention feature size, dtype,
  batch size, crop seconds, cumulative stride, and latent length.
- Forward-only milliseconds, forward/backward milliseconds, and peak GPU memory.
- Parameter count and optimizer-state estimate for each mixer insertion.
- Whether activation checkpointing is needed to train 30 s and 120 s crops.
- Whether the mixer can run causally or with a cache, because noncausal full
  recording context is an upper bound rather than a deployable streaming path.

Decision rule:

- Prefer a state-space mixer if it gives stable linear scaling across the
  30 s and 120 s rows without large dependency friction on Stanage.
- Prefer linear attention only if its implementation is already available or
  easy to add locally, and its training memory is acceptable at the selected
  bottleneck stride.
- Do not add mixers at stride 80 unless the cost probe shows the benefit is
  affordable. That insertion point has the strongest chance to help code
  assignment, but it is also the easiest place to waste memory.

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
| Enc+Dec-long | long-context encoder | long-context decoder | first long-context test, because both sides may need sequence access before a benefit appears |
| Enc-long | attention/state/cached context before RVQ | local | tests whether better codes need long context |
| Dec-long | local | attention/state/cached context after RVQ | tests whether reconstruction continuity needs long code history |

The first architecture comparison should be Local versus Enc+Dec-long. If that
shows a plausible gain or an unclear failure, split the long-context model into
Enc-long and Dec-long ablations to identify whether code assignment, waveform
reconstruction, or both are responsible. This ordering avoids prematurely
discarding long context just because only one side of the codec was modified.

## Stage 0: Feasibility Audit

Goal: prove that codec training inputs, dependencies, and a tiny training step
are practical before designing a full sweep.

Tasks:

- On Stanage, verify the current OGG source path for Spotify or another
  long-recording speech dataset and confirm `torchaudio` can load it on the fly
  to waveform in the intended environment. Record codec/container, sample rate,
  duration extraction behavior, decode speed, and any resampling assumptions.
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

- Stop and report blocker if the OGG files cannot be decoded through
  `torchaudio` on Stanage or if only mel spectrograms are reachable from the
  intended training environment.
- Stop and report blocker if the local codec dependencies cannot be satisfied
  on Stanage without invasive shared-environment changes.
- Stop and report blocker if a one-step training smoke fails before the model
  reaches forward/backward.

## Stage 1: Base Architecture And Compression Targets

Goal: turn the architecture targets above into a concrete config matrix before
spending GPU time on crop-length or long-context sweeps.

Deliverables:

- A compact architecture spec for `SpeechCodec-base` and `SpeechCodec-llm`:
  sample rate, encoder stride, latent frame rate, RVQ stream count, codebook
  size, approximate bitrate, expected tokens per second, loss set, and decoder
  upsampling shape.
- A mixer cost report for the candidate insertion sites above. The report
  should be produced before any GPU training sweep and should make the
  sequence-length cost of state-space versus linear-attention options explicit.
- A documented Local baseline and Enc+Dec-long variant for the base target. The
  first long-context model should put the same class of context module on both
  encoder latents before RVQ and quantized decoder latents before upsampling.
- A note identifying which values are chosen to resemble Mimi/EnCodec-style
  compression and which values intentionally push heavier compression for faster
  LLM generation.
- A one-recording shape test that checks OGG decode, crop sampling, forward
  reconstruction length, RVQ tensor shapes, token rate, and loss computation.

Decision rule:

- Do not launch a training sweep until the architecture table makes the two
  compression targets explicit, the mixer cost report has a viable bottleneck
  insertion site, and the one-recording shape test passes.
- If the base target cannot reconstruct and quantize one OGG crop with stable
  tensor shapes, fix that before designing the heavier LLM-compression target.
- If neither state-space nor linear attention is affordable at the bottleneck
  latent rate for 30 s crops, keep the first training sweep Local-only and
  record long-context architecture as blocked by cost rather than by codec
  quality.

## Stage 2: Matched Crop-Length Codec Training

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

## Stage 3: Effective-Context Controls

Goal: distinguish "trained on longer crops" from "uses longer context".

Controls:

- Receptive-field audit: calculate or empirically probe the codec encoder and
  decoder receptive field. If the model has no path beyond a few seconds,
  expect crop-length effects to be limited.
- Encoder/decoder context ablation: first compare Local against Enc+Dec-long at
  matched bitrate and parameter scale. If that result is promising or ambiguous,
  run Enc-long and Dec-long ablations to identify which side matters. Do not
  conclude that audio codes lack long-context benefit until at least one variant
  gives both encoder and decoder paths access to longer sequences.
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

If Stage 2 shows no gain and the receptive-field audit says the architecture is
local, the right next experiment is an explicitly long-context codec variant,
not more crop-size repeats.

## Stage 4: Long-Context Codec Variant

Goal: test an architecture that can use longer waveform context directly.

Start with the smallest modification that can be ablated cleanly:

- Add a bottleneck-level state-space or linear-attention module over encoder
  latents before RVQ, with short and long context settings.
- Add the same class of temporal module over quantized decoder latents before
  waveform upsampling, again with short and long context settings.
- Only test earlier encoder insertions after the cost probe says the sequence
  length is affordable. Early insertion may improve code assignment, but it is
  not the first training target.
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
- Report separate Local, Enc+Dec-long, Enc-long, and Dec-long rows if compute
  allows. If compute does not allow the full ladder, prioritize Local versus
  Enc+Dec-long and record the missing single-sided ablations as residual risk.
- Report both reconstruction quality and codebook behavior. A model that
  improves waveform metrics by collapsing or underusing RVQ streams is not a
  better codec for downstream token modeling.

## Stage 5: Downstream ASR-Relevant Evaluation

Goal: decide whether the trained codec is useful for the long-context ASR
program, not only whether it reconstructs audio.

Run after Stage 2 or Stage 4 has a credible codec winner:

- Encode held-out speech with each trained codec and measure token rate,
  codebook entropy, code stability over overlapping windows, and bitrate.
- Train a small, matched codec-token predictor only as a diagnostic of token
  learnability. This is not the primary ROB-87 question.
- Optionally train or probe an ASR head from codec latents/tokens on a bounded
  labeled subset and compare WER against existing mel/BEST-RQ baselines.
- Check whether long-context-trained codec tokens are more stable across
  chunk boundaries and long recordings.

## Suggested Follow-Up Issues

1. `ROB-87a`: architecture spec for two repo-local codec targets: base
   Mimi/EnCodec-like compression and heavier LLM-generation compression,
   including the conv downsampling schedule.
2. `ROB-87b`: Stanage OGG/`torchaudio` decode and dependency audit for
   repo-local codec training.
3. `ROB-87c`: state-space versus linear-attention mixer cost probe across
   encoder/decoder insertion sites and 5 s, 30 s, and 120 s latent lengths.
4. `ROB-87d`: minimal in-repo codec skeleton plus one-step training smoke with
   issue-local manifests and logs.
5. `ROB-87e`: matched crop-length in-repo codec training sweep on a small
   speech subset.
6. `ROB-87f`: metric and qualitative reconstruction summary across crop arms.
7. `ROB-87g`: effective-context controls, receptive-field audit, and
   Local-versus-Enc+Dec-long comparison, followed by single-sided ablations if
   needed.
8. `ROB-87h`: heavier-compression LLM-generation variant after the base target
   has a credible training and evaluation path.
9. `ROB-87i`: downstream ASR/token-stability probe for the best trained codec.

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
Mimi rather than using an existing codec library. Start by specifying two
architecture targets: a base target at Mimi/EnCodec-like compression and a
heavier-compression target for faster LLM generation. Use convolutional
downsampling first, then test state-space or linear-attention sequence mixers
at the latent rates where the cost probe says 30 s and 120 s crops are
practical. Then validate the Stanage OGG-to-waveform path with `torchaudio`,
add the smallest in-repo one-step training smoke, and run a matched crop-length
sweep. Treat that sweep as a baseline, not the final answer. The first
long-context architecture test should compare Local against Enc+Dec-long, with
the selected state-space or linear-attention mixer in both encoder and decoder
paths; only then split into single-sided ablations if needed.
