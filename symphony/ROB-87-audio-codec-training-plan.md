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

- Recent Linear comments had to be fetched before planning. ROB-87 had no recent comments, so no human follow-up changed or constrained the task.
- No `Branch/ref` was supplied, so `dev` is the base branch.
- Long-running GPU work should not be launched unless the issue asks for a run. This issue asks for a preliminary plan/investigation, so this handoff is docs-only.
- Repository artifacts should stay small. Large codec tokens, decoded audio, checkpoints, W&B state, and Slurm logs should live under durable issue-specific parscratch paths, not Git.
- Documentation-only validation is `git diff --check` plus diff inspection.

## Question

Test whether audio codes benefit from longer context.

This should be made precise before queueing experiments. There are two related but different hypotheses:

1. Codec-token prediction: given a frozen audio codec tokenizer, does a longer-context model predict held-out codec tokens better?
2. Codec training/reconstruction: when training or fine-tuning the codec itself, do longer input contexts improve reconstruction, code stability, or downstream usefulness?

The lower-risk first experiment is codec-token prediction. It isolates the long-context question from codec reconstruction training complexity and reuses this repo's existing long-context scheduling patterns. Codec training or fine-tuning can follow only if the token-prediction signal is positive.

## Existing Repo Fit

Relevant local affordances:

- `exp/train.py` already supports sequence scheduling over mel frames and can grow from short chunks up to long contexts through `sequence_scheduler`.
- `lcasr/models/sconformer_xl.py` has `skip_vocab_projection=True`, returning hidden states after the long-context encoder. This is already used by `lcasr/models/BestRQ.py` to train a masked code-prediction head.
- `lcasr/models/BestRQ.py` is the closest current prototype: it masks stacked mel frames, maps targets through `RandomProjectionQuantizer`, and predicts discrete classes from SCConformer hidden states.
- `exp/configs/bin/exp_set_seq_rotary_base_multi_pred.yaml` and `exp/configs/paper_templates/exp_set_seq_window_sizes.yaml` show existing sequence-length and attention-window sweep patterns.
- The current dependency list does not include external codec packages. Any DAC, EnCodec, Mimi, or AudioCraft path should be smoke-tested in an isolated Stanage environment before launch.

Main gap:

- The repo does not yet have a dataset path where long recordings are encoded into external audio codec token streams and aligned back to the spectrogram chunks consumed by the training loop.

## Codec Choice

Recommended first frozen tokenizer: DAC 16 kHz.

Rationale:

- DAC provides pretrained 16 kHz, 24 kHz, and 44.1 kHz weights and simple encode/decode CLIs. Its README notes long files should use its `compress` and `decompress` helpers rather than one-shot encode if memory is a concern: https://github.com/descriptinc/descript-audio-codec
- The DAC paper frames codec tokens as a low-dimensional discrete representation for natural audio modeling and reports a universal 44.1 kHz, 8 kbps codec: https://proceedings.neurips.cc/paper_files/paper/2023/file/58d0e78cf042af5876e12661087bea12-Paper-Conference.pdf
- The repo's ASR data path is 16 kHz mel-spectrogram based, so DAC 16 kHz avoids introducing resampling differences into the first pass.

Alternates:

- EnCodec is a mature baseline with training code in AudioCraft. It supports monophonic 24 kHz and stereo 48 kHz models, and AudioCraft exposes compression training/evaluation docs: https://audiocraft.metademolab.com/encodec.html and https://github.com/facebookresearch/audiocraft/blob/main/docs/ENCODEC.md
- Mimi is attractive for speech/audio language modeling because it is a modern streaming codec designed for Moshi-style speech models, but it is a higher integration risk for this repo's first experiment: https://kyutai.org/codec-explainer and https://kyutai.org/

## Stage 0: Feasibility Audit

Goal: prove that codec token extraction and alignment are practical on a tiny subset before model changes.

Tasks:

- Identify 10 to 50 Spotify recordings spanning short, medium, and long durations from `/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json`.
- Resolve original waveform paths if available. If only `.spec.pt` files are available, decide whether to regenerate waveform paths from the Spotify layout or use an experiment that predicts BEST-RQ/random-projection codes from mel frames first.
- On Stanage CPU or a tiny GPU smoke, install/probe the chosen codec dependency in a disposable env or existing conda env without modifying shared data.
- Encode a few short recordings and one long recording to codec tokens.
- Record token shape, codebook count, frame/token rate, sample-rate assumptions, encode time, and disk size.
- Verify alignment between mel frames and codec frames. Store only a small JSON/CSV summary in Git if needed; keep token files under `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-87/`.

Stop criteria:

- Stop and report blocker if original waveforms cannot be resolved reliably.
- Stop and report blocker if codec installation is incompatible with the Stanage env or requires invasive changes.

## Stage 1: Frozen-Codec Token Prediction

Goal: test whether longer acoustic context improves prediction of frozen codec tokens.

Minimal implementation:

- Add a codec-token manifest builder that maps each recording id to:
  - spectrogram path
  - transcript path, retained only for future downstream checks
  - codec token path
  - codec metadata: sample rate, token rate, codebook count, model name, model checkpoint/hash if available
- Add a dataloader or dataset wrapper that returns aligned mel chunks and codec-token chunks.
- Add a `CodecTokenPredictor` model wrapper patterned after `BestRQ`, but with targets loaded from frozen codec tokens rather than generated by `RandomProjectionQuantizer`.
- Support independent loss per RVQ stream. Start with either:
  - first-codebook-only CE loss for the semantic/coarse stream, or
  - sum/mean CE over the first `N` streams, with `N` fixed in config.
- Log loss, per-stream accuracy, top-k accuracy, perplexity, and token entropy.

Initial sweep:

| Arm | Max context | Attention | Notes |
| --- | ---: | --- | --- |
| A | 512 mel frames | full within chunk | short baseline, about 5 s |
| B | 4096 mel frames | full within chunk | medium baseline, about 41 s |
| C | 16384 mel frames | full within chunk | long baseline, about 164 s |
| D | 65536 mel frames | windowed or scheduled | stress long context, about 11 min |

Use one seed first. Add repeats only after the smoke and first full run establish stable loss curves.

Primary metric:

- Held-out codec-token negative log likelihood by codebook and context length.

Secondary metrics:

- Top-1/top-5 token accuracy by codebook.
- Perplexity normalized by codebook entropy.
- Loss as a function of within-recording position, to check whether long context helps later chunks more than first chunks.
- Optional frozen-codec reconstruction of predicted tokens on a small qualitative subset. This should not be the first gating metric because token sampling/argmax can confound the context question.

Decision rule:

- Treat longer context as useful if the 16k or 65k arm improves validation NLL over 512 and 4096 by a meaningful margin, especially on later chunks and coarse codebooks, without unstable training or large overfit.
- Treat the result as weak or negative if improvement appears only on fine codebooks, only on train loss, or disappears under matched token budgets.

## Stage 2: Context Controls

Goal: separate genuine long-context use from easier optimization or more compute.

Controls:

- Same token budget: compare runs at equal optimizer steps and equal seen audio hours.
- Shuffled history: preserve local chunk content but replace previous context with a different recording. If long context helps, this should degrade.
- Local attention cap: train with long chunks but restrict attention window. This distinguishes long batch shape from accessible context.
- Position-only control: keep long sequences but remove or perturb usable prior audio. This checks whether gains are from sequence length artifacts.
- Codebook depth: compare first 1, first 4, and all codebooks. Coarse streams should be more context-sensitive than fine acoustic detail if the effect is semantic.

## Stage 3: Downstream ASR Probe

Goal: check whether better codec-token modeling transfers to ASR-relevant representations.

Options:

- Initialize or auxiliary-train SCConformer from the best codec-token predictor, then evaluate TEDLIUM/Earnings-style WER using the existing eval harness.
- Freeze the encoder and train a small CTC/readout head on a limited labeled subset.
- Compare against the existing BEST-RQ/random-projection pretraining baseline and supervised ASR checkpoints where available.

This stage should not run until Stage 1 shows a clear token-prediction signal.

## Stage 4: Codec Fine-Tuning Or Training

Goal: test the harder hypothesis: whether codec reconstruction training itself benefits from longer context.

Recommended only after frozen-token experiments:

- Start from DAC or EnCodec training code rather than reimplementing a waveform codec inside `lcasr`.
- Train or fine-tune on a small speech-only subset first.
- Compare short-window and long-window training under matched data, steps, and bitrate.
- Evaluate reconstruction with SI-SNR and, if available, ViSQOL or another perceptual metric; add codebook usage/collapse statistics.

Risks:

- Codec reconstruction quality may be dominated by local waveform modeling and adversarial loss details, not long context.
- Training full neural codecs is substantially more expensive and operationally different from this repo's current mel/CTC training loop.
- Cross-repo codec training may make PR review harder. Keep this stage as a separate follow-up issue if Stage 1 justifies it.

## Suggested Implementation Tasks

1. `ROB-87a`: codec dependency and waveform/token feasibility smoke.
2. `ROB-87b`: manifest builder for frozen codec tokens.
3. `ROB-87c`: codec-token dataloader and tiny CPU/GPU smoke config.
4. `ROB-87d`: `CodecTokenPredictor` wrapper plus config template.
5. `ROB-87e`: one-seed context sweep with callback-backed Stanage jobs.
6. `ROB-87f`: summarize token NLL/perplexity/accuracy by context and decide whether to run repeats or downstream ASR transfer.

## Launch Discipline For Follow-Up Runs

Any future training issue should:

- Run a Stanage CPU smoke against the same code path, manifest, codec dependency, and output paths before submitting GPU jobs.
- Use issue-specific parscratch output paths, for example `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-87/`.
- Include an `EXIT` callback or finalizer before queueing long jobs.
- Record job id, branch, commit, script path, log paths, expected outputs, and completion-check command in Linear.

## Preliminary Recommendation

Do not start by training a codec from scratch. Start by freezing DAC 16 kHz as the tokenizer and training a long-context codec-token predictor inside this repo. That gives a direct, controlled answer to whether codec codes benefit from longer context while keeping the first implementation close to existing SCConformer, BEST-RQ, sequence-scheduler, and Stanage launch patterns.
