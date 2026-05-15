# ROB-88 Codec Architecture And Mixer Cost Probe

Issue: ROB-88, "ROB-87a: specify the repo-local conv codec architecture and mixer cost probe"

Date: 2026-05-15

## Instruction And Context Check

Read the required instruction files before planning or editing, in order:

1. `symphony/instructions/linear-context.md`
2. `symphony/instructions/repository.md`
3. `symphony/instructions/work-loop.md`
4. `symphony/instructions/experiment-execution.md`
5. `symphony/instructions/validation-and-handoff.md`

Instructions that directly affect this issue:

- Recent Linear comments must be fetched before planning, progress comments, and handoff. ROB-88 had no existing comments when this plan was written, so no later comment changed the issue constraints.
- No `Branch/ref` was supplied, so `dev` is the base branch and this work uses `symphony/ROB-88-codec-architecture-probe`.
- No long GPU work should be launched without explicit approval. The cost probe here is a docs-plus-analytic synthetic estimate with a bounded CPU script path, not a training run.
- Keep large logs, checkpoints, decoded audio, W&B state, and bulky benchmark artifacts out of Git.
- Validation for this change is diff inspection plus `git diff --check`; because a small probe script is included, also run its smallest CPU/estimate path.

Parent context was taken from PR #14's `symphony/ROB-87-audio-codec-training-plan.md`. That plan already corrected the program toward training a repo-local codec rather than evaluating an existing codec library. ROB-88 therefore specifies the first implementation target and decides whether the first long-context variant is plausible before doing the OGG/`torchaudio` audit or adding the codec skeleton.

## Recommendation

Implement `SpeechCodec-base-local` first, then `SpeechCodec-base-encdec-ssm` as the first long-context variant.

The base codec should use 24 kHz mono waveform input, cumulative encoder stride 320, 75 Hz latents, eight 1024-entry RVQ codebooks, and a non-adversarial reconstruction objective for the first smoke. This gives a 6.0 kbps codec target and 600 code tokens per second if every RVQ stream is treated as a token sequence.

The first long-context variant should add one shallow state-space or state-space-like gated temporal mixer before RVQ and the same mixer after RVQ projection in the decoder. Use the bottleneck length at stride 320. Do not add stride-80 or raw-waveform mixers in the first implementation. The current estimates say the bottleneck Enc+Dec-long path is affordable enough to test, while earlier or later stride-160 insertion should be treated as a second step after the skeleton and OGG audit pass.

The next issue should be the OGG/`torchaudio` audit, not the minimal codec skeleton yet. The architecture choice is concrete enough, but the next blocker is whether the Stanage environment can decode current OGG sources to waveform on the fly and sample crops cheaply enough.

## SpeechCodec-base

| Field | Spec |
| --- | --- |
| Name | `SpeechCodec-base` |
| Input | mono waveform, loaded from OGG with `torchaudio`, resampled to 24 kHz, amplitude normalized per crop or per recording after the audit picks the safer behavior |
| Encoder | 1D convolutional residual stack with downsample strides `[5, 4, 4, 4]`, cumulative stride 320 |
| Encoder channels | `[64, 128, 256, 384]`, projected to `d_model=256` before RVQ |
| Latent frame rate | `24000 / 320 = 75 Hz` |
| RVQ | 8 residual codebooks, 1024 entries each, 256-d code vectors |
| Bitrate | `75 frames/s * 8 codebooks * log2(1024) = 6000 bit/s` |
| Token rate | 75 frame steps/s, 600 scalar code tokens/s if streams are flattened |
| Decoder | RVQ code embedding sum/projection to `d_model=256`, mirror transposed-conv or upsample-plus-conv stack with strides `[4, 4, 4, 5]` |
| First losses | waveform L1, multi-resolution STFT magnitude+log-magnitude, RVQ commitment/codebook loss, codebook usage/perplexity metrics |
| Later losses | adversarial multi-period or multi-scale discriminator plus feature matching only after one-step non-adversarial training is stable |
| First training mode | non-streaming full crop, causal-compatible convolution padding where practical, no discriminator in the first smoke |

The stride schedule is deliberately simple. Cumulative stride 320 is the default because it keeps the base target near a familiar speech-codec bitrate while making 30 s and 120 s latent sequences tractable. Stride 160 is retained as an insertion-site probe, not as the first bottleneck. Stride 640 is retained for the heavier LLM target and for a lower-cost fallback if 320 proves too expensive after real Stanage measurements.

## Mixer Insertion Sites

| Site | Location | First use |
| --- | --- | --- |
| Early encoder | after cumulative stride 160, before the final downsample/projection | probe only; possible later if bottleneck context is too weak |
| Bottleneck encoder | after stride 320 encoder projection, before RVQ | first Enc+Dec-long encoder insertion |
| Bottleneck decoder | after RVQ lookup/sum and decoder projection, before waveform upsampling | first Enc+Dec-long decoder insertion |
| Late decoder | after partial upsampling to stride 160, before final waveform-rate decoder blocks | probe only; defer until bottleneck decoder is stable |

Exact encoder path for the first implementation:

1. `waveform -> conv_s5 -> conv_s4 -> conv_s4 -> conv_s4 -> projection`
2. Optional `encoder_mixer(latents_75hz)` for `SpeechCodec-base-encdec-ssm`
3. `rvq(latents_75hz)`

Exact decoder path:

1. `rvq_codes -> codebook_lookup_sum -> projection`
2. Optional `decoder_mixer(quantized_latents_75hz)` for `SpeechCodec-base-encdec-ssm`
3. `up_s4 -> up_s4 -> up_s4 -> up_s5 -> waveform`

Use one mixer block per side first. Use activation checkpointing only if a real Stanage smoke shows the codec stack, not the synthetic mixer alone, is memory-bound. Keep the mixer hidden size equal to the latent width so the RVQ interface does not change between Local and Enc+Dec-long variants.

## Secondary Compression Target

Keep this target secondary until `SpeechCodec-base` can decode one real OGG crop and train one step.

| Target | Sample rate | Stride | Latent rate | RVQ | Bitrate | Token rate | Purpose |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `SpeechCodec-base` | 24 kHz | 320 | 75 Hz | 8 x 1024 | 6.00 kbps | 600/s | credible repo-local speech codec baseline |
| `SpeechCodec-llm` | 24 kHz | 640 | 37.5 Hz | 4 x 512 | 1.35 kbps | 150/s | faster LLM generation or code prediction |

The LLM target should reuse the same model with config-only changes where possible. If reconstruction or intelligibility collapses at 1.35 kbps, try 6 x 512 at 2.025 kbps before changing the architecture.

## Probe Assumptions

The included helper `symphony/codec_mixer_cost_probe.py` estimates the standalone mixer cost for the required crop/stride grid. It does not claim to measure full codec training memory. Assumptions:

- sample rate: 24 kHz
- dtype: bf16/fp16 activation storage
- batch size: 1 crop
- mixer blocks: 1 per insertion site
- early/late stride-160 sites use `d_model=192`
- bottleneck stride-320/640 sites use `d_model=256`
- memory is per insertion site and excludes convolutional activations, RVQ tables, discriminator, optimizer state, and dataloader buffers
- wall-time is a conservative single-block forward+backward GPU estimate for ranking sites; real Stanage timing should be measured in the OGG audit or codec skeleton issue before a training sweep

Command:

```bash
python symphony/codec_mixer_cost_probe.py --estimate-only --format markdown
```

## Latent-Length Grid

At 24 kHz, the required first grid is:

| Stride | Latent rate | 5 s | 30 s | 120 s | Interpretation |
| ---: | ---: | ---: | ---: | ---: | --- |
| 160 | 150 Hz | 750 | 4500 | 18000 | useful for early/late probes, but not the first bottleneck |
| 320 | 75 Hz | 375 | 2250 | 9000 | first base bottleneck and Enc+Dec-long target |
| 640 | 37.5 Hz | 188 | 1125 | 4500 | cheap fallback and LLM-compression target |

Stride 80 is intentionally excluded from the first architecture. At 24 kHz it would produce 1500, 9000, and 36000 latent steps for 5 s, 30 s, and 120 s crops. That is not raw-waveform scale, but it is early enough in the codec that convolutional feature activations and any two-sided long-context variant would waste memory before the base path is proven.

## Cost Probe Summary

Key rows from the analytic grid:

| Site | Stride | Crop | Latents | d_model | SSM train MiB | Linear-attn train MiB | SSM ms | Linear-attn ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| early_encoder | 160 | 5s | 750 | 192 | 4.9 | 8.2 | 8 | 13 |
| early_encoder | 160 | 30s | 4500 | 192 | 29.7 | 49.4 | 46 | 78 |
| early_encoder | 160 | 120s | 18000 | 192 | 118.7 | 197.8 | 182 | 314 |
| bottleneck_encoder | 320 | 5s | 375 | 256 | 3.3 | 5.5 | 7 | 12 |
| bottleneck_encoder | 320 | 30s | 2250 | 256 | 19.8 | 33.0 | 40 | 70 |
| bottleneck_encoder | 320 | 120s | 9000 | 256 | 79.1 | 131.8 | 162 | 279 |
| bottleneck_decoder | 320 | 5s | 375 | 256 | 3.3 | 5.5 | 7 | 12 |
| bottleneck_decoder | 320 | 30s | 2250 | 256 | 19.8 | 33.0 | 40 | 70 |
| bottleneck_decoder | 320 | 120s | 9000 | 256 | 79.1 | 131.8 | 162 | 279 |
| late_decoder | 160 | 5s | 750 | 192 | 4.9 | 8.2 | 8 | 13 |
| late_decoder | 160 | 30s | 4500 | 192 | 29.7 | 49.4 | 46 | 78 |
| late_decoder | 160 | 120s | 18000 | 192 | 118.7 | 197.8 | 182 | 314 |
| bottleneck_encoder | 640 | 120s | 4500 | 256 | 39.6 | 65.9 | 81 | 140 |

For the first Enc+Dec-long variant at stride 320, add the bottleneck encoder and decoder rows. The mixer-only estimate for 120 s crops is therefore about 158 MiB and 324 ms for two SSM blocks, or about 264 MiB and 558 ms for two linear-attention blocks. These are small enough that the codec convolution stack, RVQ implementation, discriminator, and batch size are more likely to dominate the first real memory decision.

This result does not prove training will fit. It does justify implementing the first long-context variant at the bottleneck rather than declaring the long-context mixer too expensive.

## State-Space Versus Linear Attention

Choose a state-space or state-space-like gated temporal mixer first.

Reasons:

- It has the lower estimated activation footprint and wall time at every required grid point.
- The repo already has a `lcasr/models/mamba.py` path and mamba-oriented configs, so a real implementation can inspect existing local conventions before deciding whether to use external `mamba_ssm` kernels or a repo-local fallback.
- It is the safer match for 120 s crops, where the first question is whether a long-context path can run without distorting the codec interface.

Keep linear attention as the comparison block in the probe and as a backup implementation if mamba-style dependencies are not available on Stanage. Do not start with softmax attention. If a reviewer wants linear attention measured directly, use the helper's CPU benchmark mode first and then a Stanage CPU smoke before any GPU job:

```bash
python symphony/codec_mixer_cost_probe.py --cpu-benchmark-tokens 256 --cpu-benchmark-d-model 128
```

The CPU benchmark mode is a shape/import smoke and rough relative sanity check only. The report's ranking uses the analytic grid until a Stanage measurement exists.

## Selected First Configs

| Config | Codec | Encoder mixer | Decoder mixer | Stride | Crop targets | Purpose |
| --- | --- | --- | --- | ---: | --- | --- |
| `SpeechCodec-base-local` | base 6.0 kbps RVQ codec | none | none | 320 | 5 s smoke, then 30 s/120 s shape checks | prove repo-local codec path and local crop baseline |
| `SpeechCodec-base-encdec-ssm` | same base codec | one bottleneck SSM/gated mixer before RVQ | one bottleneck SSM/gated mixer after RVQ projection | 320 | 5 s smoke, then 30 s/120 s shape checks | first long-context architecture test |

If the SSM dependency is missing, the first implementation should provide a small repo-local gated depthwise temporal mixer behind the same config interface and record the dependency gap. If both SSM and linear attention fail at the 30 s bottleneck CPU/GPU smoke, run only `SpeechCodec-base-local` and make the next issue a revised architecture probe. Based on the estimates above, that fallback is unlikely to be necessary because the bottleneck mixer is not the dominant cost.

## Next Issue

The next issue should be the OGG/`torchaudio` audit. It should answer:

- where the current source OGG files live relative to existing Spotify manifests
- whether `torchaudio` on Stanage can load them directly
- decode speed for 5 s, 30 s, and 120 s crop sampling
- sample-rate distribution and resampling policy
- whether duration metadata can be read without full decode
- whether a one-recording waveform crop can reach the planned codec input tensor shape

Only after that audit should the repo-local codec skeleton be added. The skeleton needs the real waveform loading contract to avoid building the model against an imaginary data path.

## Validation

Smoke-test status: this ROB-88 work was not extensively smoke tested as a
runnable codec setup. It is an architecture specification plus a synthetic
cost-probe sanity check. The validation below exercises the probe's analytic
output path, a bounded CPU shape/import benchmark, script compilation, and diff
hygiene. It does not exercise real OGG/`torchaudio` loading on Stanage, a full
encoder/RVQ/decoder forward-backward pass, GPU memory/timing, or a training
config launch. Those checks belong in the OGG/`torchaudio` audit and minimal
codec skeleton issues.

Planned validation for this ROB-88 change:

```bash
python symphony/codec_mixer_cost_probe.py --estimate-only --format markdown
python symphony/codec_mixer_cost_probe.py --cpu-benchmark-tokens 128 --cpu-benchmark-d-model 64
python -m py_compile symphony/codec_mixer_cost_probe.py
git diff --check
```
