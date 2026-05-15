# ROB-89 OGG Torchaudio Waveform Loading Audit

Issue: ROB-89, "ROB-88a: audit OGG/torchaudio waveform loading for repo-local codec"

Date: 2026-05-15

## Instruction And Comment Check

Read the required instruction files before planning or editing, in order:

1. `symphony/instructions/linear-context.md`
2. `symphony/instructions/repository.md`
3. `symphony/instructions/work-loop.md`
4. `symphony/instructions/experiment-execution.md`
5. `symphony/instructions/validation-and-handoff.md`

Instructions that directly affected this issue:

- Recent Linear comments must be fetched before planning, progress comments, and handoff. ROB-89 had no comments before planning, and no new human clarification was present before the progress update.
- No `Branch/ref` was supplied, so `dev` was used as the base branch and this work uses `symphony/ROB-89-ogg-torchaudio-audit`.
- Stanage is the default execution target for bounded data-path checks. No GPU work was launched.
- Large logs, decoded audio, and generated JSON artifacts stay out of Git; this report references durable Stanage artifact paths instead.
- Because this change adds a small audit helper, validation includes script compilation, the Stanage CPU smoke, diff inspection, and `git diff --check`.

Parent context was taken from ROB-88 PR #15 because its report was still in the open PR branch rather than current `dev`. ROB-88 selected `SpeechCodec-base`: 24 kHz mono waveform input, cumulative stride 320, 75 Hz latents, and 5 s / 30 s / 120 s crop checks before adding the minimal repo-local codec skeleton.

## Source Path Contract

The current Spotify code path is spectrogram-first:

- `eval/spotify/run.py` calls `load_pairs()` and loads `sample["audio"]` with `torch.load(...)`.
- `lcasr/artifacts/spotify_long_only.json` stores `.spec.pt` paths such as `/mnt/parscratch/users/acp21rjf/spotify/audio/0/S/show_.../<episode>.spec.pt`.
- `lcasr.utils.audio_tools.pair_audio_txt(...)` pairs audio from `/mnt/parscratch/users/acp21rjf/spotify/audio/` and transcript JSON from `/mnt/parscratch/users/acp21rjf/spotify/txt/spotify-podcasts-2020/podcasts-transcripts/`.
- `lcasr.utils.audio_tools.append_timings_to_json(...)` already maps a manifest `.spec.pt` entry to its source OGG by replacing `.spec.pt` with `.ogg`.

The source OGG contract for the codec skeleton should therefore be:

```text
/mnt/parscratch/users/acp21rjf/spotify/audio/<top>/<letter>/show_<show_id>/<episode_id>.ogg
```

for each existing manifest audio path:

```text
/mnt/parscratch/users/acp21rjf/spotify/audio/<top>/<letter>/show_<show_id>/<episode_id>.spec.pt
```

A quick Stanage manifest check confirmed `/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json` exists with 105,360 records and sampled sibling `.ogg` paths exist.

## Stanage CPU Audit

Script:

```bash
python symphony/rob89_ogg_torchaudio_audit.py \
  --metadata-records 16 \
  --candidate-scan-limit 512 \
  --crop-seconds 5 30 120 \
  --output /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-89/rob89_ogg_torchaudio_audit.json
```

Final Slurm smoke:

- Job: `10222708`
- Partition: `interactive`
- State: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `00:00:53`
- Script: `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-89/rob89_ogg_torchaudio_audit_v2.sbatch`
- Output JSON: `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-89/rob89_ogg_torchaudio_audit.json`
- Logs: `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-89/rob89-ogg-audit-10222708.{out,err}`

An earlier job, `10222333`, was cancelled after `00:06:07` because the first helper version attempted too many filesystem existence checks before reaching the decode measurements. The committed helper now performs a bounded candidate scan and records that limit explicitly.

## Metadata Results

Bounded sample: first 16 manifest records found within a 512-record scan that had duration at least 120 s and an existing sibling OGG.

| Field | Result |
| --- | --- |
| Manifest path | `/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json` |
| Manifest records | 105,360 |
| Candidate scan limit | 512 |
| Existing OGG records found | 16 |
| Sample rate counts | 16 x 44,100 Hz |
| Channel counts | 16 x stereo |
| Encoding counts | 16 x `VORBIS` |
| `torchaudio.info` duration metadata | available for all 16 sampled OGGs |
| `torchaudio.info` time | min 296 ms, median 2,431 ms, max 6,705 ms |

The sample-rate result is a bounded audit sample, not a full-corpus scan. A full 105k-file metadata pass would be too large for this issue and should not be part of the minimal codec skeleton. The initial codec loader should assume source OGGs may be 44.1 kHz stereo Vorbis and normalize every crop to the codec contract.

Selected real recording:

```text
key: 4_L_show_4LHI4R4U72hQnYQITwQt3Q_2jSYn7uwJ217wABg0r5yUU
ogg: /mnt/parscratch/users/acp21rjf/spotify/audio/4/L/show_4LHI4R4U72hQnYQITwQt3Q/2jSYn7uwJ217wABg0r5yUU.ogg
manifest duration: 368.063991 s
torchaudio.info duration: 368.0639909297052 s
source sample rate: 44,100 Hz
source frames: 16,231,622
```

## Decode And Crop Results

The smoke used `torchaudio.load(path, frame_offset=0, num_frames=<crop_frames>)`, then collapsed stereo to mono by channel mean and resampled to 24 kHz. The resulting tensor shape is the planned `SpeechCodec-base` input shape `[batch, channel, samples]`.

| Crop | Source load shape | `torchaudio.load` time | 24 kHz codec input shape | Resample time |
| ---: | --- | ---: | --- | ---: |
| 5 s | `[2, 220500]` | 454 ms | `[1, 1, 120000]` | 1,540 ms |
| 30 s | `[2, 1323000]` | 332 ms | `[1, 1, 720000]` | 20 ms |
| 120 s | `[2, 5292000]` | 362 ms | `[1, 1, 2880000]` | 35 ms |

The first resample call includes a visible cold-start/setup cost. The 30 s and 120 s resample timings are more representative once the transform path is warm. The key acceptance point is that all three source-frame crops load directly from OGG and produce exact 24 kHz crop lengths without a preprocessing file.

## Duration Metadata

`torchaudio.info` can read sample rate, frame count, channel count, encoding, and duration-equivalent metadata without calling `torchaudio.load`. For the selected recording, the `torchaudio.info` duration matches the existing manifest duration within floating-point precision.

However, metadata reads are not free on these OGGs: the 16-file sample had a 2.4 s median `torchaudio.info` time and a 6.7 s max. The loader should prefer manifest `duration` fields when selecting crops and should use `torchaudio.info` for validation, cache fill, or missing metadata, not for every hot-path sample if it can be avoided.

## Resampling Policy

For `SpeechCodec-base`, use this first policy:

1. Resolve the source OGG path by replacing the manifest `.spec.pt` suffix with `.ogg`.
2. Select crop offsets in source frames from manifest duration and `torchaudio.info` sample rate when needed.
3. Load bounded crops with `torchaudio.load(..., frame_offset=..., num_frames=...)`.
4. Collapse to mono by channel mean.
5. Resample to 24,000 Hz before model input.
6. Return contiguous float32 tensors shaped `[batch, 1, crop_seconds * 24000]`.

Per-crop amplitude normalization can be added in the codec skeleton, but this audit did not compare normalization choices.

## Recommendation

OGG/`torchaudio` loading is viable for the first repo-local codec skeleton. The next issue should be the minimal `SpeechCodec-base` skeleton and its tiny one-recording smoke, not a revised data-path plan.

The skeleton should not depend on the existing `.spec.pt` spectrogram loader. It should add a waveform data path that maps existing Spotify manifest records to sibling OGGs and applies the 24 kHz mono crop policy above. Keep the spectrogram path unchanged for current ASR evaluation/training code.

Residual risk:

- The sample-rate/channel distribution is bounded to 16 real OGGs, not the full corpus.
- Only crop offset `0` was timed. Random crop offsets should be included in the skeleton smoke because OGG seeking behavior may differ away from the start of file.
- The audit measured CPU decode and tensor-shape readiness, not codec model forward/backward memory or training speed.
