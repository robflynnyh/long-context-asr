# ROB-123 TEDLIUM Eval Investigation

## Context

The callback-backed Stanage CPU eval of the ROB-123 final checkpoint reported aggregate TEDLIUM test WER `0.9924153818890661` over `28215` words:

```text
/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval/rob123-tedlium-cpu-kvcache-full-20260526T0830Z/rob123_tedlium_eval.csv
```

That run decoded each TEDLIUM talk as one full recording through `eval/run.py` with:

```yaml
transcribe_kwargs:
  use_kv_cache: true
```

At the time of that run, `StreamingDecoderASR` did not cap the KV cache length:
the cached attention path concatenated all prior keys/values and returned the
full accumulated cache. That means the full-recording result used unbounded
left context, not the intended 2048-spectrogram-frame training context. For
the ROB-123 8x causal subsampling setup, 2048 input spectrogram frames map to
257 cached decoder/KV frames.

The aggregate error mix was deletion dominated:

```text
wer=0.9924153818890661
words=28215
ins_rate=3.54421407052986e-05
del_rate=0.9919546340598973
sub_rate=0.0004253056884635832
```

## Checks

- Current `StreamingDecoderASR` greedy scoring uses the intended two-head rule: compare `log P(silence)` against `log P(not silence) + log P(token)`.
- The KV-cache decode path is covered by `tests/test_streaming_decoder_asr_rope.py::test_greedy_decode_kv_cache_matches_uncached_decode`.
- A follow-up fix added `max_kv_cache_spectrogram_length` support, deriving
  the actual KV-cache frame cap from the model subsampling path, plus a
  regression test that the cache is trimmed during cached attention.
- Training logs for the completed full-Spotify run show teacher-forced predicted non-silence fractions near target, for example final progress around `tgt_ns=0.277`, `pred_ns=0.262`, `loss=0.3318`.
- The uncapped full-recording TEDLIUM CSV is not a WER arithmetic bug; it is genuinely near-all-deletion under that decode setup.

## Bounded Probe

To separate checkpoint quality from the full-recording eval setup, a bounded utterance-level TEDLIUM probe was run against the same final checkpoint:

```text
checkpoint=/mnt/parscratch/users/acp21rjf/spotify/streaming_decoder_asr_100m_rope_rob123_full_spotify_2epoch_delay0p5_rob123-rope-full-spotify-2epoch-delay0p5-20260523T091306Z/step_272362.pt
recording=AimeeMullins_2009P
utterances=8
```

No-cache probe:

```text
job=10267377
state=COMPLETED 0:0
artifact=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval/probes/rob123-ted-utt-probe-20260526T1120Z/utt8
wer=0.1223021582733813
words=139
ins_rate=0.014388489208633094
del_rate=0.05755395683453238
sub_rate=0.050359712230215826
mean_pred_non_silence_fraction=0.20660471643444536
```

KV-cache probe:

```text
job=10267384
state=COMPLETED 0:0
artifact=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval/probes/rob123-ted-utt-kvcache-probe-20260526T1135Z/utt8
wer=0.1223021582733813
words=139
ins_rate=0.014388489208633094
del_rate=0.05755395683453238
sub_rate=0.050359712230215826
mean_pred_non_silence_fraction=0.20660471643444536
```

Representative decoded lines from the KV-cache probe:

```text
ref:  i would already finished editing the piece and i realized that i had never once in my life looked up the word disabled to see what i would find let me read you the entry
hyp:  i would already finished editing a piece and i realized that i had never once in my life looked up the word disabled to see what i would find let me read you the entry

ref:  see also hurt useless and weak antonyms healthy strong capable
hyp:  see also hurt useless and weak antonyms healthy strong
```

## Corrected Full TEDLIUM Eval

After the human clarification that `2048` refers to input spectrogram frames
rather than already-subsampled KV-cache frames, the eval wrapper was rerun with:

```yaml
transcribe_kwargs:
  use_kv_cache: true
  max_kv_cache_spectrogram_length: 2048
```

For the ROB-123 model config, this maps to an effective decoder/KV cache cap of
257 frames. The callback-backed Stanage CPU eval completed successfully:

```text
job=10267409
state=COMPLETED 0:0
elapsed=00:24:47
batch_max_rss=10219476K
artifact=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval/rob123-tedlium-cpu-kvcache-spec2048-full-20260526T114444Z
result_csv=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval/rob123-tedlium-cpu-kvcache-spec2048-full-20260526T114444Z/rob123_tedlium_eval.csv
```

Aggregate TEDLIUM test metrics from the corrected CSV:

```text
wer=0.13974836080099237
words=28215
ins_rate=0.019812156654261916
del_rate=0.04788233209285841
sub_rate=0.07205387205387205
checkpoint=/mnt/parscratch/users/acp21rjf/spotify/streaming_decoder_asr_100m_rope_rob123_full_spotify_2epoch_delay0p5_rob123-rope-full-spotify-2epoch-delay0p5-20260523T091306Z/step_272362.pt
```

The bounded log scan showed no traceback, OOM, or setup failure. This corrected
result supersedes the uncapped `0.9924` WER result for ROB-123 handoff purposes.

## Sampling Temperature Follow-up

The sampled decode path now uses the same two-head probability semantics as
greedy decode. It samples from the combined class distribution:

```text
P(silence)
P(token) = P(not_silence) * P(token | not_silence)
```

Temperature is applied after that combination. This avoids the older behavior
where sampling only drew a silence/not-silence decision and then selected text
greedily from the text head.

Validation and launch evidence:

```text
commit=402c97f44a7779684e2493d3a0ed2cdfd9b8f09c
callback_only_job=10267465 COMPLETED 0:0
smoke_job=10267467 COMPLETED 0:0 elapsed=00:04:47 batch_max_rss=10970248K
smoke_decode_mode=sample
smoke_temperature=0.3
smoke_use_kv_cache=true
smoke_max_kv_cache_spectrogram_length=2048
smoke_break_eval=1
smoke_wer=0.11219348337252268
smoke_words=2977
smoke_result_csv=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval/rob123-tedlium-sample-temp0p3-smoke-20260526T1255Z/rob123_tedlium_eval.csv
full_sample_job=10267475 COMPLETED 0:0 elapsed=00:25:04 batch_max_rss=10181720K
full_sample_wer=0.1522948786106681
full_sample_words=28215
full_sample_ins_rate=0.02130072656388446
full_sample_del_rate=0.056636540847067166
full_sample_sub_rate=0.07435761119971647
full_sample_artifact=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval/rob123-tedlium-sample-temp0p3-full-20260526T1300Z
full_sample_result_csv=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval/rob123-tedlium-sample-temp0p3-full-20260526T1300Z/rob123_tedlium_eval.csv
```

## Silence-Head Sampling Follow-up

A later follow-up tested the older silence-gate sampling shape explicitly:
sample only the binary silence/non-silence head at `temperature=0.3`, then use
greedy argmax from the text head whenever non-silence is sampled. This mode is
recorded as `decode_mode=sample_silence_greedy_text`.

Validation and completion evidence:

```text
commit=a288118f48b4436c03603ee64e90959e9561ccd5
callback_only_job=10267548 COMPLETED 0:0
smoke_job=10267549 COMPLETED 0:0 elapsed=00:03:13
smoke_decode_mode=sample_silence_greedy_text
smoke_temperature=0.3
smoke_use_kv_cache=true
smoke_max_kv_cache_spectrogram_length=2048
smoke_break_eval=1
smoke_wer=0.12025529056096741
smoke_words=2977
full_silence_sample_job=10267846 COMPLETED 0:0 elapsed=00:26:00 batch_max_rss=10106356K
full_silence_sample_wer=0.1676058833953571
full_silence_sample_words=28215
full_silence_sample_ins_rate=0.027786638312954103
full_silence_sample_del_rate=0.06308701045543151
full_silence_sample_sub_rate=0.07673223462697147
full_silence_sample_artifact=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval/rob123-tedlium-silence-sample-temp0p3-full-20260526T1345Z
full_silence_sample_result_csv=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval/rob123-tedlium-silence-sample-temp0p3-full-20260526T1345Z/rob123_tedlium_eval.csv
```

## Post-SDPA Cached-Attention Validation

After the model-local cached-attention path was simplified to use PyTorch SDPA
only at PR head `da623215b246d6344276b335531f4e97dca5e3d7`, Stanage was
offline for maintenance and the real-checkpoint decode smoke could not be run.
Once Stanage returned, the same final ROB-123 checkpoint was rerun through the
callback-capable TEDLIUM CPU wrapper with Linear callback disabled for the
bounded validation smoke:

```text
job=10270996
state=COMPLETED 0:0
elapsed=00:08:16
batch_max_rss=11167972K
commit=da623215b246d6344276b335531f4e97dca5e3d7
decode_mode=greedy
temperature=1.0
use_kv_cache=true
max_kv_cache_spectrogram_length=2048
break_eval=1
artifact=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval/rob123-tedlium-kvcache-spec2048-greedy-smoke-after-sdpa-20260528T1327Z
result_csv=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval/rob123-tedlium-kvcache-spec2048-greedy-smoke-after-sdpa-20260528T1327Z/rob123_tedlium_eval.csv
```

The smoke produced aggregate WER `0.1128653006382264` over `2977` words
(`ins_rate=0.017803157541148806`, `del_rate=0.026536781995297277`,
`sub_rate=0.06852536110178031`). The wrapper summary confirmed the checkpoint
and TEDLIUM root existed, generated the expected eval config, and exited with
status `0`.

To answer the later WER-invariance question directly, the full greedy TEDLIUM
eval was rerun on the final SDPA-only PR branch with the same corrected
2048-spectrogram-frame KV-cache cap:

```text
job=10271126
state=COMPLETED 0:0
elapsed=00:25:09
batch_max_rss=10169812K
commit=cd886667dbefeb1207216562d8445c0975c6e1f8
decode_mode=greedy
temperature=1.0
use_kv_cache=true
max_kv_cache_spectrogram_length=2048
break_eval=0
rows=12
wer=0.13974836080099237
words=28215
ins_rate=0.019812156654261916
del_rate=0.04788233209285841
sub_rate=0.07205387205387205
artifact=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval/rob123-tedlium-kvcache-spec2048-greedy-full-after-sdpa-20260528T1348Z
result_csv=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval/rob123-tedlium-kvcache-spec2048-greedy-full-after-sdpa-20260528T1348Z/rob123_tedlium_eval.csv
```

This matches the earlier corrected full greedy TEDLIUM result exactly at the
aggregate WER and error-rate fields recorded in this report.

## Earnings-22 Test Eval

A later Linear follow-up requested the smaller Earnings-22 `test` partition,
not `earnings22_full`. The same final ROB-123 checkpoint and corrected
2048-spectrogram-frame KV-cache cap were evaluated with greedy decode through
the callback-capable Stanage CPU wrapper:

```text
job=10272085
state=COMPLETED 0:0
elapsed=00:49:38
batch_max_rss=20982716K
commit=68bbb9c0c99037c70a78bc2426f0de5a06a5b2cb
dataset=earnings22
split=test
decode_mode=greedy
temperature=1.0
use_kv_cache=true
max_kv_cache_spectrogram_length=2048
break_eval=0
rows=7
wer=0.5014194391683516
words=48963
ins_rate=0.02234340216081531
del_rate=0.35061168637542633
sub_rate=0.12846435063210995
artifact=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval/rob123-earnings22-test-kvcache-greedy-full-20260528T150106Z
result_csv=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-123/eval/rob123-earnings22-test-kvcache-greedy-full-20260528T150106Z/rob123_earnings22_eval.csv
```

The callback summary confirmed the checkpoint, `earnings22` test/dev audio
paths, transcript path, eval config, and result CSV path. A direct CSV read
found one aggregate `earnings22`/`test`/`all` row with the metrics above, and
bounded log inspection found no tracebacks or failure markers.

## Conclusion

The `0.9924` full-TEDLIUM WER should not be treated as the model's recognition quality because it used an uncapped accumulated KV cache rather than the intended 2048-spectrogram-frame training context. The same checkpoint produces sensible utterance-level TEDLIUM hypotheses, and the corrected full TEDLIUM eval with `max_kv_cache_spectrogram_length: 2048` gives aggregate WER `0.13974836080099237`. The requested joint sampled eval with temperature `0.3` also completed successfully and produced aggregate WER `0.1522948786106681`. The silence-head-only sampled variant completed afterward and produced aggregate WER `0.1676058833953571`. The later Earnings-22 `test` eval completed successfully with aggregate WER `0.5014194391683516`.

For future evals of this decoder family, cached streaming decode should pass the spectrogram-frame context cap explicitly (`max_kv_cache_spectrogram_length: 2048` for this run). The smaller utterance-level probe remains useful as a bounded wiring check. For ROB-123 handoff, the current aggregate TEDLIUM results are greedy WER `0.13974836080099237`, joint sampled temperature-`0.3` WER `0.1522948786106681`, and silence-head sampled temperature-`0.3` WER `0.1676058833953571`; the current smaller Earnings-22 test result is greedy WER `0.5014194391683516`. The latest SDPA-only cached-attention branch passed both a real final-checkpoint TEDLIUM greedy KV-cache smoke and a like-for-like full greedy TEDLIUM rerun after Stanage maintenance; the full rerun reproduced the earlier corrected WER exactly.
