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
left context, not the intended 2048-frame training context.

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
- A follow-up fix added `max_kv_cache_length` support and a regression test
  that the cache is trimmed during cached attention.
- Training logs for the completed full-Spotify run show teacher-forced predicted non-silence fractions near target, for example final progress around `tgt_ns=0.277`, `pred_ns=0.262`, `loss=0.3318`.
- The completed full-recording TEDLIUM CSV is not a WER arithmetic bug; it is genuinely near-all-deletion under that decode setup.

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

## Conclusion

The `0.9924` full-TEDLIUM WER should not be treated as the model's utterance-scale recognition quality. The same checkpoint produces sensible utterance-level TEDLIUM hypotheses. The original KV-cache probe showed that uncapped KV-cache decoding matched no-cache decoding on the bounded utterance probe, but it did not prove that the full-recording eval used the intended 2048-frame context.

The most likely issue is that the generic TEDLIUM eval decoded each full TED talk as one long streaming sequence, while this streaming decoder was trained and debugged on chunk/window-scale inputs with delayed frame targets. The full-recording path also used unbounded cached attention before the follow-up cap fix. A corrected TEDLIUM evaluation should segment by STM utterances or a comparable chunked streaming window, use `max_kv_cache_length: 2048` when KV caching is enabled, and then aggregate WER.
