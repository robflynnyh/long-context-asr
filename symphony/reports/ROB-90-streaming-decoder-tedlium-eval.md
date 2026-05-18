# ROB-90 Streaming Decoder TEDLIUM Utterance Eval

This is a bounded wiring sanity check for the ROB-76 decoder-only streaming ASR checkpoint. Poor recognition quality is expected; the goal is to prove TEDLIUM utterance decoding runs through `StreamingDecoderASR.transcribe(...)` and to inspect representative outputs.

## Command

```bash
/store/store5/software/simple-gpu-schedule/with-gpu any --num 1 --idle-seconds 0 -- bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-long-context-asr/ROB-90 && PYTHONPATH=. python eval/tedlium/run.py -c /store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_two_head_mimas_full_epoch_rob76-mimas-3epoch-b48-two-head-20260517T140145Z/step_82737.pt -split test -single_utt -ted_root /store/store4/data/TEDLIUM_release1/legacy -max_utts 8 -decode_mode greedy -eval_dtype bfloat16 -output_jsonl /store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/tedlium-run-rework-8utt/predictions.jsonl -nv'
```

## Configuration

- checkpoint: `/store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_two_head_mimas_full_epoch_rob76-mimas-3epoch-b48-two-head-20260517T140145Z/step_82737.pt`
- dataset/split: `tedlium` / `test`
- TEDLIUM root: `/store/store4/data/TEDLIUM_release1/legacy`
- utterance level: `True`
- decode mode: `greedy`
- utterances: `8`
- output JSONL: `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/tedlium-run-rework-8utt/predictions.jsonl`
- device/dtype: Mimas `cuda` via `with-gpu` GPU 2 / `bfloat16`

## Summary

- WER: `0.661871`
- words: `139`
- insertions/deletions/substitutions: `0.000000` / `0.517986` / `0.143885`
- mean predicted non-silence fraction: `0.069744`

## Sample Outputs

| utterance | reference | prediction | pred non-silence |
| --- | --- | --- | --- |
| `AimeeMullins_2009P:1` | i would like to share with you a discovery that i made a few months ago while writing an article for italian wired i always keep my thesaurus handy whenever i am writing anything but | i would like to share with you with a separate if you meant to go i always keep my | 0.144 |
| `AimeeMullins_2009P:2` | i would already finished editing the piece and i realized that i had never once in my life looked up the word disabled to see what i would find let me read you the entry | i already finished editing a piece and i realized that i had never once in my life looked up the word the table the civilizing | 0.201 |
| `AimeeMullins_2009P:4` | disabled adjective crippled helpless useless wrecked | <empty> | 0.000 |
| `AimeeMullins_2009P:6` | stalled maimed wounded mangled lame mutilated | <empty> | 0.000 |
| `AimeeMullins_2009P:7` | rundown worn out weakened impotent castrated paralyzed handicapped | run down weekend | 0.025 |
| `AimeeMullins_2009P:8` | senile decrepit laid up done up done for done in cracked up counted out | female the cutest later | 0.042 |
| `AimeeMullins_2009P:10` | see also hurt useless and weak antonyms healthy strong capable | <empty> | 0.000 |
| `AimeeMullins_2009P:12` | i was reading this list out loud to a friend and at 1st was laughing it was so ludicrous but i just gotten past mangled | i was reading this without loud to a friend and it 1st laughing so but i | 0.145 |
