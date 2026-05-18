# ROB-90 Streaming Decoder TEDLIUM Utterance Eval

This is a bounded wiring sanity check for the ROB-76 decoder-only streaming ASR checkpoint. Poor recognition quality is expected; the goal is to prove TEDLIUM utterance decoding runs through the generic eval path and to inspect representative outputs.

## Command

```bash
PYTHONPATH=. python eval/run.py --dataset tedlium --checkpoint /store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_two_head_mimas_full_epoch_rob76-mimas-3epoch-b48-two-head-20260517T140145Z/step_82737.pt --tedlium-root /store/store4/data/TEDLIUM_release1/legacy --split test --utterance-level --max-recordings 1 --max-utterances 8 --decode-mode greedy --eval-dtype bfloat16 --output-jsonl /store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/generic-eval-run-8utt/predictions.jsonl --summary-json /store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/generic-eval-run-8utt/summary.json --report-md symphony/reports/ROB-90-streaming-decoder-tedlium-eval.md --report-samples 8 --no-progress
```

## Configuration

- checkpoint: `/store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_two_head_mimas_full_epoch_rob76-mimas-3epoch-b48-two-head-20260517T140145Z/step_82737.pt`
- dataset/split: `tedlium` / `test`
- TEDLIUM root: `/store/store4/data/TEDLIUM_release1/legacy`
- utterance level: `True`
- decode mode: `greedy`
- utterances: `8`
- output JSONL: `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/generic-eval-run-8utt/predictions.jsonl`
- summary JSON: `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/generic-eval-run-8utt/summary.json`
- device/dtype: `cuda` / `bfloat16`

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
