# ROB-90 Streaming Decoder TEDLIUM Utterance Eval

This is a bounded wiring sanity check for the ROB-76 decoder-only streaming ASR checkpoint. Poor recognition quality is expected; the goal is to prove TEDLIUM utterance decoding runs and to inspect representative outputs.

## Command

The run was launched on Mimas through the cooperative GPU scheduler:

```bash
/store/store5/software/simple-gpu-schedule/with-gpu 3 --idle-seconds 0 -- bash -lc 'PYTHONPATH=. python eval/tedlium/run_streaming_decoder_asr.py --checkpoint /store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_two_head_mimas_full_epoch_rob76-mimas-3epoch-b48-two-head-20260517T140145Z/step_82737.pt --tedlium-root /store/store4/data/TEDLIUM_release1/legacy --split test --max-recordings 1 --max-utterances 8 --decode-mode greedy --output-jsonl /store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/greedy-8utt/predictions.jsonl --summary-json /store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/greedy-8utt/summary.json --report-md symphony/reports/ROB-90-streaming-decoder-tedlium-eval.md --report-samples 8 --no-progress'
```

Inner eval command:

```bash
PYTHONPATH=. python eval/tedlium/run_streaming_decoder_asr.py --checkpoint /store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_two_head_mimas_full_epoch_rob76-mimas-3epoch-b48-two-head-20260517T140145Z/step_82737.pt --tedlium-root /store/store4/data/TEDLIUM_release1/legacy --split test --max-recordings 1 --max-utterances 8 --decode-mode greedy --output-jsonl /store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/greedy-8utt/predictions.jsonl --summary-json /store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/greedy-8utt/summary.json --report-md symphony/reports/ROB-90-streaming-decoder-tedlium-eval.md --report-samples 8 --no-progress
```

## Configuration

- checkpoint: `/store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_two_head_mimas_full_epoch_rob76-mimas-3epoch-b48-two-head-20260517T140145Z/step_82737.pt`
- TEDLIUM root: `/store/store4/data/TEDLIUM_release1/legacy`
- split: `test`
- decode mode: `greedy`
- utterances: `8`
- output JSONL: `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/greedy-8utt/predictions.jsonl`
- summary JSON: `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/greedy-8utt/summary.json`
- device/dtype: `cuda` / `bfloat16`

## Summary

- WER: `0.676259`
- words: `139`
- insertions/deletions/substitutions: `0.000000` / `0.266187` / `0.410072`
- mean predicted non-silence fraction: `0.152739`

## Sample Outputs

| utterance | reference | prediction | pred non-silence |
| --- | --- | --- | --- |
| `AimeeMullins_2009P:1` | i would like to share with you a discovery that i made a few months ago while writing an article for italian wired i always keep my thesaurus handy whenever i am writing anything but | i like to share with you and discover it in a nate as you meant to go along right in order before deciding where i was keeping with thers | 0.245 |
| `AimeeMullins_2009P:2` | i would already finished editing the piece and i realized that i had never once in my life looked up the word disabled to see what i would find let me read you the entry | i already finished editing the piece and i realized that eden never once in my life looked up the word the savall the sealed i find i | 0.228 |
| `AimeeMullins_2009P:4` | disabled adjective crippled helpless useless wrecked | disabled advertisement outlets | 0.114 |
| `AimeeMullins_2009P:6` | stalled maimed wounded mangled lame mutilated | <empty> | 0.000 |
| `AimeeMullins_2009P:7` | rundown worn out weakened impotent castrated paralyzed handicapped | renowned warrant weekend infant test creative | 0.092 |
| `AimeeMullins_2009P:8` | senile decrepit laid up done up done for done in cracked up counted out | female decompeted leo donut do not for donutin | 0.143 |
| `AimeeMullins_2009P:10` | see also hurt useless and weak antonyms healthy strong capable | philosophical hurt newspaper is enlightened anton is healthy | 0.183 |
| `AimeeMullins_2009P:12` | i was reading this list out loud to a friend and at 1st was laughing it was so ludicrous but i just gotten past mangled | i was reading this whole set loud to a friend and it starts with laughing at the so lucras but i | 0.218 |
