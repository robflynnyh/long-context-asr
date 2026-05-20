# ROB-90 Streaming Decoder TEDLIUM Utterance Eval

This is a bounded wiring sanity check for the ROB-76 decoder-only streaming ASR checkpoint. Poor recognition quality is expected; the goal is to prove TEDLIUM utterance decoding runs through `StreamingDecoderASR.transcribe(...)` and to inspect representative outputs.

Update on 2026-05-20: after the original ROB-90 handoff, a follow-up Linear comment asked for the same TEDLIUM sanity eval with the continued ROB-92 checkpoint. The ROB-92 result is recorded below using the same script, TEDLIUM recording, utterance cap, greedy decode path, and dtype.

## Command

```bash
/store/store5/software/simple-gpu-schedule/with-gpu any --num 1 --idle-seconds 0 -- bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-long-context-asr/ROB-90 && PYTHONPATH=. python symphony/scripts/rob90_streaming_tedlium_eval.py --checkpoint /store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_two_head_mimas_full_epoch_rob76-mimas-3epoch-b48-two-head-20260517T140145Z/step_82737.pt --tedlium-root /store/store4/data/TEDLIUM_release1/legacy --split test --max-utterances 8 --decode-mode greedy --eval-dtype bfloat16 --output-dir /store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/model-transcribe-8utt'
```

## Configuration

- checkpoint: `/store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_two_head_mimas_full_epoch_rob76-mimas-3epoch-b48-two-head-20260517T140145Z/step_82737.pt`
- dataset/split: `tedlium` / `test`
- TEDLIUM root: `/store/store4/data/TEDLIUM_release1/legacy`
- utterance level: `True`
- decode mode: `greedy`
- utterances: `8`
- output JSONL: `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/model-transcribe-8utt/predictions.jsonl`
- output summary: `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/model-transcribe-8utt/summary.json`
- device/dtype: Mimas `cuda` via `with-gpu` GPU 2 / `bfloat16`
- shared eval files changed: none; `eval/run.py` and `eval/tedlium/run.py` are left unchanged from `dev`

## Summary

- WER: `0.661871`
- words: `139`
- insertions/deletions/substitutions: `0.000000` / `0.517986` / `0.143885`
- mean predicted non-silence fraction: `0.069744`

## Sample Outputs

| utterance | reference | prediction | pred non-silence |
| --- | --- | --- | --- |
| `AimeeMullins_2009P:0` | i would like to share with you a discovery that i made a few months ago while writing an article for italian wired i always keep my thesaurus handy whenever i am writing anything but | i would like to share with you with a separate if you meant to go i always keep my | 0.144 |
| `AimeeMullins_2009P:1` | i would already finished editing the piece and i realized that i had never once in my life looked up the word disabled to see what i would find let me read you the entry | i already finished editing a piece and i realized that i had never once in my life looked up the word the table the civilizing | 0.201 |
| `AimeeMullins_2009P:2` | disabled adjective crippled helpless useless wrecked | <empty> | 0.000 |
| `AimeeMullins_2009P:3` | stalled maimed wounded mangled lame mutilated | <empty> | 0.000 |
| `AimeeMullins_2009P:4` | rundown worn out weakened impotent castrated paralyzed handicapped | run down weekend | 0.025 |
| `AimeeMullins_2009P:5` | senile decrepit laid up done up done for done in cracked up counted out | female the cutest later | 0.042 |
| `AimeeMullins_2009P:6` | see also hurt useless and weak antonyms healthy strong capable | <empty> | 0.000 |
| `AimeeMullins_2009P:7` | i was reading this list out loud to a friend and at 1st was laughing it was so ludicrous but i just gotten past mangled | i was reading this without loud to a friend and it 1st laughing so but i | 0.145 |

## ROB-92 Follow-Up Checkpoint

### Command

```bash
/store/store5/software/simple-gpu-schedule/with-gpu any --num 1 --idle-seconds 0 -- bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-long-context-asr/ROB-90 && PYTHONPATH=. python symphony/scripts/rob90_streaming_tedlium_eval.py --checkpoint /store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_two_head_rob92_mimas_5epoch_rob92-mimas-5epoch-continuation-20260518T221140Z/step_137895.pt --tedlium-root /store/store4/data/TEDLIUM_release1/legacy --split test --max-utterances 8 --decode-mode greedy --eval-dtype bfloat16 --output-dir /store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/rob92-model-transcribe-8utt'
```

### Configuration

- checkpoint: `/store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_two_head_rob92_mimas_5epoch_rob92-mimas-5epoch-continuation-20260518T221140Z/step_137895.pt`
- source issue: ROB-92 continued the ROB-76 two-head checkpoint for another 5 epochs on Mimas
- dataset/split: `tedlium` / `test`
- TEDLIUM root: `/store/store4/data/TEDLIUM_release1/legacy`
- utterance level: `True`
- decode mode: `greedy`
- utterances: `8`
- output JSONL: `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/rob92-model-transcribe-8utt/predictions.jsonl`
- output summary: `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/rob92-model-transcribe-8utt/summary.json`
- device/dtype: Mimas `cuda` via `with-gpu` GPU 0 / `bfloat16`
- shared eval files changed: none; `eval/run.py` and `eval/tedlium/run.py` are left unchanged from `dev`

### Smoke Test

```bash
/store/store5/software/simple-gpu-schedule/with-gpu any --num 1 --idle-seconds 0 -- bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-long-context-asr/ROB-90 && PYTHONPATH=. python symphony/scripts/rob90_streaming_tedlium_eval.py --checkpoint /store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_two_head_rob92_mimas_5epoch_rob92-mimas-5epoch-continuation-20260518T221140Z/step_137895.pt --tedlium-root /store/store4/data/TEDLIUM_release1/legacy --split test --max-utterances 1 --decode-mode greedy --eval-dtype bfloat16 --max-output-frames 2 --output-dir /store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/rob92-model-transcribe-smoke-1utt'
```

The one-utterance smoke loaded the ROB-92 checkpoint, TEDLIUM audio, and `StreamingDecoderASR.transcribe(...)` path successfully. It intentionally capped generation to 2 output frames and wrote `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-90/rob92-model-transcribe-smoke-1utt/`.

### Summary

- WER: `0.438849`
- words: `139`
- insertions/deletions/substitutions: `0.021583` / `0.244604` / `0.172662`
- mean predicted non-silence fraction: `0.160292`

### Sample Outputs

| utterance | reference | ROB-92 prediction | pred non-silence |
| --- | --- | --- | --- |
| `AimeeMullins_2009P:0` | i would like to share with you a discovery that i made a few months ago while writing an article for italian wired i always keep my thesaurus handy whenever i am writing anything but | like to share with you at the discovery that i made a few months ago while writing an article for the italian wire i always keep my piss | 0.252 |
| `AimeeMullins_2009P:1` | i would already finished editing the piece and i realized that i had never once in my life looked up the word disabled to see what i would find let me read you the entry | i would already finished editing the piece and i realized that i had never once in my life looked up the word to savolt the sequitite fine | 0.243 |
| `AimeeMullins_2009P:2` | disabled adjective crippled helpless useless wrecked | disabled adjectives crippled helpless | 0.125 |
| `AimeeMullins_2009P:3` | stalled maimed wounded mangled lame mutilated | stalled main mangled | 0.078 |
| `AimeeMullins_2009P:4` | rundown worn out weakened impotent castrated paralyzed handicapped | run down one out we can impotent castrated | 0.109 |
| `AimeeMullins_2009P:5` | senile decrepit laid up done up done for done in cracked up counted out | cena the crepet later donut done for done in | 0.126 |
| `AimeeMullins_2009P:6` | see also hurt useless and weak antonyms healthy strong capable | see also hurt useless and week anthin is healthy | 0.113 |
| `AimeeMullins_2009P:7` | i was reading this list out loud to a friend and at 1st was laughing it was so ludicrous but i just gotten past mangled | i was reading this without loud to a friend and it 1st was laughing it was so ludicrous but i | 0.236 |
