# Encoder-Decoder Configs

This folder contains the encoder-decoder configs that are useful as reference points for trained checkpoints. Older exploratory configs are kept under `bin/experimental/`.

## Baseline 2048-Context Encoder-Decoder Checkpoints

These runs are normal `EncDecSconformerV2` encoder-decoder baselines from `masking.yaml`, with `audio_chunking.size: 2048` and `training.mask_pass: false`.

| Run | Config | Checkpoint |
| --- | --- | --- |
| `baseline_rp_1` | `masking.yaml` | `/mnt/parscratch/users/acp21rjf/spotify/checkpoints/enc_dec/baseline_rp_1/step_3937542.pt` |
| `baseline_rp_2` | `masking.yaml` | `/mnt/parscratch/users/acp21rjf/spotify/checkpoints/enc_dec/baseline_rp_2/step_3937542.pt` |
| `baseline_rp_3` | `masking.yaml` | `/mnt/parscratch/users/acp21rjf/spotify/checkpoints/enc_dec/baseline_rp_3/step_3937542.pt` |

The paired `masking_rp_*` runs in `masking.yaml` use `training.mask_pass: true` and are not the normal baselines.

## Other 2048-Context Encoder-Decoder Reference

`0_873621.yaml` is the generated config corresponding to:

```text
/mnt/parscratch/users/acp21rjf/spotify/checkpoints/enc_dec/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt
```

`enc_dec_test.yaml` is the source/template config for that run family.

## Archived Experimental Configs

`bin/experimental/` contains older one-off experiments such as constant-sequence history variants, RL/shared-KV experiments, and generated-context configs. Treat these as historical references unless a task explicitly asks for them.
