# RL Post-Training Notes

ROB-26 adds `exp/train_files/train_enc_dec_rl.py` for on-policy encoder-decoder post-training.

## Algorithm

Each optimizer step is one rollout batch:

1. Load a floras-50 audio/text batch and split it into spectrogram chunks.
2. Sample `rl.num_rollouts` transcripts per chunk from the current policy.
3. Reward each rollout with `1.0` only when normalized WER is exactly `0.0`; otherwise reward is `0.0`.
4. Compute advantages from that same rollout group.
5. Recompute sequence log-probabilities with gradients and take exactly one optimizer update.

No replay buffer, old-policy cache, or repeated update over the same rollout batch is used.
Rollouts and gradient log-prob recomputation both run with the model in eval mode so decoder/encoder dropout does not make the recomputed policy differ from the sampled rollout policy. Gradients are still enabled for the log-prob pass.

When variable-length recordings are batched together, RL chunking drops a recording from later chunks once its cumulative chunk offset reaches its true audio length. This avoids producing zero-length audio chunks for shorter recordings in the batch.

`rl.algorithm: max_rl` uses the MaxRL-style binary-verifier advantage:

```text
A_i = (r_i - mean(r_group)) / max(mean(r_group), eps)
```

Groups with zero successes get zero advantage. This follows the MaxRL implementation note that the key change from group policy-gradient baselines is dividing the centered reward by the mean reward for binary success feedback.

`rl.algorithm: grpo` uses group-relative normalization:

```text
A_i = (r_i - mean(r_group)) / max(std(r_group), eps)
```

Groups with zero reward variance get zero advantage.

## ROB-26 Configs

Training config:

```bash
exp/configs/enc_dec/rl_floras50_3k.yaml
```

It starts from:

```text
/mnt/parscratch/users/acp21rjf/spotify/checkpoints/enc_dec/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt
```

This is the `enc_dec_3l_no_anorm_v2` checkpoint selected for ROB-26 after the benchmarking clarification in Linear.

It reads floras-50 training pairs from the committed manifest:

```text
lcasr/artifacts/floras50_long_only.json
```

It writes checkpoints and WandB files under:

```text
/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/
```

CPU validation launcher:

```bash
sbatch exp/configs/enc_dec/rl_floras50_3k_cpu_debug.sh
```

CPU real-model rollout smoke launcher:

```bash
sbatch exp/configs/enc_dec/rl_floras50_3k_cpu_rollout_smoke.sh
```

GPU training launcher:

```bash
sbatch exp/configs/enc_dec/rl_floras50_3k_gpu.sh
```

Follow-up constant-LR sweep after the 3K run:

```bash
for lr in 3e-6 6e-6 1e-5 2e-5 4e-5 8e-5; do
  cfg="exp/configs/enc_dec/rl_floras50_30k_b6_r24_const_lr_${lr}.yaml"
  sbatch --job-name="ROB26-lr${lr}" \
    --export=ALL,CONFIG="${cfg}" \
    exp/configs/enc_dec/rl_floras50_30k_b6_r24_const_lr_gpu.sh
done
```

These jobs use batch size 6, 24 rollouts, 30K max steps, constant LR, and save every 2K steps. Each LR writes to its own checkpoint directory under:

```text
/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/checkpoints/rl_floras50_30k_b6_r24_const_lr_<lr>
```

The launcher pins `PYTHONPATH` to `/mnt/parscratch/users/acp21rjf/symphony-workspaces/ROB-26` so the job uses this checkout's `lcasr` package.

Tedlium eval config for the final 3K-step checkpoint:

```bash
eval/eval_configs/enc_dec_rl_tedlium.yaml
```

Before running the eval manager, create the output directory:

```bash
mkdir -p /mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-26/eval
cd eval
sbatch --export=CONFIG='./eval_configs/enc_dec_rl_tedlium.yaml' ./run_eval_h100.sh
```

The issue-specific launcher does the directory setup and uses the same config by default:

```bash
sbatch eval/run_eval_rob26_tedlium_h100.sh
```
