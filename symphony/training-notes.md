# Training Notes

These notes summarize the training flow discovered from `exp/README.md`, `exp/run_launcher.py`, `exp/train.py`, `exp/train_files/train_enc_dec.py`, and representative configs under `exp/configs/`.

## Entry Points

- Acoustic model training:

```bash
cd exp
python train.py -config <generated-or-handwritten-config.yaml>
```

- Encoder-decoder training:

```bash
cd exp/train_files
python train_enc_dec.py -config ../configs/enc_dec/<config.yaml>
```

`train.py` and `train_enc_dec.py` both support:

```text
-config/--config
-rm_sched/--remove_scheduler
-reset_step/--reset_step
-anomaly/--anomaly
-num_workers/--num_workers
-pin_memory/--pin_memory
-prefetch/--prefetch_factor
-debug_hooks/--debug_hooks
```

`-debug_hooks` expects WandB to be enabled.

## Config Shape

Normal training configs include:

- `model_class`: class selector used by `lcasr.utils.general.get_model_class`.
- `model`: model dimensions, normalization, attention, subsampling, rotary settings, and optional gradient checkpointing.
- `optimizer`, `scheduler`: loaded by `lcasr.utils.general.load_optimizer`.
- `audio_chunking`: input sequence length in spectrogram frames. With 16 kHz audio and 160-sample hop, 100 frames is about 1 second.
- `sequence_scheduler`: optional long-context schedule that grows sequence length and adjusts batch size.
- `wandb`: logging and resume settings. Some scripts update the config with the WandB run ID.
- `checkpointing`: checkpoint directory, optional `pretrained`, and save cadence.
- `data.path`: JSON mapping audio spectrogram paths to aligned transcript JSON.
- `spec_augment`: optional augmentation block.
- `training`: batch size, gradient accumulation, epochs, dtype, random seed, clipping, and task-specific switches.

`exp/configs/bin/README.md` has a field-by-field config example. `exp/configs/paper_templates/` contains paper experiment templates. `exp/configs/enc_dec/README.md` lists known encoder-decoder reference checkpoints.

## Template Launcher

`exp/run_launcher.py` expands a template with `template_info` into per-run YAML and Slurm scripts:

```bash
cd exp
python run_launcher.py \
  -template configs/paper_templates/exp_set_seq_rotary_base.yaml \
  -mode h100 \
  -l train.py
```

Modes are defined in the script:

- `a100`: `gpu`, one GPU, 82 GB, 8 CPUs, 80 hours.
- `h100`: `gpu-h100`, one GPU, 100 GB, 16 CPUs, 60 hours.
- `h100nvl`: `gpu-h100-nvl`, one H100, 130 GB, 8 CPUs, 90 hours.

The launcher writes generated configs and scripts into `exp/.tmp/` and submits with `sbatch`. `.tmp/` is gitignored. For Symphony work, put Slurm stdout/stderr and any large outputs under `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts`.

## Restarts

`exp/run_restarter.py` resubmits runs created by `run_launcher.py`:

```bash
cd exp
python run_restarter.py \
  -run_names <name-from-.tmp-without-extension> \
  -tmp_dir ./.tmp \
  -mode h100 \
  -l train.py
```

By default it randomizes `training.random_seed` to avoid repeatedly hitting the same bad batch. Use `-keep_seed` or `-seed <int>` only when reproducibility requires it.

## Checkpoint Loading

`exp/train.py` loads from `checkpointing.pretrained` when that config key exists and is not `null`; otherwise it resumes from `checkpointing.dir`. Use `-rm_sched` and `-reset_step` when intentionally discarding scheduler state or restarting step counts.

## Common Caveats

- Training is GPU work. Submit through Slurm; do not run meaningful training in the interactive agent process.
- Many configs contain absolute checkpoint/data paths. Verify they exist before launching a long job.
- `audio_chunking.overlap` is present in many configs, but `train.py` currently sets training chunk overlap to `0`.
- Batch-size comments in configs are useful reference points: `512 -> 704`, `1024 -> 352`, `2048 -> 176`, `4096 -> 88`, `8192 -> 44`, `16384 -> 22`, `65536 -> 5`, `131072 -> 2`, `360000 -> 1`.
