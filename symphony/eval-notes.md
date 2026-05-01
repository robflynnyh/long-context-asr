# Evaluation Notes

These notes summarize the evaluation flow from `eval/README.md`, `eval/eval_manager.py`, `eval/run.py`, dataset-specific `run.py` files, and existing Slurm scripts.

## Evaluation Modes

`eval/run.py` supports three modes:

- `averaged_moving_window`: default and used in the Interspeech 2024 paper.
- `windowed_attention`: sets the model attention window from `seq_len / subsampling_factor / 2`, then evaluates very long sequences.
- `buffered`: uses `lcasr.eval.buffered_transcription.fetch_logits`.

The generic manager passes `evaluation_mode` through YAML under `args`.

## Manager Flow

Run grids with:

```bash
cd eval
python eval_manager.py -config eval_configs_for_journal/eval_config_rb_windowed.yaml
```

Or on Slurm:

```bash
cd eval
sbatch --export=CONFIG='./eval_configs_for_journal/eval_config_rb_windowed.yaml' ./run_eval_h100.sh
sbatch --export=CONFIG='./eval_configs_for_journal/eval_config_rb_windowed.yaml' ./run_eval_a100.sh
```

`eval/eval_manager.py` loads:

- `models`: checkpoint path, `seq_len`, `overlap_ratio`, repeat/name/metadata.
- `datasets`: dataset names and splits.
- `args`: shared runtime args such as `model_class`, `evaluation_mode`, `save_dataframe_path`, `verbose`, and optional module selector.

It skips model/dataset/split combinations already present in `save_dataframe_path`, then appends rows with WER, word count, error rates, checkpoint path, sequence length, overlap ratio, and model metadata.

## Dataset Modules

The generic `eval/run.py` imports these dataset loaders:

- `earnings22`
- `earnings22_full`
- `tedlium`
- `rev16`
- `this_american_life`
- `spotify`
- `floras50`

Each dataset has its own accepted splits and path assumptions. Inspect `eval/<dataset>/run.py` before running.

Observed path handling:

- `earnings22`, `earnings22_full`, `this_american_life`, and `floras50` read `eval/paths.yaml` if present.
- `tedlium`, `rev16`, and `spotify` use hard-coded defaults in their modules or helper functions.
- `eval/paths.yaml` is gitignored. `eval/paths_template.yaml` currently gives Earnings-22/Earnings-22-full examples and Tedlium keys, but Tedlium's loader does not read it. Add `this_american_life` or `floras50` keys yourself if you need those overrides.

## Single-Run Evaluation

For a direct check without the grid manager:

```bash
cd eval
python run.py \
  --dataset earnings22 \
  -c /path/to/checkpoint.pt \
  -split test \
  -seq 16384 \
  -overlap 14336 \
  -model_class SCConformerXL \
  -eval_mode averaged_moving_window \
  -break
```

`-break` stops after the first recording and is useful for debugging script/config/path errors. Even debug evaluations can load large models and audio, so use a Slurm job if it may take more than a short metadata check.

## Outputs And Metrics

- Manager outputs are CSVs, usually under `eval/results/`.
- Dataset-specific scripts may append text logs when `-log <path>` is supplied.
- Treat WERs as valid only when extracted from output artifacts or the script's returned data. Do not infer metrics from partial logs.
- `include_per_recording_evaluations: true` in manager args makes `eval/run.py` include per-recording rows before the aggregate `recording: all` row.

## Common Caveats

- `save_dataframe_path` parent directory must exist; `eval_manager.py` asserts this before running.
- `model.path` must exist for every model in the YAML.
- `overlap` defaults to `int(seq_len * overlap_ratio)` in the manager.
- Some configs under `eval/eval_configs/` are older and may not match the current generic manager shape; `eval/eval_configs_for_journal/` is usually the cleaner starting point.
