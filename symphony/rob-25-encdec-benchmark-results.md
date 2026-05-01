# ROB-25 Encoder-Decoder TEDLIUM Benchmark

Issue: ROB-25
Branch: `symphony/ROB-25-benchmark-encdec`
Benchmark run code commit: `1aec3fc`
Current rebased equivalent code commit: `564496d`

## Run Configuration

- Config: `eval/eval_configs/rob25_enc_dec_tedlium.yaml`
- Dataset: `tedlium`
- Split: `test`
- Model class: `EncDecSconformerV2`
- Sequence length: `2048`
- Overlap ratio: `0`
- Output CSV: `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-25/results/enc_dec_tedlium_test.csv`
- Slurm job: `10094215` (`rob25-full-int`), interactive partition, completed `0:0` in `01:34:24`
- Logs:
  - `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-25/rob25-full-int-10094215.out`
  - `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-25/rob25-full-int-10094215.err`

The CSV was parsed with Python's `csv.DictReader` and checked for 84 data rows, 7 aggregate rows, expected checkpoint names, `tedlium`/`test`, `overlap_ratio == 0`, and `28215` aggregate words for each model.

## Aggregate Results

| name | decoding | WER | ins_rate | del_rate | sub_rate | words |
|---|---|---:|---:|---:|---:|---:|
| `enc_dec_3l_no_anorm_v2` | `direct` | 0.09803296119085593 | 0.013361687045897573 | 0.037568669147616515 | 0.04710260499734184 | 28215 |
| `baseline_rp_1` | `direct` | 0.37859294701399965 | 0.2814814814814815 | 0.03604465709728868 | 0.061066808435229485 | 28215 |
| `baseline_rp_2` | `direct` | 0.42721956406166933 | 0.33088782562466773 | 0.03636363636363636 | 0.05996810207336523 | 28215 |
| `baseline_rp_3` | `direct` | 0.4018429913166755 | 0.3037037037037037 | 0.03473329789119263 | 0.0634059897217792 | 28215 |
| `masking_rp_1` | `masked_conditioning` | 0.43320928584086477 | 0.3085238348396243 | 0.03615098351940457 | 0.0885344674818359 | 28215 |
| `masking_rp_2` | `masked_conditioning` | 0.4555378344852029 | 0.3342902711323764 | 0.03639907850434166 | 0.08484848484848485 | 28215 |
| `masking_rp_3` | `masked_conditioning` | 0.423498139287613 | 0.29845826687931953 | 0.03714336345915293 | 0.08789650894914053 | 28215 |

Best aggregate WER in this benchmark is `enc_dec_3l_no_anorm_v2` at `0.09803296119085593`.

## Validation

- `python -m py_compile eval/run.py lcasr/models/enc_dec_sconformer_v2.py`
- Slurm env check job `10094097`: completed `0:0`; confirmed torch `2.5.1+cu118`, 7 models, TEDLIUM test, all `overlap_ratio` values `0.0`, and no missing checkpoint paths.
- Slurm smoke job `10094080`: completed `0:0`; wrote `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-25/smoke_enc_dec_tedlium.csv`.
- Slurm full benchmark job `10094215`: completed `0:0`; wrote `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-25/results/enc_dec_tedlium_test.csv`.
- Failure scan across envcheck, smoke, and full benchmark logs found no tracebacks, exceptions, errors, failed steps, OOMs, kills, or out-of-memory messages.
