# ROB-81 TEDLIUM Evaluation

Issue: ROB-81

## Run

- Eval config: `eval/eval_configs/rob81_floras_finetune_tedlium.yaml`
- Slurm wrapper: `symphony/rob81_tedlium_eval_cpu.sbatch`
- Stanage job: `10253671` (`rob81-ted-eval`), interactive partition, completed `0:0` in `00:11:17`
- Output CSV: `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/eval/tedlium_floras_finetune.csv`
- Logs:
  - `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/eval/tedlium-eval-10253671.out`
  - `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/eval/tedlium-eval-10253671.err`
- Checkpoint:
  `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/checkpoints/supervised_floras50_spotifytok_safe_norm_drop_oov_lr1e-4_12ep_nw0/step_323484.pt`

The run used the same TEDLIUM test setup as ROB-25 for the source checkpoint:
`EncDecSconformerV2`, direct decoding, sequence length `2048`, overlap ratio
`0.0`, and `28215` aggregate words.

## Aggregate Result

| checkpoint | WER | ins_rate | del_rate | sub_rate | words |
|---|---:|---:|---:|---:|---:|
| ROB-25 source `enc_dec_3l_no_anorm_v2` | 0.09803296119085593 | 0.013361687045897573 | 0.037568669147616515 | 0.04710260499734184 | 28215 |
| ROB-81 Floras supervised finetune | 0.08258018784334574 | 0.010030125819599504 | 0.0329257487152224 | 0.03962431330852383 | 28215 |

The Floras supervised finetune improved TEDLIUM aggregate WER by
`0.01545277334751019` absolute, a `15.76%` relative reduction versus the ROB-25
source checkpoint result.

## Validation Notes

- Slurm accounting reported `COMPLETED|0:0` for job `10253671`.
- The result CSV has 11 per-recording rows plus one aggregate `recording=all`
  row.
- The aggregate row reports `28215` words, matching the ROB-25 comparison row.
- A Mimas GPU attempt was not used for the final result because the local Mimas
  torch environment failed this model path at scaled-dot-product attention with
  the `scale` keyword. The final comparison therefore used the user-approved
  Stanage CPU route.
