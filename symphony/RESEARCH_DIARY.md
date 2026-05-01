# Symphony Research Diary

This diary is for concise, durable notes from Symphony-managed work on this repository.

Agents should append dated entries for meaningful implementation changes, experiment launches, completed Slurm jobs, failed runs, fixes, validation outcomes, PR handoffs, and interpretation updates. Keep large logs and artifacts in parscratch and reference their paths here.

## 2026-05-01

- ROB-24 on branch `symphony/ROB-24-agent-docs`: identified sparse/scattered operational docs, added focused future-agent notes under `symphony/` for repo orientation, training, evaluation, and Slurm patterns. No Slurm jobs launched.
- ROB-24 PR handoff: opened GitHub PR https://github.com/robflynnyh/long-context-asr/pull/2 against `dev` after docs validation.
- ROB-28 on branch `symphony/ROB-28-setup-ssl-model`: fixed BEST-RQ SSL trainer path to call `BestRQ`, normalize masked-token CE loss, pass augmentation, and resume BEST-RQ projection/quantizer state; added ROB-28 SSL/fine-tune configs and Slurm launchers.
- ROB-28 validation: CPU Slurm smoke job `10094333` completed (`COMPLETED 0:0`, 00:01:07); logs under `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-28/logs/best-rq-smoke-10094333.{out,err}` show finite BEST-RQ loss and no traceback.
- ROB-28 validation: submitted GPU one-batch SSL dry-run job `10094336` with script `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-28/scripts/ssl_gpu_dry_run.sbatch`; pending at time of entry.
- ROB-28 validation: after dtype hardening, CPU Slurm smoke job `10094348` completed (`COMPLETED 0:0`, 00:02:03); logs under `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-28/logs/best-rq-smoke-10094348.{out,err}` again show finite BEST-RQ loss and no traceback.
- ROB-28 experiment queue: submitted dependent Slurm chain: SSL 1-epoch Spotify job `10094411` (`afterok:10094336`, logs `ssl-10094411.{out,err}`), floras-50 fine-tune job `10094412` (`afterok:10094411`, logs `finetune-10094412.{out,err}`), earnings22-full eval job `10094413` (`afterok:10094412`, logs `eval-10094413.{out,err}`). All write under `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-28/` except the eval result CSV path in the repo.
- ROB-28 commit: initial implementation committed as `11bc83fb6fb2aca592ec9d82ee27019262d94c0d` on `symphony/ROB-28-setup-ssl-model`; Slurm chain was pending at commit time.
- ROB-28 queue adjustment: H100 queue estimate was several days out, so job scripts were moved to `gpu-h100-nvl` and committed as `72bc4674c10a4c42880417bbb4b36e0254aea55a`. Superseded pending H100 jobs `10094336`, `10094411`, `10094412`, and `10094413` were canceled before running. Replacement NVL chain: dry run `10094428`, SSL `10094431` (`afterok:10094428`), fine-tune `10094432` (`afterok:10094431`), eval `10094434` (`afterok:10094432`).
