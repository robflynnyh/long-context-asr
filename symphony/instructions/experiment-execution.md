# Experiment Execution

Do not launch long-running GPU work unless the issue asks for a run.

The Symphony service is launched from Mimas, but Stanage is the default
execution target for this project. Treat Mimas as the coordination host for
repository edits, Linear updates, short inspections, and bounded local checks.
Use the current Mimas server for compute only when the issue or a later human
Linear comment explicitly asks for Mimas/local execution.

Never use `/tmp` on Mimas for Symphony work. For short-lived local scratch on
Mimas, use the repo-local ignored `.tmp/` directory or another issue-specific,
user-owned path under `/exp/exp4/acp21rjf/`, and clean it up before handoff
unless it is intentionally retained as validation evidence.

For default Stanage work, use short bounded SSH commands from Mimas and submit
meaningful compute through Slurm. Do not run training, ASR evaluation, large
preprocessing, large scans, or multi-minute validation on a Stanage login node.

Follow `symphony/slurm-notes.md` and the repo's existing `exp/`, `eval/`, and
`job_scripts/` Slurm patterns. Reuse existing partitions, module/conda setup,
log locations, and resource requests when possible.

Before queueing any Stanage GPU job, always run the smallest practical CPU
smoke test on Stanage against the same code path, config family, data paths,
checkpoint paths, imports, and output directory assumptions. This CPU smoke can
be a reduced-recording, reduced-step, validation-only, syntax-plus-load, or
callback-only job, but it must exercise the failure-prone setup that the GPU job
will rely on. Do not submit the GPU job until that CPU smoke test completes
successfully and its logs show no immediate setup, import, data-loading,
checkpoint, permission, callback, or output-path errors. Stanage GPU jobs can
wait in queue for a long time; a job that fails as soon as it is allocated is a
waste of queue time.

Keep Stanage stdout/stderr and generated artifacts under durable issue-specific
paths on `/mnt/parscratch/users/acp21rjf/`, usually under
`/mnt/parscratch/users/acp21rjf/symphony-job-artifacts`. Use
`/mnt/parscratch/users/acp21rjf/symphony-tmp` for short-lived temp files.

Hard data protection rule: never edit, delete, move, rename, overwrite, clean
up, or reorganize existing training data, evaluation data, manifests,
checkpoints, logs, or experiment outputs under `/mnt/parscratch/users/acp21rjf`
unless the issue explicitly asks for that exact mutation. Treat existing
Stanage artifacts as read-only evidence.

For Stanage jobs, record the job ID, script path, log paths, purpose, expected
outputs, follow-up command, git branch, and commit in the Linear workpad or
queue comment.

For short jobs, use bounded status commands such as:

```bash
squeue -j <job_id> -o '%i|%j|%T|%R|%S|%M|%l|%P'
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed
```

If `sacct` is unavailable, rely on `squeue` disappearance, logs, and generated
success markers. Treat nonzero exit codes, failed/cancelled/timeout states,
tracebacks, uncaught exceptions, and obvious error lines as validation failures.

Keep Slurm and SSH output bounded. Prefer targeted commands such as `rg` with
specific patterns, `head`, `tail`, `sed -n`, filtered `find`, `git diff --stat`,
`squeue -j <job_id> -o <fields>`, and `sacct -j <job_id> --format=<fields>`.
Do not dump large logs, full Slurm outputs, large diffs, full CSVs, generated
data, or broad search results into Codex context unless genuinely needed for
the next decision.

Do not spend agent turns waiting for a queued or running experiment to start or
finish. After queueing a long experiment, post a Linear comment with the queued
command, Slurm job ID, log path, expected result path, git branch and commit,
callback/handoff path, and exact completion-check command. Then move the issue
back to the Linear state named `Backlog`.

Every queued long experiment should have a completion callback or finalizer path
before it is queued. The callback/finalizer must run when the experiment exits
for success, nonzero exit, Python exception, shell error, timeout-wrapper exit,
or manual termination where the shell can still run traps.

Prefer an `EXIT` trap or equivalent wrapper-level hook that records the
experiment exit status, then calls a Linear callback helper from the repo or an
issue-local script. The callback should post success or failure evidence, log
path, output path, and residual risk, then move the issue back to `Todo` so
Symphony can resume finalization. Detached experiment processes cannot use
Codex-only tools such as `linear_graphql`.

For multi-run Stanage jobs such as sweeps or GPU job arrays, the callback can be
implemented as a separate lightweight Slurm finalizer job instead of a callback
in every array task. Submit the finalizer with a dependency on the GPU array,
for example `--dependency=afterany:<array_job_id>`, and have that finalizer
inspect the array logs/results, compute an overall status, call the callback,
and move the issue back to `Todo`. Record both the array job ID and finalizer
job ID in the Linear queue comment.

Do not queue a long GPU or CPU experiment if the launched code lacks this
completion callback or finalizer path. First add or fix the hook, then validate
the actual wrapper or finalizer with the smallest practical smoke test or
callback-only dry run. Test the wrapper/finalizer that will actually be
launched, not just the callback helper in isolation.

When Symphony relaunches from a callback comment, inspect the log and results
before deciding whether to finalize, diagnose, or rerun. If a run failed, fix
the concrete issue before queueing another run. Do not blindly relaunch an
unchanged failing command.

Use Mimas compute only when a human explicitly asks for it. In that case, use
`/store/store5/software/simple-gpu-schedule/with-gpu` for cooperative GPU
allocation, launch long jobs in durable detached `screen` sessions with log
files, never write temporary files under `/tmp`, and follow the same
callback/handoff discipline above.
