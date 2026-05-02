---
tracker:
  kind: linear
  project_slug: "long-context-asr-16cd3fe5f7e3"
  api_key: $LINEAR_API_KEY
  active_states:
    - Todo
    - In Progress
    - Merging
    - Rework
  terminal_states:
    - Closed
    - Cancelled
    - Canceled
    - Duplicate
    - Done
polling:
  interval_ms: 5000
workspace:
  root: /mnt/parscratch/users/acp21rjf/symphony-workspaces
hooks:
  timeout_ms: 300000
  after_create: |
    git clone --branch dev /users/acp21rjf/long-context-asr .
    git remote set-url origin https://github.com/robflynnyh/long-context-asr.git
    git fetch origin dev
    git checkout dev
    git reset --hard origin/dev
agent:
  max_concurrent_agents: 1
  max_turns: 10
codex:
  command: >-
    module load conda_alma9_container/v1;
    eval "$(conda shell.bash hook)";
    conda activate /mnt/parscratch/users/acp21rjf/conda/main;
    mkdir -p /mnt/parscratch/users/acp21rjf/symphony-tmp;
    export TMPDIR=/mnt/parscratch/users/acp21rjf/symphony-tmp;
    export TEMP=/mnt/parscratch/users/acp21rjf/symphony-tmp;
    export TMP=/mnt/parscratch/users/acp21rjf/symphony-tmp;
    exec codex
    --config shell_environment_policy.inherit=all
    --config 'model="gpt-5.5"'
    --config model_reasoning_effort=high
    app-server
  approval_policy: never
  thread_sandbox: danger-full-access
  turn_sandbox_policy:
    type: dangerFullAccess
---

You are working on Linear ticket `{{ issue.identifier }}` for the long-context-asr repository.

Issue context:
Identifier: {{ issue.identifier }}
Title: {{ issue.title }}
Current status: {{ issue.state }}
Labels: {{ issue.labels }}
URL: {{ issue.url }}

Description:
{% if issue.description %}
{{ issue.description }}
{% else %}
No description provided.
{% endif %}

Instructions:

1. Work only inside the Symphony-provided workspace clone.
2. Before planning or editing, verify the workspace is on `dev` and matches `origin/dev`. The base branch for all work is `dev` unless the issue explicitly says otherwise and a human confirms that override.
   - Run `git status --short --branch`.
   - Run `git fetch origin dev`.
   - Run `git checkout dev`.
   - Run `git reset --hard origin/dev` only if the workspace is the fresh Symphony clone and has no intentional local work from a prior attempt.
3. Before making code changes, read the Linear issue comments and history. Treat recent comments as potentially superseding the issue description, especially if the issue has been partially addressed or the user has added clarification.
   - Use the `linear_graphql` tool when available to fetch recent comments, newest last.
   - Use this query shape with `{{ issue.id }}`:
     ```graphql
     query IssueComments($id: String!) {
       issue(id: $id) {
         comments(first: 20) {
           nodes {
             body
             createdAt
             user {
               name
             }
           }
         }
       }
     }
     ```
   - In the workpad plan, explicitly note which recent comments changed or constrained the task.
   - If a recent human comment asks a question or requests clarification rather than implementation, answer it in Linear and do not make code changes until the task is clear.
   - If recent comments mention an existing branch or PR, inspect it before starting new implementation.
4. If the issue is in `Todo`, move it to `In Progress` before making changes.
5. Keep one persistent Linear workpad comment headed `## Codex Workpad`; update it as work progresses.
6. Create a working branch from `dev` named `symphony/{{ issue.identifier }}-<short-slug>` before committing changes. Do not commit directly to `dev`.
7. Reproduce or identify the requested behavior before editing code.
8. Keep changes narrowly scoped to the issue.
9. Be deliberate with context budget.
   - Do not dump large logs, full Slurm outputs, large diffs, full CSVs, generated data, or broad search results into the Codex context unless the content is genuinely needed for the next decision.
   - Prefer targeted commands such as `rg` with specific patterns, `head`, `tail`, `sed -n`, `squeue -j <job_id> -o <fields>`, `sacct -j <job_id> --format=<fields>`, and filtered `find` or `git diff --stat` output.
   - When logs or outputs are large, inspect only the relevant slices, summarize the result, and reference the file path for follow-up. Keep raw artifacts in parscratch.
   - Use strict output limits for exploratory commands and rerun with a narrower command if more detail is needed.
10. Store large or non-committed working files under parscratch, not in the repo and not in `/tmp`.
   - Hard data protection rule: never edit, delete, move, rename, overwrite, clean up, or reorganize any existing training data, evaluation data, manifests, checkpoints, logs, or experiment outputs under `/mnt/parscratch/users/acp21rjf` under any circumstance.
   - Treat existing parscratch training/evaluation artifacts as read-only evidence. You may inspect paths and read files when needed, but do not mutate them.
   - Use `/mnt/parscratch/users/acp21rjf/symphony-tmp` for temporary files.
   - Use `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts` for job logs, generated outputs, checkpoints, downloaded data, and other non-commit artifacts.
   - Only write new scratch/output artifacts inside the dedicated Symphony parscratch directories above, or inside a new issue-specific subdirectory there.
   - Before finishing the issue, remove temporary files you created under `/mnt/parscratch/users/acp21rjf/symphony-tmp` unless they are needed as explicit validation evidence. If retained, document the exact path and reason in the workpad.
   - Do not commit large generated files, model checkpoints, logs, datasets, caches, or local environment files.
11. Use Slurm for long-running, CPU/GPU, or cluster-scale validation instead of running heavy work in the interactive agent process.
   - Login-node work is limited to repository inspection, file edits, formatting, small metadata checks, and bounded searches. Put meaningful compute, training, full evals, large scans, data processing, and multi-minute validation in Slurm.
   - Use CPU Slurm jobs for significant non-CUDA work and GPU Slurm jobs for model training or ASR evals. If the GPU queue is long, use CPU jobs to debug script, config, path, data-loading, or lightweight dry-run failures that do not require CUDA.
   - Reuse existing repo `sbatch`/`srun` patterns for partitions, modules, conda activation, logs, and resources. Prefer a small Slurm script when the command needs environment setup or will run for more than a few minutes.
   - Put Slurm stdout/stderr under `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts`, submit with `sbatch <script>`, and record the job ID, script path, log paths, purpose, expected outputs, follow-up command, and current git state in the workpad.
   - For short jobs, use bounded `squeue -j <job_id>` and `sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed`, then inspect logs. If `sacct` is unavailable, rely on `squeue` disappearance, logs, and generated success markers.
   - Treat nonzero exit codes, failed/cancelled/timeout states, tracebacks, uncaught exceptions, and obvious error lines as validation failures.
   - Wait for Slurm jobs only when they are expected to finish within a short validation window and can be checked with a small number of bounded status/log commands.
   - Do not keep Codex active for long-running jobs or GPU queue availability. After launching a job expected to queue or run for a long time, hand off in the workpad with the exact completion-check command and stop the agent turn.
   - Do not repeatedly poll `squeue`, run replacement queue probes, or resume solely to see whether a pending GPU job has started. If queue placement is uncertain, perform at most one deliberate placement check, record the decision, and hand off.
   - Do not move the issue to `Human Review`, mark it done, or claim validation passed until every required Slurm job has finished and its logs show no errors. For long-running training/eval jobs, leave the issue in progress with a clear handoff unless the job has already completed and the logs were inspected in this turn.
12. Keep the repository research diary up to date.
   - Append concise dated entries to `symphony/RESEARCH_DIARY.md` for meaningful implementation changes, experiment launches, completed Slurm jobs, failed runs, fixes, validation outcomes, PR handoffs, and important interpretation updates.
   - Keep entries factual and brief: issue ID, branch, command/job ID when relevant, outcome, and links or paths to logs/artifacts.
   - Do not use the diary as a live work log. Do not add routine queue polls, repeated resume checks, transient scheduler estimates, or every failed replacement probe. Summarize repeated attempts as one outcome-oriented entry.
   - Do not put credentials, raw data, large logs, or large generated output in the diary; reference parscratch paths instead.
13. Document hard-won repo knowledge for future agents.
   - If you spend meaningful time figuring out a non-obvious repo procedure, dependency, Slurm pattern, data/checkpoint location, evaluation command, failure mode, or environment setup, add or update a concise note under `symphony/`.
   - Prefer small focused docs such as `symphony/agent-notes.md`, `symphony/slurm-notes.md`, or `symphony/eval-notes.md` over long prose in the workpad.
   - Include reusable commands, relevant paths, and caveats, but avoid issue-specific blow-by-blow history. Keep raw investigation details in the Linear workpad unless they are broadly useful.
   - Reference the new or updated doc in the workpad and, when meaningful, in `symphony/RESEARCH_DIARY.md`.
14. For experiment or evaluation work, preserve reproducibility:
   - Record exact commands, configs, checkpoint paths, input manifests, output paths, log paths, commit SHA, and Slurm job IDs in the workpad.
   - Extract metrics programmatically from output artifacts. Do not hand-compute, infer, or report metrics from impressions or partial logs.
   - If a result cannot be extracted programmatically yet, add the extraction gap to the workpad instead of writing an unsupported numeric conclusion.
15. Run relevant validation before handoff.
16. GitHub handoff is required for completed code changes:
   - Commit completed changes on the issue branch, not on `dev`.
   - Push the issue branch to `origin`.
   - Open a GitHub PR against `dev` with `gh pr create` when available.
   - Include the PR URL in the Linear workpad or completion comment.
   - If commit, push, or PR creation fails, do not move the issue to `Human Review`; record the exact failing command and error in the workpad as a blocker.
17. Before ending, verify that the expected Linear workpad/completion comment exists and that the issue is in the intended state.
18. When blocked by missing credentials, permissions, or unavailable infrastructure, record the blocker in the workpad and move the issue to `Human Review`.
19. When implementation, validation, and GitHub handoff are complete, move the issue to `Human Review`.

Use the injected Linear tool for issue updates when available.
