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
  after_create: |
    git clone --branch dev /users/acp21rjf/long-context-asr .
    git remote set-url origin https://github.com/robflynnyh/long-context-asr.git
    git fetch origin dev
    git checkout dev
    git reset --hard origin/dev
agent:
  max_concurrent_agents: 5
  max_turns: 30
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
    --config model_reasoning_effort=xhigh
    app-server
  approval_policy: never
  thread_sandbox: workspace-write
  turn_sandbox_policy:
    type: workspaceWrite
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
9. Store large or non-committed working files under parscratch, not in the repo and not in `/tmp`.
   - Use `/mnt/parscratch/users/acp21rjf/symphony-tmp` for temporary files.
   - Use `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts` for job logs, generated outputs, checkpoints, downloaded data, and other non-commit artifacts.
   - Before finishing the issue, remove temporary files you created under `/mnt/parscratch/users/acp21rjf/symphony-tmp` unless they are needed as explicit validation evidence. If retained, document the exact path and reason in the workpad.
   - Do not commit large generated files, model checkpoints, logs, datasets, caches, or local environment files.
10. Use Slurm for long-running, CPU/GPU, or cluster-scale validation instead of running heavy work in the interactive agent process.
   - Symphony is expected to run from a login node. All repository file inspection and file edits may be done on the login node, including normal searches, reads, patching, formatting, and small metadata checks.
   - Use discretion for very large searches or scans. If a search may traverse large datasets, checkpoint trees, generated outputs, parscratch-wide paths, or otherwise run for more than a few minutes, put it in a CPU Slurm job instead of running it on the login node.
   - Do not run meaningful compute, training, full evaluations, large data processing, or multi-minute validation directly on the login node.
   - Use a CPU Slurm job for any significant non-GPU work, including heavier tests, dataset preprocessing, metric extraction, and reproductions/debugging that do not require CUDA.
   - Use a GPU Slurm job for model training and most ASR evaluations.
   - If the GPU queue is long, it is acceptable to use a CPU Slurm job to debug script errors, config parsing, data loading, path issues, lightweight dry-runs, or other failures that can be reproduced without CUDA.
   - First look for existing repo scripts and patterns for `sbatch`, `srun`, partitions, modules, conda activation, log paths, and resource settings.
   - Prefer adding or updating a small Slurm script when the command needs environment setup or will run for more than a few minutes.
   - Put Slurm stdout/stderr logs under `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts` or another clearly named parscratch path.
   - Submit with `sbatch <script>` and record the job ID, script path, log paths, and purpose in the workpad.
   - Monitor with `squeue -j <job_id>` while running, then confirm completion with `sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed` when available.
   - Inspect stdout/stderr logs after completion. Treat nonzero exit codes, failed/cancelled/timeout states, tracebacks, uncaught exceptions, and obvious error lines as validation failures.
   - If `sacct` is unavailable, rely on `squeue` disappearance plus the Slurm output logs and any generated success markers.
   - Do not move the issue to `Human Review`, mark it done, or claim validation passed until every Slurm job you launched for the issue has finished and its logs show no errors.
11. Keep the repository research diary up to date.
   - Append concise dated entries to `symphony/RESEARCH_DIARY.md` for meaningful implementation changes, experiment launches, completed Slurm jobs, failed runs, fixes, validation outcomes, PR handoffs, and important interpretation updates.
   - Keep entries factual and brief: issue ID, branch, command/job ID when relevant, outcome, and links or paths to logs/artifacts.
   - Do not put credentials, raw data, large logs, or large generated output in the diary; reference parscratch paths instead.
12. Document hard-won repo knowledge for future agents.
   - If you spend meaningful time figuring out a non-obvious repo procedure, dependency, Slurm pattern, data/checkpoint location, evaluation command, failure mode, or environment setup, add or update a concise note under `symphony/`.
   - Prefer small focused docs such as `symphony/agent-notes.md`, `symphony/slurm-notes.md`, or `symphony/eval-notes.md` over long prose in the workpad.
   - Include exact commands, relevant paths, and caveats, but do not include secrets or large outputs.
   - Reference the new or updated doc in the workpad and, when meaningful, in `symphony/RESEARCH_DIARY.md`.
13. For experiment or evaluation work, preserve reproducibility:
   - Record exact commands, configs, checkpoint paths, input manifests, output paths, log paths, commit SHA, and Slurm job IDs in the workpad.
   - Extract metrics programmatically from output artifacts. Do not hand-compute, infer, or report metrics from impressions or partial logs.
   - If a result cannot be extracted programmatically yet, add the extraction gap to the workpad instead of writing an unsupported numeric conclusion.
14. Run relevant validation before handoff.
15. GitHub handoff is required for completed code changes:
   - Commit completed changes on the issue branch, not on `dev`.
   - Push the issue branch to `origin`.
   - Open a GitHub PR against `dev` with `gh pr create` when available.
   - Include the PR URL in the Linear workpad or completion comment.
   - If commit, push, or PR creation fails, do not move the issue to `Human Review`; record the exact failing command and error in the workpad as a blocker.
16. Before ending, verify that the expected Linear workpad/completion comment exists and that the issue is in the intended state.
17. When blocked by missing credentials, permissions, or unavailable infrastructure, record the blocker in the workpad and move the issue to `Human Review`.
18. When implementation, validation, and GitHub handoff are complete, move the issue to `Human Review`.

Use the injected Linear tool for issue updates when available.
