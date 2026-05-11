---
tracker:
  kind: linear
  project_slug: "long-context-asr-16cd3fe5f7e3"
  api_key: $LINEAR_API_KEY
  active_states:
    - Todo
    - In Progress
    - Rework
    - Merging
  terminal_states:
    - Closed
    - Cancelled
    - Canceled
    - Duplicate
    - Done
polling:
  interval_ms: 30000
workspace:
  root: /exp/exp4/acp21rjf/symphony-workspaces-long-context-asr
hooks:
  timeout_ms: 120000
  after_create: |
    if [ -f /exp/exp4/acp21rjf/long-context-asr/symphony/.env ]; then
      set -a
      . /exp/exp4/acp21rjf/long-context-asr/symphony/.env
      set +a
    fi
    git clone --branch dev --depth 1 "$SOURCE_REPO_URL" .
    if [ -n "${SOURCE_REF:-}" ]; then
      git fetch --depth 1 origin "$SOURCE_REF"
      git checkout -B symphony-source FETCH_HEAD
    fi
agent:
  max_concurrent_agents: 1
  max_turns: 30
codex:
  command: /home/acp21rjf/.npm-global/bin/codex --config shell_environment_policy.inherit=all --config 'model="gpt-5.5"' --config model_reasoning_effort=high app-server
  approval_policy: never
  thread_sandbox: danger-full-access
  turn_sandbox_policy:
    type: dangerFullAccess
---

You are working on Linear issue {{ issue.identifier }} for the
long-context-asr repository.

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

Required instruction index:
- Before planning or editing, read every file in `symphony/instructions/` in
  the order below.
- Treat those files as binding project instructions, not optional background.
- In your plan, state that you read them and note any instruction that directly
  affects the issue.

Read order:
1. `symphony/instructions/linear-context.md`
2. `symphony/instructions/repository.md`
3. `symphony/instructions/work-loop.md`
4. `symphony/instructions/experiment-execution.md`
5. `symphony/instructions/validation-and-handoff.md`

Instruction map:
- Linear issue context, comment rereads, and clarification handling:
  `symphony/instructions/linear-context.md`
- Repository layout, local configuration, and artifact policy:
  `symphony/instructions/repository.md`
- Branch setup, planning, implementation style, and progress comments:
  `symphony/instructions/work-loop.md`
- Stanage-default execution, Mimas opt-in rules, Slurm jobs, detached jobs, and
  callbacks: `symphony/instructions/experiment-execution.md`
- Validation, GitHub PR handoff, Linear review handoff, and final state rules:
  `symphony/instructions/validation-and-handoff.md`
