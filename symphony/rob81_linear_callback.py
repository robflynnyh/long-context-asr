#!/usr/bin/env python3
"""Post ROB-81 Slurm completion status back to Linear."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


LINEAR_URL = "https://api.linear.app/graphql"
ISSUE_ID = "ROB-81"
MAX_LOG_CHARS = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-job-id", required=True)
    parser.add_argument("--status", choices=("success", "failure"), required=True)
    parser.add_argument("--out-log", required=True)
    parser.add_argument("--err-log", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def tail_text(path: str, chars: int = MAX_LOG_CHARS) -> str:
    log_path = Path(path)
    if not log_path.exists():
        return f"{path} does not exist"
    data = log_path.read_text(encoding="utf-8", errors="replace")
    return data[-chars:]


def graphql(api_key: str, query: str, variables: dict) -> dict:
    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    request = urllib.request.Request(
        LINEAR_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": api_key,
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    path = Path(checkpoint_dir)
    if not path.exists():
        return "checkpoint directory missing"
    checkpoints = sorted(path.glob("step_*.pt"), key=lambda item: int(item.stem.split("_")[1]))
    return str(checkpoints[-1]) if checkpoints else "no checkpoints found"


def sacct_summary(job_id: str) -> str:
    try:
        return subprocess.check_output(
            ["sacct", "-j", job_id, "--format=JobID,JobName,State,ExitCode,Elapsed", "-n", "-P"],
            text=True,
            timeout=20,
        ).strip()
    except Exception as exc:
        return f"sacct unavailable: {exc}"


def build_comment(args: argparse.Namespace) -> str:
    latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
    status_title = "completed successfully" if args.status == "success" else "finished with failure"
    body = f"""## ROB-81 finetune job {status_title}

GPU job: `{args.gpu_job_id}`
Slurm summary:
```text
{sacct_summary(args.gpu_job_id)}
```

Checkpoint dir: `{args.checkpoint_dir}`
Latest checkpoint: `{latest_checkpoint}`
Stdout: `{args.out_log}`
Stderr: `{args.err_log}`

Stderr tail:
```text
{tail_text(args.err_log)}
```

Stdout tail:
```text
{tail_text(args.out_log)}
```

The issue is moved back to `Todo` for Symphony to inspect logs/results and decide whether to finalize or diagnose.
"""
    return body[:15000]


def main() -> None:
    args = parse_args()
    comment = build_comment(args)
    if args.dry_run:
        print(comment)
        return

    api_key = os.environ.get("LINEAR_API_KEY")
    if not api_key:
        raise RuntimeError("LINEAR_API_KEY is required")

    issue_query = """
    query IssueState($id: String!) {
      issue(id: $id) {
        id
        team { states { nodes { id name } } }
      }
    }
    """
    issue_result = graphql(api_key, issue_query, {"id": ISSUE_ID})
    issue = issue_result["data"]["issue"]
    todo_state_id = next(
        state["id"]
        for state in issue["team"]["states"]["nodes"]
        if state["name"] == "Todo"
    )

    graphql(
        api_key,
        "mutation CommentCreate($input: CommentCreateInput!) { commentCreate(input: $input) { success } }",
        {"input": {"issueId": issue["id"], "body": comment}},
    )
    graphql(
        api_key,
        "mutation IssueUpdate($id: String!, $input: IssueUpdateInput!) { issueUpdate(id: $id, input: $input) { success } }",
        {"id": issue["id"], "input": {"stateId": todo_state_id}},
    )


if __name__ == "__main__":
    main()
