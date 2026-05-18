#!/usr/bin/env python3
"""Post ROB-70 experiment completion back to Linear.

This helper is intentionally stdlib-only so Slurm wrappers can call it from a
plain training environment. It moves the issue back to Todo so Symphony can
inspect the run and decide whether to finalize or debug.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request


GRAPHQL_URL = "https://api.linear.app/graphql"


def graphql(query, variables, api_key):
    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    request = urllib.request.Request(
        GRAPHQL_URL,
        data=payload,
        headers={
            "Authorization": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))
    if result.get("errors"):
        raise RuntimeError(result["errors"])
    return result["data"]


def tail_text(path, max_chars=6000):
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_chars), os.SEEK_SET)
        return handle.read().decode("utf-8", errors="replace")


def slurm_state(job_id):
    if not job_id or not job_id.isdigit():
        return None
    try:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                job_id,
                "--format=JobID,State,ExitCode",
                "-P",
                "-n",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return f"unavailable ({exc})"
    if result.returncode != 0:
        stderr = result.stderr.strip()
        return f"unavailable ({stderr or 'sacct failed'})"
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= 3 and parts[0] == job_id:
            return f"{parts[1]} {parts[2]}"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue-id", default="ROB-70")
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID", "unknown"))
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--wandb-name", default="rob70_bestrq_6l_2048_1epoch")
    parser.add_argument("--command", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    observed_slurm_state = slurm_state(args.slurm_job_id)
    failed_slurm_prefixes = (
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    )
    slurm_failed = bool(
        observed_slurm_state
        and observed_slurm_state.startswith(failed_slurm_prefixes)
    )
    status = "succeeded" if args.exit_code == 0 and not slurm_failed else "failed"
    log_tail = tail_text(args.train_log)
    body = (
        f"ROB-70 BEST-RQ training callback: `{status}`\n\n"
        f"- Slurm job: `{args.slurm_job_id}`\n"
        f"- Exit code: `{args.exit_code}`\n"
        f"- Slurm state: `{observed_slurm_state or 'unknown'}`\n"
        f"- Train log: `{args.train_log}`\n"
        f"- Checkpoint dir: `{args.checkpoint_dir}`\n"
        f"- W&B run name: `{args.wandb_name}`\n"
        f"- Command: `{args.command}`\n\n"
        "Recent log tail:\n"
        "```text\n"
        f"{log_tail[-6000:]}\n"
        "```"
    )

    if args.dry_run:
        print(body)
        return 0

    api_key = os.environ.get("LINEAR_API_KEY")
    if not api_key:
        print("LINEAR_API_KEY is not set; cannot post callback", file=sys.stderr)
        return 2

    issue_data = graphql(
        """
        query IssueAndStates($id: String!) {
          issue(id: $id) {
            id
            team {
              states(first: 100) {
                nodes { id name }
              }
            }
          }
        }
        """,
        {"id": args.issue_id},
        api_key,
    )
    issue = issue_data["issue"]
    todo_state_id = None
    for state in issue["team"]["states"]["nodes"]:
        if state["name"] == "Todo":
            todo_state_id = state["id"]
            break
    if todo_state_id is None:
        raise RuntimeError("Could not find Linear state named Todo")

    graphql(
        """
        mutation CreateComment($input: CommentCreateInput!) {
          commentCreate(input: $input) { success }
        }
        """,
        {"input": {"issueId": issue["id"], "body": body}},
        api_key,
    )
    graphql(
        """
        mutation UpdateIssue($id: String!, $input: IssueUpdateInput!) {
          issueUpdate(id: $id, input: $input) { success }
        }
        """,
        {"id": issue["id"], "input": {"stateId": todo_state_id}},
        api_key,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
