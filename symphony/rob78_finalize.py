#!/usr/bin/env python3
"""Slurm finalizer and Linear callback for ROB-78."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "https://api.linear.app/graphql"
ISSUE_ID = "71d3aa55-a418-4c74-bcfd-201c3047aa97"
ISSUE_KEY = "ROB-78"
ARTIFACT_ROOT = Path("/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-78")
OUTPUT_ROOT = Path("/mnt/parscratch/users/acp21rjf/spotify/long_only/FT_3epoch_6epoch")


def run_command(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.stdout.strip()


def linear_graphql(api_key: str, query: str, variables: dict) -> dict:
    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    request = Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": api_key},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Linear API request failed: {exc}") from exc
    if data.get("errors"):
        raise RuntimeError(f"Linear API returned errors: {data['errors']}")
    return data


def get_state_id(api_key: str, state_name: str) -> str:
    query = """
    query IssueStates($id: String!) {
      issue(id: $id) {
        team {
          states {
            nodes { id name }
          }
        }
      }
    }
    """
    data = linear_graphql(api_key, query, {"id": ISSUE_ID})
    states = data["data"]["issue"]["team"]["states"]["nodes"]
    for state in states:
        if state["name"] == state_name:
            return state["id"]
    raise RuntimeError(f"could not find Linear state named {state_name!r}")


def post_linear_comment(api_key: str, body: str) -> None:
    mutation = """
    mutation CommentCreate($input: CommentCreateInput!) {
      commentCreate(input: $input) { success }
    }
    """
    linear_graphql(api_key, mutation, {"input": {"issueId": ISSUE_ID, "body": body}})


def move_issue(api_key: str, state_name: str) -> None:
    mutation = """
    mutation IssueUpdate($id: String!, $input: IssueUpdateInput!) {
      issueUpdate(id: $id, input: $input) { success }
    }
    """
    linear_graphql(api_key, mutation, {"id": ISSUE_ID, "input": {"stateId": get_state_id(api_key, state_name)}})


def summarize_sacct(array_job_id: str) -> tuple[str, bool]:
    output = run_command([
        "sacct",
        "-j",
        array_job_id,
        "--parsable2",
        "--noheader",
        "--format=JobID,State,ExitCode,Elapsed",
    ])
    bad_markers = ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL")
    failed = any(marker in output for marker in bad_markers)
    return output or "sacct produced no rows", failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--finalizer-job-id", default=os.environ.get("SLURM_JOB_ID", "unknown"))
    parser.add_argument("--state", default="Todo")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sacct_summary, failed = summarize_sacct(args.array_job_id)
    status = "failed or incomplete" if failed else "finished without Slurm failure markers"
    follow_up = (
        f"ssh acp21rjf@stanage.shef.ac.uk \"sacct -j {args.array_job_id} "
        "--format=JobID,JobName,State,ExitCode,Elapsed; "
        f"tail -80 {ARTIFACT_ROOT}/gpu-{args.array_job_id}_0.err\""
    )
    body = (
        f"ROB-78 training array {status}.\n\n"
        f"Array job: `{args.array_job_id}`\n"
        f"Finalizer job: `{args.finalizer_job_id}`\n"
        f"Artifact root: `{ARTIFACT_ROOT}`\n"
        f"Expected checkpoint root: `{OUTPUT_ROOT}`\n"
        f"Follow-up command: `{follow_up}`\n\n"
        "Slurm summary:\n"
        f"```text\n{sacct_summary[:3500]}\n```"
    )

    if args.dry_run or os.environ.get("ROB78_CALLBACK_DRY_RUN") == "1":
        print(body)
        print(f"would move {ISSUE_KEY} to {args.state}")
        return 1 if failed else 0

    api_key = os.environ.get("LINEAR_API_KEY")
    if not api_key:
        print("LINEAR_API_KEY is required for ROB-78 finalizer", file=sys.stderr)
        print(body)
        return 4

    post_linear_comment(api_key, body)
    move_issue(api_key, args.state)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
