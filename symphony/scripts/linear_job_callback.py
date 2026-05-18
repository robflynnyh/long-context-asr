#!/usr/bin/env python3
import argparse
import json
import os
import sys
import urllib.error
import urllib.request


LINEAR_URL = "https://api.linear.app/graphql"


def graphql(query, variables, api_key):
    req = urllib.request.Request(
        LINEAR_URL,
        data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
        headers={
            "Authorization": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("errors"):
        raise RuntimeError(json.dumps(payload["errors"], indent=2))
    return payload["data"]


def read_tail(path, max_chars):
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_chars), os.SEEK_SET)
        data = f.read().decode("utf-8", errors="replace")
    if len(data) > max_chars:
        data = data[-max_chars:]
    return data.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue-id", required=True)
    parser.add_argument("--state-name", default="Todo")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--exit-code", required=True)
    parser.add_argument("--log-out", required=True)
    parser.add_argument("--log-err", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--summary-file", default="")
    parser.add_argument("--title", default="Experiment job finished")
    parser.add_argument("--job-label", default="Slurm job")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-log-chars", type=int, default=3500)
    args = parser.parse_args()

    summary = read_tail(args.summary_file, args.max_log_chars)
    stderr_tail = read_tail(args.log_err, args.max_log_chars)
    stdout_tail = read_tail(args.log_out, args.max_log_chars)

    status = "success" if str(args.exit_code) == "0" else "failure"
    body_parts = [
        f"{args.title}: {status}",
        "",
        f"- {args.job_label}: `{args.job_id}`",
        f"- Exit code: `{args.exit_code}`",
        f"- Stdout: `{args.log_out}`",
        f"- Stderr: `{args.log_err}`",
        f"- Output/checkpoint path: `{args.output_path}`",
        "",
        f"Moving issue back to `{args.state_name}` for Symphony follow-up.",
    ]
    if summary:
        body_parts.extend(["", "Summary tail:", "```text", summary, "```"])
    elif stderr_tail:
        body_parts.extend(["", "Stderr tail:", "```text", stderr_tail, "```"])
    elif stdout_tail:
        body_parts.extend(["", "Stdout tail:", "```text", stdout_tail, "```"])
    body = "\n".join(body_parts)
    if len(body) > 9000:
        body = body[:8900] + "\n\n[callback body truncated]"

    if args.dry_run:
        print(body)
        return 0

    api_key = os.environ.get("LINEAR_API_KEY")
    if not api_key:
        print("LINEAR_API_KEY is required unless --dry-run is set", file=sys.stderr)
        return 2

    issue_data = graphql(
        """
        query IssueForCallback($id: String!) {
          issue(id: $id) {
            id
            team { states { nodes { id name } } }
          }
        }
        """,
        {"id": args.issue_id},
        api_key,
    )
    issue = issue_data["issue"]
    if issue is None:
        raise RuntimeError(f"issue not found: {args.issue_id}")
    state_id = None
    for state in issue["team"]["states"]["nodes"]:
        if state["name"] == args.state_name:
            state_id = state["id"]
            break
    if state_id is None:
        raise RuntimeError(f"state not found: {args.state_name}")

    graphql(
        """
        mutation AddComment($input: CommentCreateInput!) {
          commentCreate(input: $input) { success }
        }
        """,
        {"input": {"issueId": issue["id"], "body": body}},
        api_key,
    )
    graphql(
        """
        mutation UpdateIssue($id: String!, $input: IssueUpdateInput!) {
          issueUpdate(id: $id, input: $input) { success issue { identifier state { name } } }
        }
        """,
        {"id": issue["id"], "input": {"stateId": state_id}},
        api_key,
    )
    print(f"posted callback and moved {args.issue_id} to {args.state_name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        raise
