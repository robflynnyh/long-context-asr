#!/usr/bin/env python3
import argparse
import json
import os
import urllib.error
import urllib.request


API_URL = "https://api.linear.app/graphql"


def graphql(query, variables, token):
    request = urllib.request.Request(
        API_URL,
        data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
        headers={
            "Authorization": token,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Linear HTTP error {exc.code}: {body}") from exc
    if payload.get("errors"):
        raise SystemExit(json.dumps(payload["errors"], indent=2))
    return payload["data"]


def issue_and_state_id(issue_id, state_name, token):
    data = graphql(
        """
        query IssueState($id: String!) {
          issue(id: $id) {
            id
            identifier
            team {
              states(first: 100) {
                nodes { id name }
              }
            }
          }
        }
        """,
        {"id": issue_id},
        token,
    )
    issue = data["issue"]
    for state in issue["team"]["states"]["nodes"]:
        if state["name"] == state_name:
            return issue["id"], state["id"]
    raise SystemExit(f"State {state_name!r} not found for issue {issue['identifier']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", default="ROB-69")
    parser.add_argument("--state", default="Todo")
    parser.add_argument("--status", required=True)
    parser.add_argument("--summary-file")
    parser.add_argument("--log-path")
    parser.add_argument("--result-path")
    parser.add_argument("--job-id")
    args = parser.parse_args()

    token = os.environ.get("LINEAR_API_KEY")
    if not token:
        raise SystemExit("LINEAR_API_KEY is required")

    issue_id, state_id = issue_and_state_id(args.issue, args.state, token)
    body = [f"Experiment callback: ROB-69 18L long-context finetuning eval {args.status}."]
    if args.job_id:
        body.append(f"Slurm job: `{args.job_id}`")
    if args.log_path:
        body.append(f"Log: `{args.log_path}`")
    if args.result_path:
        body.append(f"Result CSV: `{args.result_path}`")
    if args.summary_file and os.path.exists(args.summary_file):
        with open(args.summary_file, "r", encoding="utf-8") as handle:
            summary = handle.read().strip()
        if summary:
            body.append("")
            body.append(summary[:12000])

    graphql(
        """
        mutation CallbackUpdate($issueId: String!, $body: String!, $stateId: String!) {
          commentCreate(input: {issueId: $issueId, body: $body}) {
            success
          }
          issueUpdate(id: $issueId, input: {stateId: $stateId}) {
            success
          }
        }
        """,
        {"issueId": issue_id, "body": "\n".join(body), "stateId": state_id},
        token,
    )


if __name__ == "__main__":
    main()
