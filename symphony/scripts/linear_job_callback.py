#!/usr/bin/env python3
"""Reusable Linear completion callback for Symphony-managed jobs.

The script is intentionally stdlib-only so Slurm finalizers and detached Mimas
screen wrappers can run it from minimal training environments. It supports the
older issue-local call shape while allowing newer wrappers to pass richer
Stanage or Mimas metadata.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request


LINEAR_URL = "https://api.linear.app/graphql"
FAILED_SLURM_PREFIXES = (
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


class SafeFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


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
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_chars), os.SEEK_SET)
        data = handle.read().decode("utf-8", errors="replace")
    if len(data) > max_chars:
        data = data[-max_chars:]
    return data.strip()


def bounded_text(text, max_chars):
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:].strip()


def split_key_value(values):
    result = []
    for value in values:
        if "=" in value:
            key, item = value.split("=", 1)
            key = key.strip()
            item = item.strip()
            if key:
                result.append((key, item))
        else:
            result.append(("Metadata", value.strip()))
    return result


def terminal_slurm_state(job_id):
    if not job_id or not str(job_id).isdigit():
        return ""
    try:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                str(job_id),
                "--format=JobID,JobName,State,ExitCode,Elapsed",
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
        if len(parts) >= 4 and parts[0] == str(job_id):
            return f"{parts[2]} {parts[3]} elapsed={parts[4] if len(parts) > 4 else 'unknown'}"
    return ""


def infer_status(exit_code, explicit_status, slurm_state):
    if explicit_status:
        return explicit_status
    slurm_failed = bool(slurm_state and slurm_state.startswith(FAILED_SLURM_PREFIXES))
    return "success" if str(exit_code) == "0" and not slurm_failed else "failure"


def format_path_lines(label, values):
    lines = []
    for value in values:
        if value:
            lines.append(f"- {label}: `{value}`")
    return lines


def read_template(path, fields):
    if not path:
        return ""
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read().strip()
    return text.format_map(SafeFormatDict(fields))


def build_body(args):
    mode = args.mode
    slurm_job_id = args.slurm_job_id or os.environ.get("SLURM_JOB_ID", "")
    screen_session = args.screen_session
    if mode == "mimas" and not screen_session:
        screen_session = os.environ.get("STY", "")
    slurm_state = terminal_slurm_state(slurm_job_id) if mode == "stanage" else ""
    status = infer_status(args.exit_code, args.status, slurm_state)

    job_id = args.job_id
    if not job_id and mode == "stanage":
        job_id = slurm_job_id or "unknown"
    if not job_id and mode == "mimas":
        job_id = screen_session or "unknown"
    if not job_id:
        job_id = "unknown"

    default_label = {
        "stanage": "Stanage Slurm job",
        "mimas": "Mimas screen session",
        "generic": "Job",
    }[mode]
    job_label = args.job_label or default_label

    fields = {
        "issue_id": args.issue_id,
        "state_name": args.state_name,
        "status": status,
        "exit_code": str(args.exit_code),
        "job_id": job_id,
        "slurm_job_id": slurm_job_id,
        "screen_session": screen_session,
        "host": os.uname().nodename,
    }
    summary_text = bounded_text(args.summary_text, args.max_log_chars)
    template_text = bounded_text(read_template(args.summary_template, fields), args.max_log_chars)
    summary_file_tail = read_tail(args.summary_file, args.max_log_chars)

    log_entries = []
    if args.log_out:
        log_entries.append(("Stdout", args.log_out))
    if args.log_err:
        log_entries.append(("Stderr", args.log_err))
    for log_path in args.log:
        log_entries.append(("Log", log_path))

    body_parts = [
        f"{args.title}: `{status}`",
        "",
        f"- Mode: `{mode}`",
        f"- {job_label}: `{job_id}`",
        f"- Exit code: `{args.exit_code}`",
        f"- Host: `{fields['host']}`",
    ]
    if slurm_job_id and slurm_job_id != job_id:
        body_parts.append(f"- Slurm job id: `{slurm_job_id}`")
    if slurm_state:
        body_parts.append(f"- Slurm state: `{slurm_state}`")
    if screen_session and screen_session != job_id:
        body_parts.append(f"- Screen session: `{screen_session}`")
    for label, path in log_entries:
        body_parts.append(f"- {label}: `{path}`")
    body_parts.extend(format_path_lines("Artifact path", args.artifact_path))
    body_parts.extend(format_path_lines("Checkpoint path", args.checkpoint_path))
    if args.output_path:
        body_parts.append(f"- Output/checkpoint path: `{args.output_path}`")
    for key, value in split_key_value(args.metadata):
        if value:
            body_parts.append(f"- {key}: `{value}`")
    body_parts.extend(["", f"Moving issue to `{args.state_name}` for Symphony follow-up."])

    excerpt_sections = [
        ("Summary", summary_text),
        ("Summary template", template_text),
        ("Summary file tail", summary_file_tail),
    ]
    for label, path in log_entries:
        excerpt_sections.append((f"{label} tail", read_tail(path, args.max_log_chars)))

    used_excerpt = False
    for label, text in excerpt_sections:
        if text:
            body_parts.extend(["", f"{label}:", "```text", text, "```"])
            used_excerpt = True
            if args.first_excerpt_only:
                break

    if not used_excerpt:
        body_parts.extend(["", "No log excerpt was available."])

    body = "\n".join(body_parts)
    if len(body) > args.max_body_chars:
        body = body[: max(0, args.max_body_chars - 35)] + "\n\n[callback body truncated]"
    return body


def post_callback(issue_id, state_name, body):
    api_key = os.environ.get("LINEAR_API_KEY")
    if not api_key:
        print("LINEAR_API_KEY is required unless --dry-run is set", file=sys.stderr)
        return 2

    issue_data = graphql(
        """
        query IssueForCallback($id: String!) {
          issue(id: $id) {
            id
            team { states(first: 100) { nodes { id name } } }
          }
        }
        """,
        {"id": issue_id},
        api_key,
    )
    issue = issue_data["issue"]
    if issue is None:
        raise RuntimeError(f"issue not found: {issue_id}")
    state_id = None
    for state in issue["team"]["states"]["nodes"]:
        if state["name"] == state_name:
            state_id = state["id"]
            break
    if state_id is None:
        raise RuntimeError(f"state not found: {state_name}")

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
    print(f"posted callback and moved {issue_id} to {state_name}")
    return 0


def build_parser(default_mode="generic"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("generic", "stanage", "mimas"), default=default_mode)
    parser.add_argument("--issue-id", required=True)
    parser.add_argument("--state-name", default="Todo")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID", ""))
    parser.add_argument("--screen-session", default="")
    parser.add_argument("--exit-code", required=True)
    parser.add_argument("--status", default="")
    parser.add_argument("--log-out", default="")
    parser.add_argument("--log-err", default="")
    parser.add_argument("--log", action="append", default=[])
    parser.add_argument("--output-path", default="")
    parser.add_argument("--artifact-path", action="append", default=[])
    parser.add_argument("--checkpoint-path", action="append", default=[])
    parser.add_argument("--metadata", action="append", default=[], help="Repeatable key=value metadata.")
    parser.add_argument("--summary-text", default="")
    parser.add_argument("--summary-template", default="")
    parser.add_argument("--summary-file", default="")
    parser.add_argument("--title", default="Experiment job finished")
    parser.add_argument("--job-label", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-log-chars", type=int, default=3500)
    parser.add_argument("--max-body-chars", type=int, default=9000)
    parser.add_argument(
        "--first-excerpt-only",
        action="store_true",
        default=True,
        help="Include only the first available excerpt to keep Linear comments compact.",
    )
    parser.add_argument(
        "--include-all-excerpts",
        action="store_false",
        dest="first_excerpt_only",
        help="Include summary and all available log excerpts, still bounded by --max-body-chars.",
    )
    return parser


def main(argv=None, default_mode="generic"):
    parser = build_parser(default_mode=default_mode)
    args = parser.parse_args(argv)
    body = build_body(args)
    if args.dry_run:
        print(body)
        return 0
    return post_callback(args.issue_id, args.state_name, body)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        raise
