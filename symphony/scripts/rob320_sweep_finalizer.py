#!/usr/bin/env python3
"""Summarize the ROB-320 GRU sequence sweep and call the Linear callback."""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path


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
ERROR_RE = re.compile(
    r"(Traceback|RuntimeError|Exception|CUDA out of memory|OutOfMemory|slurmstepd: error|srun: error)",
    re.IGNORECASE,
)


def read_job_manifest(path: str):
    jobs = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                jobs.append({
                    "job_id": parts[0],
                    "name": parts[1],
                    "config_path": parts[2],
                    "script_path": parts[3],
                })
    return jobs


def sacct_state(job_id: str) -> str:
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
            timeout=15,
        )
    except Exception as exc:
        return f"UNAVAILABLE ({exc})"
    if result.returncode != 0:
        return f"UNAVAILABLE ({result.stderr.strip() or 'sacct failed'})"
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= 4 and parts[0] == str(job_id):
            elapsed = parts[4] if len(parts) > 4 else "unknown"
            return f"{parts[2]} {parts[3]} elapsed={elapsed}"
    return "UNKNOWN"


def tail_text(path: str, max_chars: int = 8192) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_chars), os.SEEK_SET)
        return handle.read().decode("utf-8", errors="replace")


def log_paths(log_dir: str, name: str, job_id: str):
    return {
        "stdout": os.path.join(log_dir, f"{name}-{job_id}.out"),
        "stderr": os.path.join(log_dir, f"{name}-{job_id}.err"),
    }


def log_has_error(paths) -> bool:
    for path in paths.values():
        if ERROR_RE.search(tail_text(path)):
            return True
    return False


def checkpoint_summary(checkpoint_root: str):
    run_dirs = sorted(path for path in glob.glob(os.path.join(checkpoint_root, "*")) if os.path.isdir(path))
    rows = []
    total_checkpoints = 0
    for run_dir in run_dirs:
        checkpoints = sorted(glob.glob(os.path.join(run_dir, "step_*.pt")))
        total_checkpoints += len(checkpoints)
        rows.append((os.path.basename(run_dir), len(checkpoints), os.path.basename(checkpoints[-1]) if checkpoints else "none"))
    return rows, total_checkpoints


def write_summary(args, jobs, job_rows, checkpoint_rows, total_checkpoints, success):
    summary_dir = os.path.join(args.artifact_root, "finalizer")
    os.makedirs(summary_dir, exist_ok=True)
    finalizer_job = os.environ.get("SLURM_JOB_ID", "manual")
    summary_path = os.path.join(summary_dir, f"rob320_sweep_finalizer_{finalizer_job}.md")

    lines = [
        "# ROB-320 GRU sequence sweep finalizer",
        "",
        f"overall_status: {'success' if success else 'failure'}",
        f"sweep_jobs: {len(jobs)}",
        f"expected_jobs: {args.expected_jobs}",
        f"checkpoint_root: {args.checkpoint_root}",
        f"total_checkpoints_seen: {total_checkpoints}",
        f"generated_config_manifest: {args.jobs_file}",
        "",
        "## Slurm jobs",
        "",
        "| job_id | name | state | log_error |",
        "| --- | --- | --- | --- |",
    ]
    for row in job_rows:
        lines.append(f"| {row['job_id']} | {row['name']} | {row['state']} | {row['log_error']} |")

    lines.extend([
        "",
        "## Checkpoints",
        "",
        "| run | checkpoint_count | latest |",
        "| --- | ---: | --- |",
    ])
    for name, count, latest in checkpoint_rows:
        lines.append(f"| {name} | {count} | {latest} |")

    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return summary_path


def call_callback(args, summary_path, exit_code):
    callback = Path(args.repo_dir) / "symphony" / "scripts" / "linear_stanage_callback.py"
    cmd = [
        sys.executable,
        str(callback),
        "--issue-id",
        args.issue_id,
        "--state-name",
        args.state_name,
        "--slurm-job-id",
        os.environ.get("SLURM_JOB_ID", "manual"),
        "--exit-code",
        str(exit_code),
        "--artifact-path",
        args.artifact_root,
        "--checkpoint-path",
        args.checkpoint_root,
        "--summary-file",
        summary_path,
        "--title",
        "ROB-320 GRU sequence sweep finalizer",
        "--job-label",
        "Stanage finalizer job",
        "--metadata",
        f"job_manifest={args.jobs_file}",
        "--metadata",
        f"config_template={args.config_template}",
        "--metadata",
        f"branch={args.branch}",
        "--metadata",
        f"commit={args.commit}",
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    return subprocess.run(cmd, check=False).returncode


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue-id", default="ROB-320")
    parser.add_argument("--state-name", default="Todo")
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--jobs-file", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--config-template", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--expected-jobs", type=int, default=27)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    jobs = read_job_manifest(args.jobs_file)
    job_rows = []
    success = len(jobs) == args.expected_jobs
    for job in jobs:
        paths = log_paths(args.log_dir, job["name"], job["job_id"])
        state = sacct_state(job["job_id"])
        log_error = log_has_error(paths)
        if (
            state.startswith(FAILED_SLURM_PREFIXES)
            or not state.startswith("COMPLETED 0:0")
            or log_error
        ):
            success = False
        job_rows.append({
            "job_id": job["job_id"],
            "name": job["name"],
            "state": state,
            "log_error": str(log_error).lower(),
        })

    checkpoint_rows, total_checkpoints = checkpoint_summary(args.checkpoint_root)
    summary_path = write_summary(args, jobs, job_rows, checkpoint_rows, total_checkpoints, success)
    callback_code = call_callback(args, summary_path, 0 if success else 1)
    if args.dry_run:
        return callback_code
    if callback_code != 0:
        return callback_code
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
