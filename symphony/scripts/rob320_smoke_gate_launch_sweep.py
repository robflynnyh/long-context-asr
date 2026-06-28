#!/usr/bin/env python3
"""Launch the ROB-320 sweep only after the representative GPU smoke passes."""

from __future__ import annotations

import argparse
import json
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


def tail_text(path: str, max_chars: int = 12000) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_chars), os.SEEK_SET)
        return handle.read().decode("utf-8", errors="replace")


def sacct_state(job_id: str) -> str:
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
    if result.returncode != 0:
        return f"UNAVAILABLE ({result.stderr.strip() or 'sacct failed'})"
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= 4 and parts[0] == str(job_id):
            elapsed = parts[4] if len(parts) > 4 else "unknown"
            return f"{parts[2]} {parts[3]} elapsed={elapsed}"
    return "UNKNOWN"


def read_smoke_summary(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def manifest_line_count(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def parse_launch_output(text: str):
    details = {
        "sweep_job_ids": "",
        "job_ids_file": "",
        "finalizer_sbatch": "",
        "finalizer_job_id": "",
    }
    for line in text.splitlines():
        if line.startswith("sweep_job_ids="):
            details["sweep_job_ids"] = line.split("=", 1)[1].strip()
        elif line.startswith("job_ids_file="):
            details["job_ids_file"] = line.split("=", 1)[1].strip()
        elif line.startswith("finalizer_sbatch="):
            details["finalizer_sbatch"] = line.split("=", 1)[1].strip()
        else:
            match = re.search(r"Submitted batch job (\d+)", line)
            if match:
                details["finalizer_job_id"] = match.group(1)
    return details


def smoke_passed(args):
    state = sacct_state(args.smoke_job_id)
    summary = read_smoke_summary(args.smoke_summary_json)
    stdout_tail = tail_text(args.smoke_log_out)
    stderr_tail = tail_text(args.smoke_log_err)
    log_error = bool(ERROR_RE.search(stdout_tail) or ERROR_RE.search(stderr_tail))
    passed = (
        state.startswith("COMPLETED 0:0")
        and summary.get("status") == "success"
        and summary.get("device") == "cuda"
        and summary.get("model_size") == "final"
        and not log_error
    )
    return passed, state, summary, log_error


def write_summary(args, smoke_state, smoke_summary, log_error, launch_output, launch_details, status):
    os.makedirs(args.summary_dir, exist_ok=True)
    path = os.path.join(
        args.summary_dir,
        f"rob320_smoke_gate_launch_{os.environ.get('SLURM_JOB_ID', 'manual')}.md",
    )
    lines = [
        "# ROB-320 smoke-gated sweep launcher",
        "",
        f"status: {status}",
        f"smoke_job_id: {args.smoke_job_id}",
        f"smoke_state: {smoke_state}",
        f"smoke_log_error: {str(log_error).lower()}",
        f"smoke_summary_json: {args.smoke_summary_json}",
        f"smoke_stdout: {args.smoke_log_out}",
        f"smoke_stderr: {args.smoke_log_err}",
        f"branch: {args.branch}",
        f"commit: {args.commit}",
        f"launch_log: {args.launch_log}",
        f"checkpoint_root: {args.checkpoint_root}",
        f"job_ids_file: {launch_details.get('job_ids_file', args.job_ids_file)}",
        f"sweep_finalizer_sbatch: {launch_details.get('finalizer_sbatch', '')}",
        f"sweep_finalizer_job_id: {launch_details.get('finalizer_job_id', '')}",
        "",
        "## Smoke summary",
        "```json",
        json.dumps(smoke_summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Launch output",
        "```text",
        launch_output.strip()[-8000:],
        "```",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def call_callback(args, summary_path, exit_code, title, state_name, launch_details):
    callback = Path(args.repo_dir) / "symphony" / "scripts" / "linear_stanage_callback.py"
    metadata = [
        f"branch={args.branch}",
        f"commit={args.commit}",
        f"smoke_job_id={args.smoke_job_id}",
        f"job_ids_file={launch_details.get('job_ids_file', args.job_ids_file)}",
        f"sweep_finalizer={launch_details.get('finalizer_job_id', '')}",
    ]
    cmd = [
        sys.executable,
        str(callback),
        "--issue-id",
        args.issue_id,
        "--state-name",
        state_name,
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
        title,
        "--job-label",
        "Smoke-gated sweep launcher",
    ]
    for item in metadata:
        cmd.extend(["--metadata", item])
    return subprocess.run(cmd, check=False).returncode


def run_launch(args):
    if os.path.exists(args.job_ids_file) and manifest_line_count(args.job_ids_file) > 0:
        raise RuntimeError(f"refusing to relaunch: job manifest already exists and is nonempty: {args.job_ids_file}")
    os.makedirs(os.path.dirname(args.launch_log), exist_ok=True)
    env = os.environ.copy()
    env.update({
        "REPO_DIR": args.repo_dir,
        "ARTIFACT_ROOT": args.artifact_root,
        "BRANCH": args.branch,
        "COMMIT": args.commit,
        "JOB_IDS_FILE": args.job_ids_file,
        "CHECKPOINT_ROOT": args.checkpoint_root,
    })
    command = [str(Path(args.repo_dir) / "symphony" / "scripts" / "launch_rob320_gru_sweep.sh")]
    result = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    launch_output = (
        f"$ {' '.join(command)}\n"
        f"exit_code={result.returncode}\n\n"
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}\n"
    )
    with open(args.launch_log, "w", encoding="utf-8") as handle:
        handle.write(launch_output)
    if result.returncode != 0:
        raise RuntimeError(f"sweep launch failed with exit code {result.returncode}; see {args.launch_log}")
    return launch_output, parse_launch_output(result.stdout)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue-id", default="ROB-320")
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--smoke-job-id", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--state-name-queued", default="Backlog")
    parser.add_argument("--state-name-failed", default="Todo")
    parser.add_argument("--sweep-name", default="gru_conformer_seq_sweep_3epoch")
    args = parser.parse_args(argv)

    args.smoke_log_out = os.path.join(args.artifact_root, "logs", f"gpu_smoke-{args.smoke_job_id}.out")
    args.smoke_log_err = os.path.join(args.artifact_root, "logs", f"gpu_smoke-{args.smoke_job_id}.err")
    args.smoke_summary_json = os.path.join(args.artifact_root, "cuda_final_smoke", f"summary_{args.smoke_job_id}.json")
    args.summary_dir = os.path.join(args.artifact_root, "finalizer")
    args.job_ids_file = os.path.join(args.artifact_root, f"{args.sweep_name}_job_ids.tsv")
    args.checkpoint_root = os.path.join(args.artifact_root, "checkpoints", args.sweep_name)
    args.launch_log = os.path.join(args.artifact_root, "logs", f"{args.sweep_name}_launch_{os.environ.get('SLURM_JOB_ID', 'manual')}.log")

    passed, smoke_state, smoke_summary, log_error = smoke_passed(args)
    launch_output = ""
    launch_details = {}
    if passed:
        try:
            launch_output, launch_details = run_launch(args)
            job_count = manifest_line_count(launch_details.get("job_ids_file") or args.job_ids_file)
            if job_count != 27:
                raise RuntimeError(f"expected 27 launched sweep jobs, saw {job_count}")
            summary_path = write_summary(args, smoke_state, smoke_summary, log_error, launch_output, launch_details, "queued")
            return call_callback(
                args,
                summary_path,
                0,
                "ROB-320 GRU sequence sweep queued",
                args.state_name_queued,
                launch_details,
            )
        except Exception as exc:
            launch_output = f"{launch_output}\nERROR: {exc}\n"

    summary_path = write_summary(args, smoke_state, smoke_summary, log_error, launch_output, launch_details, "blocked")
    callback_code = call_callback(
        args,
        summary_path,
        1,
        "ROB-320 GPU smoke gate blocked sweep launch",
        args.state_name_failed,
        launch_details,
    )
    return callback_code if callback_code != 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
