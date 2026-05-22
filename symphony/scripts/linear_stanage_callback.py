#!/usr/bin/env python3
"""Thin Stanage wrapper for the reusable Linear job callback."""

import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from linear_job_callback import main  # noqa: E402


if __name__ == "__main__":
    argv = list(sys.argv[1:])
    if "--slurm-job-id" not in argv and os.environ.get("SLURM_JOB_ID"):
        argv.extend(["--slurm-job-id", os.environ["SLURM_JOB_ID"]])
    raise SystemExit(main(argv, default_mode="stanage"))
