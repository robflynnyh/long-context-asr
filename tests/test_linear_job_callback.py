import shutil
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = ROOT / ".tmp" / "test_linear_job_callback"


class LinearJobCallbackDryRunTest(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(TMP_ROOT, ignore_errors=True)

    def run_callback(self, script, args):
        result = subprocess.run(
            [sys.executable, str(ROOT / script), *args, "--dry-run"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def test_stanage_dry_run_includes_bounded_metadata(self):
        err_log = TMP_ROOT / "stanage.err"
        err_log.write_text("prefix\n" + ("x" * 100) + "\nTAIL\n", encoding="utf-8")

        body = self.run_callback(
            "symphony/scripts/linear_stanage_callback.py",
            [
                "--issue-id",
                "ROB-118",
                "--state-name",
                "Todo",
                "--slurm-job-id",
                "manual-smoke",
                "--exit-code",
                "1",
                "--log-err",
                str(err_log),
                "--artifact-path",
                "/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-118",
                "--checkpoint-path",
                "/mnt/parscratch/users/acp21rjf/checkpoints/ROB-118",
                "--metadata",
                "branch=symphony/rob-118-reusable-linear-callback",
                "--summary-text",
                "callback-only Stanage validation",
                "--title",
                "ROB-118 Stanage finalizer finished",
                "--max-log-chars",
                "40",
            ],
        )

        self.assertIn("- Mode: `stanage`", body)
        self.assertIn("ROB-118 Stanage finalizer finished: `failure`", body)
        self.assertIn("- Stanage Slurm job: `manual-smoke`", body)
        self.assertIn("- Artifact path: `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-118`", body)
        self.assertIn("- Checkpoint path: `/mnt/parscratch/users/acp21rjf/checkpoints/ROB-118`", body)
        self.assertIn("- branch: `symphony/rob-118-reusable-linear-callback`", body)
        self.assertIn("callback-only Stanage validation", body)

    def test_mimas_dry_run_renders_summary_template(self):
        out_log = TMP_ROOT / "mimas.out"
        summary_template = TMP_ROOT / "summary_template.txt"
        out_log.write_text("mimas log\n", encoding="utf-8")
        summary_template.write_text("mode={status} session={screen_session}", encoding="utf-8")

        body = self.run_callback(
            "symphony/scripts/linear_mimas_callback.py",
            [
                "--issue-id",
                "ROB-118",
                "--state-name",
                "Todo",
                "--screen-session",
                "rob118-callback-smoke",
                "--exit-code",
                "0",
                "--log-out",
                str(out_log),
                "--artifact-path",
                "/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-118",
                "--summary-template",
                str(summary_template),
                "--title",
                "ROB-118 Mimas screen finished",
            ],
        )

        self.assertIn("- Mode: `mimas`", body)
        self.assertIn("ROB-118 Mimas screen finished: `success`", body)
        self.assertIn("- Mimas screen session: `rob118-callback-smoke`", body)
        self.assertIn("mode=success session=rob118-callback-smoke", body)
        self.assertIn("Moving issue to `Todo`", body)


if __name__ == "__main__":
    unittest.main()
