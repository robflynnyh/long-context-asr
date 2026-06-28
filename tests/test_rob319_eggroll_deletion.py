import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from lcasr.models.sconformer_xl import SCConformerXL


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "symphony" / "scripts" / "rob319_eggroll_deletion_search.py"
SPEC = importlib.util.spec_from_file_location("rob319_eggroll_deletion_search", SCRIPT_PATH)
rob319 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = rob319
SPEC.loader.exec_module(rob319)


def tiny_sconformer():
    return SCConformerXL(
        vocab_size=16,
        feat_in=8,
        n_layers=2,
        d_model=16,
        n_heads=2,
        head_dim=8,
        expansion_factor=2,
        conv_kernel_size=3,
        subsampling_conv_channels=8,
        dropout_ff=0.0,
        dropout_conv=0.0,
        dropout_attn=0.0,
        use_rotary=True,
    )


class Rob319EggrollDeletionTests(unittest.TestCase):
    def test_target_selector_only_matches_conformer_ffn_and_conv_weights(self):
        model = tiny_sconformer()
        targets = rob319.select_target_tensors(model)
        names = [target.name for target in targets]

        self.assertEqual(len(targets), 14)
        self.assertIn("layers.0.ff1.fn.fn.fc1.weight", names)
        self.assertIn("layers.0.ff2.fn.fn.fc2.weight", names)
        self.assertIn("layers.1.conv.fn.depthwise_conv.weight", names)
        self.assertFalse(any("subsampling" in name for name in names))
        self.assertFalse(any("attend" in name for name in names))
        self.assertFalse(any("decoder" in name for name in names))
        self.assertFalse(any("norm" in name.lower() for name in names))
        self.assertTrue(all(name.endswith(".weight") for name in names))

    def test_low_rank_delta_is_deterministic_by_tensor_name_and_pair(self):
        kwargs = {
            "shape": (5, 7),
            "tensor_name": "layers.0.ff1.fn.fn.fc1.weight",
            "pair_id": 3,
            "rank": 2,
            "base_seed": 319,
            "device": torch.device("cpu"),
            "dtype": torch.float32,
        }

        delta_a = rob319.low_rank_delta(**kwargs)
        delta_b = rob319.low_rank_delta(**kwargs)
        delta_c = rob319.low_rank_delta(**{**kwargs, "pair_id": 4})

        self.assertTrue(torch.equal(delta_a, delta_b))
        self.assertFalse(torch.equal(delta_a, delta_c))

    def test_antithetic_candidate_application_restores_weights(self):
        model = tiny_sconformer()
        target = rob319.select_target_tensors(model)[0]
        parameter = dict(model.named_parameters())[target.name]
        before = parameter.detach().clone()
        candidate = rob319.Candidate(candidate_id=0, pair_id=0, sign=1)

        with rob319.applied_candidate_perturbation(
            model,
            [target],
            candidate=candidate,
            rank=2,
            sigma=1e-3,
            base_seed=319,
        ):
            self.assertFalse(torch.equal(parameter, before))

        torch.testing.assert_close(parameter, before)

    def test_blocks_use_full_groups_of_five_and_hold_out_validation(self):
        records = [
            rob319.EarningsRecord(id=str(i), audio=f"{i}.mp3", text="hello", transcript_key=str(i))
            for i in range(13)
        ]

        search_blocks, validation_blocks = rob319.make_blocks(records, block_size=5, search_fraction=0.5)

        self.assertEqual([block.block_id for block in search_blocks], [0])
        self.assertEqual([block.block_id for block in validation_blocks], [1])
        self.assertEqual(search_blocks[0].recording_ids, ["0", "1", "2", "3", "4"])
        self.assertEqual(validation_blocks[0].recording_ids, ["5", "6", "7", "8", "9"])

    def test_context_score_preserves_required_diagnostics(self):
        clean = {"short": 0.30, "medium": 0.25, "long": 0.20}
        damaged = {"short": 0.36, "medium": 0.30, "long": 0.24}

        score = rob319.score_candidate(clean, damaged, {})

        for key in rob319.WANDB_REQUIRED_KEYS:
            self.assertIn(key, score)
        self.assertGreater(score["retained_gain"], 0)
        self.assertEqual(score["ordering_penalty"], 0)
        self.assertGreater(score["mean_damage"], 0)

    def test_earnings_loader_verifies_alias_transcript_mapping(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train = root / "train"
            train.mkdir()
            (train / "long-audio-name.mp3").write_bytes(b"")
            (train / "exact.mp3").write_bytes(b"")
            transcripts = {
                "long-audio": "The truncated transcript key still maps.",
                "exact": "The exact transcript key maps.",
            }
            transcript_path = root / "full_transcripts.json"
            transcript_path.write_text(__import__("json").dumps(transcripts), encoding="utf-8")

            records, metadata = rob319.load_earnings_records(
                {
                    "dataset": {
                        "root": str(root),
                        "split": "train",
                        "transcripts": str(transcript_path),
                        "expected_recordings": 2,
                    }
                }
            )

        self.assertEqual([record.id for record in records], ["exact", "long-audio-name"])
        self.assertEqual(metadata["transcript_aliases"]["long-audio-name"], "long-audio")

    def test_checkpoint_root_override_rewrites_resolved_model_paths(self):
        config = {
            "paths": {"checkpoint_root": "/mnt/original"},
            "search": {"overlap_ratio": 0.875},
            "models": [
                {
                    "label": "short",
                    "seq_len": 1024,
                    "repeat": 1,
                    "path": "/mnt/original/n_seq_sched_1024_rp_1/step_105360.pt",
                },
                {
                    "label": "medium",
                    "seq_len": 8192,
                    "repeat": 1,
                    "path": "/mnt/original/n_seq_sched_8192_rp_1/step_105360.pt",
                },
                {
                    "label": "long",
                    "seq_len": 16384,
                    "repeat": 1,
                    "path": "/mnt/original/n_seq_sched_16384_rp_1/step_105360.pt",
                },
            ],
        }

        specs = rob319.load_model_specs(config, "/store/copied")

        self.assertEqual(
            [spec.path for spec in specs],
            [
                "/store/copied/n_seq_sched_1024_rp_1/step_105360.pt",
                "/store/copied/n_seq_sched_8192_rp_1/step_105360.pt",
                "/store/copied/n_seq_sched_16384_rp_1/step_105360.pt",
            ],
        )


if __name__ == "__main__":
    unittest.main()
