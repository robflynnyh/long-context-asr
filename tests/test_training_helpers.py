import unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import json
import os

import torch
from omegaconf import OmegaConf

from lcasr.utils.training import batch_to_chunks, build_training_dataloader, load_training_pairs


class FakeTokenizer:
    def encode(self, text):
        return [ord(char) % 31 + 1 for char in text]


class TrainingHelperTests(unittest.TestCase):
    def test_recording_manifest_batch_to_chunks_preserves_chunking_path(self):
        audio = torch.arange(1 * 2 * 8, dtype=torch.float32).reshape(1, 2, 8)
        audio_lengths = torch.LongTensor([8])
        text = [[
            {'startTime': '0.00s', 'endTime': '0.02s', 'word': 'a'},
            {'startTime': '0.02s', 'endTime': '0.04s', 'word': 'b'},
            {'startTime': '0.06s', 'endTime': '0.08s', 'word': 'c'},
        ]]

        chunks, ids, cur_batch_size = batch_to_chunks(
            batch=(audio, audio_lengths, text, ('recording-1',)),
            tokenizer=FakeTokenizer(),
            pad_id=0,
            chunk_size=4,
            chunk_overlap=0,
        )

        self.assertEqual(ids, ('recording-1',))
        self.assertEqual(cur_batch_size, 1)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(torch.equal(chunks[0]['selection_mask'], torch.BoolTensor([True])))
        self.assertEqual(chunks[0]['audio'].shape, (1, 2, 4))
        self.assertEqual(chunks[1]['audio'].shape, (1, 2, 4))
        self.assertGreater(chunks[0]['txt_lengths'].item(), 0)
        self.assertGreater(chunks[1]['txt_lengths'].item(), 0)

    def test_recording_manifest_dataloader_smoke(self):
        with TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, 'audio.pt')
            text_path = os.path.join(tmpdir, 'text.json')
            manifest_path = os.path.join(tmpdir, 'manifest.json')
            torch.save(torch.arange(1 * 2 * 8, dtype=torch.float32).reshape(1, 2, 8), audio_path)
            with open(text_path, 'w', encoding='utf-8') as handle:
                json.dump({
                    'results': [{
                        'alternatives': [{
                            'words': [
                                {'startTime': '0.00s', 'endTime': '0.02s', 'word': 'a'},
                                {'startTime': '0.02s', 'endTime': '0.04s', 'word': 'b'},
                                {'startTime': '0.06s', 'endTime': '0.08s', 'word': 'c'},
                            ],
                        }],
                    }],
                }, handle)
            with open(manifest_path, 'w', encoding='utf-8') as handle:
                json.dump({'recording-1': {'audio': audio_path, 'txt': text_path, 'duration': 1.0}}, handle)

            config = OmegaConf.create({
                'data': {'path': manifest_path, 'format': 'recording_manifest'},
                'training': {'batch_size': 1},
                'audio_chunking': {'size': 4, 'overlap': 0},
            })
            args = SimpleNamespace(config=config, num_workers=0, pin_memory=False, prefetch_factor=2)
            pairs = load_training_pairs(config)
            dataloader = build_training_dataloader(
                args=args,
                tokenizer=FakeTokenizer(),
                seen_ids=[],
                random_seed=1234,
                paired_data=pairs,
            )

            batch = next(iter(dataloader))
            chunks, ids, cur_batch_size = batch_to_chunks(
                batch=batch,
                tokenizer=FakeTokenizer(),
                pad_id=0,
                chunk_size=4,
                chunk_overlap=0,
            )

            self.assertEqual(ids, ('recording-1',))
            self.assertEqual(cur_batch_size, 1)
            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0]['audio'].shape, (1, 2, 4))

    def test_utterance_folder_batch_to_chunks_returns_single_presegmented_chunk(self):
        batch = {
            'ids': ('utt-1', 'utt-2'),
            'audio': torch.zeros(2, 2, 5),
            'audio_lengths': torch.LongTensor([5, 4]),
            'text': torch.LongTensor([[1, 2, 0], [3, 4, 5]]),
            'text_lengths': torch.LongTensor([2, 3]),
        }

        chunks, ids, cur_batch_size = batch_to_chunks(
            batch=batch,
            tokenizer=FakeTokenizer(),
            pad_id=0,
            chunk_size=4,
            chunk_overlap=0,
        )

        self.assertEqual(ids, ['utt-1', 'utt-2'])
        self.assertEqual(cur_batch_size, 2)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(torch.equal(chunks[0]['selection_mask'], torch.BoolTensor([True, True])))


if __name__ == '__main__':
    unittest.main()
