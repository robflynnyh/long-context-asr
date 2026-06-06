import random
import unittest
from unittest import mock

import torch

from exp.train_streaming_decoder_asr import (
    build_chunk_with_subsampling_history,
    chunk_starts_for_batch,
    final_targets_for_lengths,
    positive_output_length,
)
from lcasr.models.streaming_decoder_asr import StreamingDecoderASR
from tests.test_streaming_decoder_asr_rope import tiny_streaming_config


class StreamingDecoderOrderedChunksTest(unittest.TestCase):
    def test_chunk_starts_stay_ordered_when_shuffle_disabled(self):
        lengths = torch.tensor([21, 9])

        starts = chunk_starts_for_batch(lengths, stride=8, shuffle_chunks=False)

        self.assertEqual(starts, [0, 8, 16])

    def test_chunk_starts_use_rng_when_shuffle_enabled(self):
        lengths = torch.tensor([21])
        rng = random.Random(7)

        starts = chunk_starts_for_batch(lengths, stride=8, shuffle_chunks=True, rng=rng)

        self.assertCountEqual(starts, [0, 8, 16])
        self.assertNotEqual(starts, [0, 8, 16])

    def test_subsampling_history_prepends_previous_raw_frames(self):
        audio = torch.arange(2 * 1 * 20, dtype=torch.float32).view(2, 1, 20)
        lengths = torch.tensor([20, 13])
        active = lengths > 8

        segment, segment_lengths, current_lengths, history_len = build_chunk_with_subsampling_history(
            audio=audio,
            audio_lengths=lengths,
            active=active,
            chunk_start=8,
            chunk_size=8,
            history_frames=8,
        )

        self.assertEqual(history_len, 8)
        self.assertEqual(segment.shape, (2, 1, 16))
        self.assertEqual(segment_lengths.tolist(), [16, 13])
        self.assertEqual(current_lengths.tolist(), [8, 5])
        torch.testing.assert_close(segment[0, 0, :8], audio[0, 0, :8])
        torch.testing.assert_close(segment[0, 0, 8:], audio[0, 0, 8:16])

    def test_loss_alignment_slices_history_prefixed_subsampled_span(self):
        model = StreamingDecoderASR(**tiny_streaming_config())
        model.eval()
        audio = torch.randn(1, 8, 48)
        lengths = torch.tensor([48])
        history_len = 16
        history_output_len = positive_output_length(model, history_len)
        current_output_len = int((model.output_lengths(lengths) - history_output_len)[0].item())
        frame_targets = torch.full((1, current_output_len), model.get_silence_id(), dtype=torch.long)

        out = model.calc_loss_with_cache(
            audio_signal=audio,
            length=lengths,
            frame_targets=frame_targets,
            feature_start=history_output_len,
            return_cache_slice=(0, current_output_len),
            detach_cache=True,
        )

        self.assertEqual(out["logits"].shape[1], current_output_len)
        self.assertEqual(int(out["length"][0].item()), current_output_len)
        self.assertEqual(out["cache"][0].shape[1], current_output_len)
        self.assertFalse(out["cache"][0].requires_grad)

    def test_cached_training_path_matches_uncached_without_previous_cache(self):
        torch.manual_seed(0)
        model = StreamingDecoderASR(**tiny_streaming_config())
        model.eval()
        audio = torch.randn(1, 8, 64)
        lengths = torch.tensor([64])
        output_len = int(model.output_lengths(lengths)[0].item())
        frame_targets = torch.full((1, output_len), model.get_silence_id(), dtype=torch.long)
        frame_targets[0, 2] = 3

        uncached = model.calc_loss(audio_signal=audio, length=lengths, frame_targets=frame_targets)
        cached = model.calc_loss_with_cache(
            audio_signal=audio,
            length=lengths,
            frame_targets=frame_targets,
            return_cache_slice=(0, output_len),
            detach_cache=True,
        )

        torch.testing.assert_close(cached["logits"], uncached["logits"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(cached["loss"], uncached["loss"], atol=1e-5, rtol=1e-5)

    def test_cache_is_carried_and_trimmed_without_unbounded_rope_offsets(self):
        torch.manual_seed(0)
        model = StreamingDecoderASR(**tiny_streaming_config())
        model.train()
        lengths = torch.tensor([40])
        first = torch.randn(1, 8, 40)
        second = torch.randn(1, 8, 56)
        first_len = int(model.output_lengths(lengths)[0].item())
        history_len = 16
        history_output_len = positive_output_length(model, history_len)
        second_len = int((model.output_lengths(torch.tensor([56])) - history_output_len)[0].item())
        first_retain_len = int(model.output_lengths(torch.tensor([40]))[0].item())
        second_retain_len = second_len
        first_targets = torch.full((1, first_len), model.get_silence_id(), dtype=torch.long)
        second_targets = torch.full((1, second_len), model.get_silence_id(), dtype=torch.long)
        observed = []
        original_rotary = model._rotary_emb_fn

        def record_rotary(seq_len, device, offset=0, q_offset=0, trim_k=False):
            observed.append((seq_len, q_offset))
            return original_rotary(seq_len, device, offset=offset, q_offset=q_offset, trim_k=trim_k)

        with mock.patch.object(model, "_rotary_emb_fn", side_effect=record_rotary):
            first_out = model.calc_loss_with_cache(
                audio_signal=first,
                length=lengths,
                frame_targets=first_targets,
                return_cache_slice=(0, first_retain_len),
                detach_cache=True,
            )
            self.assertEqual(first_out["cache"][0].shape[1], first_retain_len)
            first_out["loss"].backward()
            second_out = model.calc_loss_with_cache(
                audio_signal=second,
                length=torch.tensor([56]),
                frame_targets=second_targets,
                cached_kvs=first_out["cache"],
                feature_start=history_output_len,
                return_cache_slice=(0, second_retain_len),
                initial_frame_targets=final_targets_for_lengths(
                    first_targets,
                    torch.tensor([first_retain_len]),
                    model.get_silence_id(),
                ),
                detach_cache=True,
            )

        self.assertEqual(second_out["cache"][0].shape[1], second_retain_len)
        self.assertEqual(observed[0], (first_len, 0))
        self.assertEqual(observed[1], (first_retain_len + second_len, first_retain_len))
        self.assertFalse(second_out["cache"][0].requires_grad)


if __name__ == "__main__":
    unittest.main()
