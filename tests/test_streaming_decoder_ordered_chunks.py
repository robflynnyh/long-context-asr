import random
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from exp.train_streaming_decoder_asr import (
    add_final_flush_padding,
    build_chunk_with_subsampling_history,
    chunk_starts_for_batch,
    final_targets_for_lengths,
    positive_output_length,
    select_kv_caches_for_active,
    train,
)
from lcasr.utils.dataloading import SimpleDataset
from lcasr.models.streaming_decoder_asr import StreamingDecoderASR
from lcasr.utils.streaming_targets import (
    build_streaming_frame_targets,
    build_streaming_frame_targets_from_events,
    build_streaming_target_events,
)
from tests.test_streaming_decoder_asr_rope import tiny_streaming_config


class ToyTokenizer:
    def __init__(self):
        self.tokens = {"cross": [5], "pair": [6, 7]}

    def vocab_size(self):
        return 16

    def encode(self, surface):
        return self.tokens[surface]


class OneBatchLoader:
    def __init__(self, batch, tokenizer):
        self.batch = batch
        self.tokenizer = tokenizer

    def __iter__(self):
        yield self.batch


class NoopScheduler:
    is_warmup = False

    def step(self, epoch=None):
        pass

    def get_last_lr(self):
        return [0.0]


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

    def test_final_flush_padding_only_extends_final_chunk_lengths(self):
        chunk = torch.ones(2, 1, 8)
        lengths = torch.tensor([8, 5])

        padded, padded_lengths = add_final_flush_padding(
            chunk=chunk,
            chunk_lengths=lengths,
            final_chunks=torch.tensor([False, True]),
            flush_frames=4,
        )

        self.assertEqual(padded.shape, (2, 1, 12))
        self.assertEqual(padded_lengths.tolist(), [8, 9])
        torch.testing.assert_close(padded[:, :, :8], chunk)
        torch.testing.assert_close(padded[:, :, 8:], torch.zeros(2, 1, 4))

    def test_delayed_boundary_crossing_target_is_owned_by_next_chunk(self):
        tokenizer = ToyTokenizer()
        transcript = [[{"start": 0.10, "end": 0.12, "text": "cross"}]]
        silence_id = tokenizer.vocab_size()

        first_chunk_targets = build_streaming_frame_targets(
            transcripts=transcript,
            output_lengths=torch.tensor([4]),
            tokenizer=tokenizer,
            subsampling_factor=4,
            delay_seconds=0.08,
            chunk_start_frames=torch.tensor([0]),
            silence_id=silence_id,
        )
        second_chunk_targets = build_streaming_frame_targets(
            transcripts=transcript,
            output_lengths=torch.tensor([4]),
            tokenizer=tokenizer,
            subsampling_factor=4,
            delay_seconds=0.08,
            chunk_start_frames=torch.tensor([16]),
            silence_id=silence_id,
        )

        self.assertTrue(torch.equal(first_chunk_targets, torch.full((1, 4), silence_id)))
        self.assertEqual(second_chunk_targets.tolist(), [[silence_id, 5, silence_id, silence_id]])

    def test_real_subsampling_mapping_moves_boundary_delay_to_next_chunk(self):
        model = StreamingDecoderASR(**tiny_streaming_config())
        tokenizer = ToyTokenizer()
        transcript = [[{"start": 0.07, "end": 0.08, "text": "cross"}]]
        silence_id = tokenizer.vocab_size()
        chunk_size = 16
        delay_seconds = 0.08
        first_len = positive_output_length(model, chunk_size)
        history_output_len = positive_output_length(model, chunk_size)
        second_len = positive_output_length(model, chunk_size * 2) - history_output_len

        floor_division_targets = build_streaming_frame_targets(
            transcripts=transcript,
            output_lengths=torch.tensor([first_len]),
            tokenizer=tokenizer,
            subsampling_factor=model.subsampling_factor,
            delay_seconds=delay_seconds,
            chunk_start_frames=torch.tensor([0]),
            silence_id=silence_id,
        )
        first_chunk_targets = build_streaming_frame_targets(
            transcripts=transcript,
            output_lengths=torch.tensor([first_len]),
            tokenizer=tokenizer,
            subsampling_factor=model.subsampling_factor,
            output_length_fn=model.output_lengths,
            delay_seconds=delay_seconds,
            chunk_start_frames=torch.tensor([0]),
            silence_id=silence_id,
        )
        second_chunk_targets = build_streaming_frame_targets(
            transcripts=transcript,
            output_lengths=torch.tensor([second_len]),
            tokenizer=tokenizer,
            subsampling_factor=model.subsampling_factor,
            output_length_fn=model.output_lengths,
            delay_seconds=delay_seconds,
            chunk_start_frames=torch.tensor([chunk_size]),
            silence_id=silence_id,
        )
        expected_second_pos = positive_output_length(model, chunk_size) - history_output_len

        self.assertEqual(floor_division_targets[0, first_len - 1].item(), 5)
        self.assertTrue(torch.equal(first_chunk_targets, torch.full((1, first_len), silence_id)))
        self.assertEqual(expected_second_pos, 0)
        self.assertEqual(second_chunk_targets[0, expected_second_pos].item(), 5)

    def test_continuous_delayed_timeline_spills_tokens_across_chunks(self):
        tokenizer = ToyTokenizer()
        transcript = [[{"start": 0.02, "end": 0.04, "text": "pair"}]]
        silence_id = tokenizer.vocab_size()

        first_chunk_targets = build_streaming_frame_targets(
            transcripts=transcript,
            output_lengths=torch.tensor([4]),
            tokenizer=tokenizer,
            subsampling_factor=4,
            delay_seconds=0.08,
            chunk_start_frames=torch.tensor([0]),
            silence_id=silence_id,
        )
        second_chunk_targets = build_streaming_frame_targets(
            transcripts=transcript,
            output_lengths=torch.tensor([4]),
            tokenizer=tokenizer,
            subsampling_factor=4,
            delay_seconds=0.08,
            chunk_start_frames=torch.tensor([16]),
            silence_id=silence_id,
        )

        self.assertEqual(first_chunk_targets.tolist(), [[silence_id, silence_id, silence_id, 6]])
        self.assertEqual(second_chunk_targets.tolist(), [[7, silence_id, silence_id, silence_id]])

    def test_precomputed_target_events_match_delayed_timeline_chunks(self):
        tokenizer = ToyTokenizer()
        transcript = [[{"start": 0.02, "end": 0.04, "text": "pair"}]]
        silence_id = tokenizer.vocab_size()
        events = [
            build_streaming_target_events(
                transcript[0],
                tokenizer=tokenizer,
                subsampling_factor=4,
                delay_seconds=0.08,
            )
        ]

        first_expected = build_streaming_frame_targets(
            transcripts=transcript,
            output_lengths=torch.tensor([4]),
            tokenizer=tokenizer,
            subsampling_factor=4,
            delay_seconds=0.08,
            chunk_start_frames=torch.tensor([0]),
            silence_id=silence_id,
        )
        second_expected = build_streaming_frame_targets(
            transcripts=transcript,
            output_lengths=torch.tensor([4]),
            tokenizer=tokenizer,
            subsampling_factor=4,
            delay_seconds=0.08,
            chunk_start_frames=torch.tensor([16]),
            silence_id=silence_id,
        )

        first_targets, offsets = build_streaming_frame_targets_from_events(
            event_sequences=events,
            output_lengths=torch.tensor([4]),
            chunk_start_frames=torch.tensor([0]),
            tokenizer=tokenizer,
            subsampling_factor=4,
            delay_seconds=0.08,
            silence_id=silence_id,
            return_event_offsets=True,
        )
        second_targets = build_streaming_frame_targets_from_events(
            event_sequences=events,
            output_lengths=torch.tensor([4]),
            chunk_start_frames=torch.tensor([16]),
            tokenizer=tokenizer,
            subsampling_factor=4,
            delay_seconds=0.08,
            silence_id=silence_id,
            event_offsets=offsets,
        )

        torch.testing.assert_close(first_targets, first_expected)
        torch.testing.assert_close(second_targets, second_expected)

    def test_real_subsampling_mapping_places_final_word_in_flush_region(self):
        model = StreamingDecoderASR(**tiny_streaming_config())
        tokenizer = ToyTokenizer()
        transcript = [[{"start": 0.15, "end": 0.16, "text": "cross"}]]
        silence_id = tokenizer.vocab_size()
        chunk_start = 16
        current_raw_frames = 8
        final_flush_frames = 8
        delay_seconds = 0.08
        history_output_len = positive_output_length(model, chunk_start)
        raw_output_len = positive_output_length(model, chunk_start + current_raw_frames) - history_output_len
        flush_output_len = (
            positive_output_length(model, chunk_start + current_raw_frames + final_flush_frames) - history_output_len
        )

        raw_targets = build_streaming_frame_targets(
            transcripts=transcript,
            output_lengths=torch.tensor([raw_output_len]),
            tokenizer=tokenizer,
            subsampling_factor=model.subsampling_factor,
            output_length_fn=model.output_lengths,
            delay_seconds=delay_seconds,
            chunk_start_frames=torch.tensor([chunk_start]),
            silence_id=silence_id,
        )
        flush_targets = build_streaming_frame_targets(
            transcripts=transcript,
            output_lengths=torch.tensor([flush_output_len]),
            tokenizer=tokenizer,
            subsampling_factor=model.subsampling_factor,
            output_length_fn=model.output_lengths,
            delay_seconds=delay_seconds,
            chunk_start_frames=torch.tensor([chunk_start]),
            silence_id=silence_id,
        )

        self.assertTrue(torch.equal(raw_targets, torch.full((1, raw_output_len), silence_id)))
        self.assertGreater(flush_output_len, raw_output_len)
        self.assertEqual(flush_targets[0, raw_output_len].item(), 5)

    def test_subgroup_shuffle_size_equal_batch_size_keeps_duration_tight_batches(self):
        pairs = {
            f"sample-{idx}": {"duration": duration, "audio": "unused.pt", "txt": "unused.json"}
            for idx, duration in enumerate([1.0, 2.0, 100.0, 101.0, 200.0, 201.0])
        }

        dataset = SimpleDataset(
            pairs,
            batch_size=2,
            subgroup_shuffle_size=2,
            random_seed=3,
        )

        durations = [float(value) for value in dataset.pairs["duration"].tolist()]
        for start in range(0, len(durations), 2):
            batch_durations = durations[start : start + 2]
            self.assertLessEqual(max(batch_durations) - min(batch_durations), 1.0)

    def test_cache_selection_keeps_active_recording_slots(self):
        cache = torch.tensor([0.0, 2.0, 4.0]).view(3, 1, 1, 1, 1)

        selected = select_kv_caches_for_active(
            caches=[cache],
            cached_batch_indices=torch.tensor([0, 2, 4]),
            active_indices=torch.tensor([2, 4]),
        )

        self.assertEqual(selected[0][:, 0, 0, 0, 0].tolist(), [2.0, 4.0])

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

    def test_cached_loss_pads_previous_targets_for_ignored_tail_features(self):
        model = StreamingDecoderASR(**tiny_streaming_config())
        model.eval()
        audio = torch.randn(2, 8, 40)
        lengths = torch.tensor([32, 37])
        history_output_len = positive_output_length(model, 16)
        target_len = int((model.output_lengths(lengths) - history_output_len).max().item())
        physical_feature_len = positive_output_length(model, 40) - history_output_len
        frame_targets = torch.full((2, target_len), model.get_silence_id(), dtype=torch.long)

        self.assertGreater(physical_feature_len, target_len)

        out = model.calc_loss_with_cache(
            audio_signal=audio,
            length=lengths,
            frame_targets=frame_targets,
            feature_start=history_output_len,
            return_cache_slice=(0, target_len),
            detach_cache=True,
        )

        self.assertEqual(out["logits"].shape[1], physical_feature_len)
        self.assertEqual(out["length"].tolist(), [4, 5])
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_ordered_training_caps_feature_span_for_mixed_final_flush_batch(self):
        torch.manual_seed(0)
        model = StreamingDecoderASR(**tiny_streaming_config())
        tokenizer = ToyTokenizer()
        audio = torch.randn(2, 8, 48)
        audio_lengths = torch.tensor([48, 29])
        transcripts = [[], []]
        ids = ["continues", "final"]
        dataloader = OneBatchLoader((audio, audio_lengths, transcripts, ids), tokenizer)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        args = SimpleNamespace(
            config={
                "training": {
                    "dtype": "bfloat16",
                    "clip_value": 0.0,
                    "max_epochs": 1,
                    "max_steps": 2,
                    "backprop_every": 1,
                    "shuffle_chunks": False,
                    "scheduler_total_steps": 2,
                    "ordered_chunk_training": {
                        "enabled": True,
                        "subsampling_history_frames": 16,
                        "decoder_history_frames": 16,
                        "detach_cache": True,
                    },
                    "debug_generation": {"enabled": False},
                },
                "streaming": {"delay_seconds": 0.08, "buffer_seconds": 0.0},
                "audio_chunking": {"size": 16, "overlap": 0},
                "checkpointing": {"save_every_n_steps": 0, "dir": ".tmp/rob209-test-checkpoints"},
                "wandb": {"use": False},
            }
        )

        expected_history_len = positive_output_length(model, 16)
        expected_feature_length = int(
            (
                model.output_lengths(torch.tensor([32, 37]))
                - expected_history_len
            ).max().item()
        )
        physical_feature_length = positive_output_length(model, 40) - expected_history_len
        self.assertGreater(physical_feature_length, expected_feature_length)

        with mock.patch("torch.cuda.is_available", return_value=False), mock.patch(
            "exp.train_streaming_decoder_asr.save_model"
        ), mock.patch.object(
            model,
            "calc_loss_with_cache",
            wraps=model.calc_loss_with_cache,
        ) as cached_loss:
            train(args, model, dataloader, optimizer, NoopScheduler(), torch.device("cpu"))

        feature_lengths = [call.kwargs.get("feature_length") for call in cached_loss.call_args_list]
        self.assertIn(expected_feature_length, feature_lengths)

    def test_ordered_training_logs_debug_generation_when_record_threshold_crosses(self):
        torch.manual_seed(0)
        model = StreamingDecoderASR(**tiny_streaming_config())
        tokenizer = ToyTokenizer()
        audio = torch.randn(1, 8, 32)
        audio_lengths = torch.tensor([32])
        transcripts = [[{"start": 0.02, "end": 0.04, "text": "cross"}]]
        ids = ["recording-0"]
        dataloader = OneBatchLoader((audio, audio_lengths, transcripts, ids), tokenizer)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        args = SimpleNamespace(
            config={
                "training": {
                    "dtype": "bfloat16",
                    "clip_value": 0.0,
                    "max_epochs": 1,
                    "max_steps": 2,
                    "backprop_every": 1,
                    "shuffle_chunks": False,
                    "scheduler_total_steps": 2,
                    "ordered_chunk_training": {
                        "enabled": True,
                        "subsampling_history_frames": 16,
                        "decoder_history_frames": 16,
                        "detach_cache": True,
                    },
                    "debug_generation": {"enabled": True, "every_records": 1, "max_frames": 4},
                },
                "streaming": {"delay_seconds": 0.08, "buffer_seconds": 0.0},
                "audio_chunking": {"size": 16, "overlap": 0},
                "checkpointing": {"save_every_n_steps": 0, "dir": ".tmp/rob209-test-checkpoints"},
                "wandb": {"use": False},
            }
        )

        with mock.patch("torch.cuda.is_available", return_value=False), mock.patch(
            "exp.train_streaming_decoder_asr.save_model"
        ), mock.patch("exp.train_streaming_decoder_asr.maybe_log_debug_generation") as debug_generation:
            train(args, model, dataloader, optimizer, NoopScheduler(), torch.device("cpu"))

        self.assertEqual(debug_generation.call_count, 1)
        call_kwargs = debug_generation.call_args.kwargs
        self.assertEqual(call_kwargs["ids"], ids)
        self.assertEqual(call_kwargs["records_seen"], 1)
        self.assertEqual(call_kwargs["global_step"], 0)
        self.assertEqual(tuple(call_kwargs["chunk"].shape), (1, 8, 16))
        self.assertEqual(call_kwargs["chunk_lengths"].tolist(), [16])
        self.assertEqual(call_kwargs["chunk_transcripts"], transcripts)

    def test_resume_rebuilds_full_dataloader_after_partial_epoch(self):
        torch.manual_seed(0)
        model = StreamingDecoderASR(**tiny_streaming_config())
        tokenizer = ToyTokenizer()
        tail_audio = torch.randn(1, 8, 16)
        full_audio = torch.randn(2, 8, 16)
        tail_loader = OneBatchLoader((tail_audio, torch.tensor([16]), [[]], ["tail"]), tokenizer)
        full_loader = OneBatchLoader((full_audio, torch.tensor([16, 16]), [[], []], ["head", "tail"]), tokenizer)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        args = SimpleNamespace(
            config={
                "training": {
                    "dtype": "bfloat16",
                    "clip_value": 0.0,
                    "max_epochs": 2,
                    "max_steps": 10,
                    "backprop_every": 1,
                    "shuffle_chunks": False,
                    "scheduler_total_steps": 2,
                    "ordered_chunk_training": {"enabled": False},
                    "debug_generation": {"enabled": False},
                },
                "streaming": {"delay_seconds": 0.08, "buffer_seconds": 0.0},
                "audio_chunking": {"size": 16, "overlap": 0},
                "checkpointing": {"save_every_n_steps": 0, "dir": ".tmp/rob209-test-checkpoints"},
                "wandb": {"use": False},
            }
        )
        factory_seen_ids = []

        def dataloader_factory(seen_ids):
            factory_seen_ids.append(list(seen_ids))
            return full_loader

        with mock.patch("torch.cuda.is_available", return_value=False), mock.patch(
            "exp.train_streaming_decoder_asr.save_model"
        ) as save_model_mock:
            train(
                args,
                model,
                tail_loader,
                optimizer,
                NoopScheduler(),
                torch.device("cpu"),
                seen_ids=["head"],
                epoch=0,
                dataloader_factory=dataloader_factory,
            )

        self.assertEqual(factory_seen_ids, [["epoch_0_head", "epoch_0_tail"]])
        final_seen_ids = save_model_mock.call_args.kwargs["seen_ids"]
        self.assertIn("epoch_1_head", final_seen_ids)
        self.assertIn("epoch_1_tail", final_seen_ids)

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

    def test_chunked_decode_uses_subsampling_history_windows(self):
        torch.manual_seed(0)
        model = StreamingDecoderASR(**tiny_streaming_config())
        model.eval()
        audio = torch.randn(1, 8, 48)
        lengths = torch.tensor([48])

        with mock.patch.object(model.subsampling, "forward", wraps=model.subsampling.forward) as subsampling:
            model.greedy_decode(
                audio_signal=audio,
                length=lengths,
                use_kv_cache=True,
                chunked_kv_cache=True,
                kv_cache_chunk_spectrogram_length=16,
                subsampling_history_spectrogram_length=16,
                decoder_history_spectrogram_length=16,
            )

        input_lengths = [call.args[0].size(1) for call in subsampling.call_args_list]
        self.assertEqual(input_lengths, [16, 32, 32])

    def test_chunked_decode_adds_flush_only_to_final_chunk(self):
        torch.manual_seed(0)
        model = StreamingDecoderASR(**tiny_streaming_config())
        model.eval()
        audio = torch.randn(1, 8, 32)
        lengths = torch.tensor([32])

        with mock.patch.object(model.subsampling, "forward", wraps=model.subsampling.forward) as subsampling:
            no_flush = model.greedy_decode(
                audio_signal=audio,
                length=lengths,
                use_kv_cache=True,
                chunked_kv_cache=True,
                kv_cache_chunk_spectrogram_length=16,
                subsampling_history_spectrogram_length=16,
                decoder_history_spectrogram_length=16,
            )
        no_flush_input_lengths = [call.args[0].size(1) for call in subsampling.call_args_list]

        with mock.patch.object(model.subsampling, "forward", wraps=model.subsampling.forward) as subsampling:
            with_flush = model.greedy_decode(
                audio_signal=audio,
                length=lengths,
                use_kv_cache=True,
                chunked_kv_cache=True,
                kv_cache_chunk_spectrogram_length=16,
                subsampling_history_spectrogram_length=16,
                decoder_history_spectrogram_length=16,
                final_flush_spectrogram_length=8,
            )
        flush_input_lengths = [call.args[0].size(1) for call in subsampling.call_args_list]

        self.assertEqual(no_flush_input_lengths, [16, 32])
        self.assertEqual(flush_input_lengths, [16, 40])
        self.assertGreater(int(with_flush["length"][0].item()), int(no_flush["length"][0].item()))

    def test_chunked_decode_matches_cached_decode_when_audio_fits_one_chunk(self):
        torch.manual_seed(0)
        model = StreamingDecoderASR(**tiny_streaming_config())
        model.eval()
        audio = torch.randn(1, 8, 48)
        lengths = torch.tensor([48])

        cached = model.greedy_decode(audio_signal=audio, length=lengths, use_kv_cache=True)
        chunked = model.greedy_decode(
            audio_signal=audio,
            length=lengths,
            use_kv_cache=True,
            chunked_kv_cache=True,
            kv_cache_chunk_spectrogram_length=96,
            subsampling_history_spectrogram_length=96,
            decoder_history_spectrogram_length=96,
        )

        self.assertTrue(torch.equal(cached["length"], chunked["length"]))
        self.assertTrue(torch.equal(cached["predictions"], chunked["predictions"]))

    def test_chunked_decode_resets_rope_positions_at_chunk_boundaries(self):
        torch.manual_seed(0)
        model = StreamingDecoderASR(**tiny_streaming_config())
        model.eval()
        audio = torch.randn(1, 8, 64)
        lengths = torch.tensor([64])
        chunk_size = 16
        observed_q_offsets = []
        original_rotary = model._rotary_emb_fn

        def record_rotary(seq_len, device, offset=0, q_offset=0, trim_k=False):
            observed_q_offsets.append(q_offset)
            return original_rotary(seq_len, device, offset=offset, q_offset=q_offset, trim_k=trim_k)

        expected_q_offsets = []
        retained_len = 0
        cap = positive_output_length(model, chunk_size)
        for chunk_start in range(0, int(lengths[0].item()), chunk_size):
            history_start = max(0, chunk_start - chunk_size)
            history_len = chunk_start - history_start
            segment_len = min(int(lengths[0].item()), chunk_start + chunk_size) - history_start
            current_len = positive_output_length(model, segment_len) - positive_output_length(model, history_len)
            expected_q_offsets.extend(range(retained_len, retained_len + current_len))
            retained_len = min(current_len, cap)

        with mock.patch.object(model, "_rotary_emb_fn", side_effect=record_rotary):
            decoded = model.greedy_decode(
                audio_signal=audio,
                length=lengths,
                use_kv_cache=True,
                chunked_kv_cache=True,
                kv_cache_chunk_spectrogram_length=chunk_size,
                subsampling_history_spectrogram_length=chunk_size,
                decoder_history_spectrogram_length=chunk_size,
            )

        self.assertEqual(int(decoded["length"][0].item()), len(expected_q_offsets))
        self.assertEqual(observed_q_offsets, expected_q_offsets)


if __name__ == "__main__":
    unittest.main()
