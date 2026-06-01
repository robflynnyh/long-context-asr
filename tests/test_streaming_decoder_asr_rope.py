import unittest
from unittest import mock

import torch

from lcasr.models.streaming_decoder_asr import CausalDecoderLayer, StreamingDecoderASR


def tiny_streaming_config():
    return {
        "vocab_size": 16,
        "feat_in": 8,
        "n_layers": 2,
        "d_model": 32,
        "n_heads": 4,
        "expansion_factor": 2,
        "dropout_ff": 0.0,
        "dropout_attn": 0.0,
        "subsampling_factor": 4,
        "subsampling": "dw_striding",
        "subsampling_act": "silu",
        "subsampling_conv_channels": 16,
        "subsampling_norm_out": True,
        "decoder_norm": True,
        "previous_token_dropout": 0.0,
    }


class StreamingDecoderASRRoPETest(unittest.TestCase):
    def test_default_rope_forward_backward(self):
        torch.manual_seed(0)
        model = StreamingDecoderASR(**tiny_streaming_config())
        self.assertTrue(model.use_rotary)
        self.assertEqual(model.rotary_base_freq, 1_500_000)
        self.assertIsNotNone(model.rotary_pos_emb)

        audio = torch.randn(2, 8, 64)
        lengths = torch.full((2,), 64, dtype=torch.long)
        output_lengths = model.output_lengths(lengths)
        frame_targets = torch.full(
            (2, int(output_lengths.max().item())),
            model.get_silence_id(),
            dtype=torch.long,
        )
        frame_targets[0, 1] = 3
        frame_targets[1, 2] = 7

        out = model.calc_loss(audio_signal=audio, length=lengths, frame_targets=frame_targets)
        self.assertEqual(out["logits"].shape[:2], frame_targets.shape)
        self.assertEqual(out["logits"].shape[-1], model.num_classes)
        out["loss"].backward()
        self.assertIsNotNone(model.text_head.weight.grad)

    def test_no_rope_state_loads_strictly_into_default_rope_model(self):
        old_config = tiny_streaming_config()
        old_model = StreamingDecoderASR(**old_config, use_rotary=False)
        old_state = old_model.state_dict()
        self.assertFalse(any(key.startswith("rotary_pos_emb.") for key in old_state))

        model_from_old_config = StreamingDecoderASR(**old_config)
        self.assertTrue(model_from_old_config.use_rotary)
        incompatible = model_from_old_config.load_state_dict(old_state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_rope_can_be_disabled_for_ablations(self):
        model = StreamingDecoderASR(**tiny_streaming_config(), use_rotary=False)
        self.assertFalse(model.use_rotary)
        self.assertIsNone(model.rotary_pos_emb)

    def test_greedy_decode_kv_cache_matches_uncached_decode(self):
        torch.manual_seed(0)
        model = StreamingDecoderASR(**tiny_streaming_config())
        model.eval()
        audio = torch.randn(1, 8, 64)
        lengths = torch.full((1,), 64, dtype=torch.long)

        uncached = model.greedy_decode(audio_signal=audio, length=lengths, use_kv_cache=False)
        cached = model.greedy_decode(audio_signal=audio, length=lengths, use_kv_cache=True)

        self.assertTrue(torch.equal(uncached["length"], cached["length"]))
        self.assertTrue(torch.equal(uncached["predictions"], cached["predictions"]))

    def test_cached_attention_respects_max_cache_length(self):
        torch.manual_seed(0)
        layer = CausalDecoderLayer(d_model=32, n_heads=4, dropout_ff=0.0, dropout_attn=0.0)
        layer.eval()
        cache = None

        for step in range(5):
            x = torch.randn(1, 1, 32)
            _, cache = layer(x, cached_kv=cache, use_cache=True, max_cache_length=3)
            self.assertLessEqual(cache.shape[1], 3)
            self.assertEqual(cache.shape[1], min(step + 1, 3))

    def test_cached_attention_uses_torch_sdpa(self):
        torch.manual_seed(0)
        layer = CausalDecoderLayer(d_model=32, n_heads=4, dropout_ff=0.0, dropout_attn=0.0)
        layer.eval()
        x = torch.randn(1, 1, 32)

        with mock.patch(
            "lcasr.models.streaming_decoder_asr.F.scaled_dot_product_attention",
            wraps=torch.nn.functional.scaled_dot_product_attention,
        ) as sdpa:
            layer(x, use_cache=True)

        self.assertEqual(sdpa.call_count, 1)

    def test_kv_cache_spectrogram_length_uses_subsampled_frame_count(self):
        model = StreamingDecoderASR(**tiny_streaming_config())

        self.assertEqual(
            model.kv_cache_length_from_spectrogram_length(64),
            int(model.output_lengths(torch.tensor([64]))[0].item()),
        )
        self.assertLess(model.kv_cache_length_from_spectrogram_length(64), 64)

    def test_combined_logits_are_normalized_joint_distribution(self):
        model = StreamingDecoderASR(**tiny_streaming_config())
        silence_logits = torch.tensor([[[1.25, -0.5]]])
        text_logits = torch.randn(1, 1, model.vocab_size)

        combined_probs = model._combined_logits(silence_logits, text_logits).exp()

        torch.testing.assert_close(combined_probs.sum(dim=-1), torch.ones(1, 1))
        torch.testing.assert_close(
            combined_probs[..., -1],
            torch.softmax(silence_logits, dim=-1)[..., 0],
        )

    def test_sampling_draws_from_joint_text_and_silence_distribution(self):
        model = StreamingDecoderASR(**tiny_streaming_config())
        silence_logits = torch.tensor([[[0.5, 1.0]]])
        text_logits = torch.randn(1, 1, model.vocab_size)
        sampled_id = torch.full((1, 1), model.get_silence_id())
        observed_probs = []

        def fake_multinomial(probs, num_samples):
            observed_probs.append(probs.detach().clone())
            return sampled_id.reshape(-1, 1)

        with mock.patch("torch.multinomial", side_effect=fake_multinomial):
            prediction = model._predict_ids(
                silence_logits,
                text_logits,
                sample=True,
                temperature=0.3,
            )

        self.assertTrue(torch.equal(prediction, sampled_id))
        self.assertEqual(observed_probs[0].shape[-1], model.num_classes)
        torch.testing.assert_close(observed_probs[0].sum(dim=-1), torch.ones(1))

    def test_silence_sampling_uses_greedy_text_head(self):
        model = StreamingDecoderASR(**tiny_streaming_config())
        silence_logits = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
        text_logits = torch.full((1, 2, model.vocab_size), -10.0)
        text_logits[0, 0, 5] = 3.0
        text_logits[0, 1, 7] = 4.0

        def fake_multinomial(probs, num_samples):
            self.assertEqual(probs.shape[-1], 2)
            return torch.tensor([[1], [0]])

        with mock.patch("torch.multinomial", side_effect=fake_multinomial):
            prediction = model._predict_ids(
                silence_logits,
                text_logits,
                sample=True,
                temperature=0.3,
                sample_silence_only=True,
            )

        expected = torch.tensor([[5, model.get_silence_id()]])
        self.assertTrue(torch.equal(prediction, expected))


if __name__ == "__main__":
    unittest.main()
