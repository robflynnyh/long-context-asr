import unittest

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


if __name__ == "__main__":
    unittest.main()
