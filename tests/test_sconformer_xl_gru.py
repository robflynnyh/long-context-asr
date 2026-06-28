import unittest

import torch

from lcasr.models.sconformer_xl import ConformerGRUModule, SCConformerXL


def tiny_sconformer_config(**overrides):
    config = {
        "vocab_size": 16,
        "feat_in": 8,
        "subsampling": "stacking",
        "subsampling_factor": 2,
        "subsampling_conv_channels": 8,
        "n_layers": 1,
        "d_model": 16,
        "n_heads": 2,
        "head_dim": 8,
        "expansion_factor": 2,
        "dropout_ff": 0.0,
        "dropout_conv": 0.0,
        "dropout_attn": 0.0,
        "conv_kernel_size": 3,
        "conv_expansion_factor": 1,
        "decoder_norm": True,
        "use_rotary": False,
        "self_conditioning": False,
        "default_norm": "layer_norm",
        "bias_in_ff": False,
    }
    config.update(overrides)
    return config


class SCConformerXLGRUTest(unittest.TestCase):
    def test_default_model_has_no_gru_branch(self):
        model = SCConformerXL(**tiny_sconformer_config())

        self.assertFalse(model.layers[0].gru_module)
        self.assertFalse(hasattr(model.layers[0], "gru"))

    def test_gru_model_forward_backward_is_finite(self):
        torch.manual_seed(0)
        model = SCConformerXL(
            **tiny_sconformer_config(
                gru_module=True,
                gru_hidden_size=12,
                gru_num_layers=1,
                gru_dropout=0.0,
                gru_bidirectional=True,
            )
        )

        audio = torch.randn(2, 8, 16)
        lengths = torch.LongTensor([16, 14])
        out = model(audio_signal=audio, length=lengths, return_logits=True)

        self.assertEqual(out["final_posteriors"].shape[:2], (2, 8))
        self.assertTrue(torch.isfinite(out["final_posteriors"]).all())

        loss = -out["final_posteriors"][:, :, 0].mean()
        loss.backward()

        gru_grads = [
            param.grad
            for name, param in model.named_parameters()
            if ".gru.fn.gru." in name
        ]
        self.assertTrue(gru_grads)
        self.assertTrue(any(grad is not None and torch.isfinite(grad).all() for grad in gru_grads))

    def test_gru_output_is_zero_on_padded_positions(self):
        torch.manual_seed(0)
        module = ConformerGRUModule(d_model=4, hidden_size=6, bidirectional=True)
        x = torch.randn(2, 5, 4)
        lengths = torch.LongTensor([5, 3])
        pad_mask = torch.arange(5).expand(2, 5) >= lengths.unsqueeze(1)

        out = module(x, length=lengths, pad_mask=pad_mask)

        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.equal(out[pad_mask], torch.zeros_like(out[pad_mask])))

    def test_gru_params_are_classified_for_weight_decay_groups(self):
        model = SCConformerXL(**tiny_sconformer_config(gru_module=True))

        groups = model.get_param_groups({"weight_decay": 0.1})

        self.assertEqual(len(groups), 2)
        grouped_params = sum(len(group["params"]) for group in groups)
        self.assertEqual(grouped_params, sum(1 for _ in model.parameters()))


if __name__ == "__main__":
    unittest.main()
