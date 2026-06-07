import unittest

import torch

from lcasr.models.BestRQ import BestRQ


def _mask_selector(mask_prob=0.12, mask_length=4):
    selector = BestRQ.__new__(BestRQ)
    selector.mask_mode = "speechbrain"
    selector.mask_prob = mask_prob
    selector.mask_length = mask_length
    return selector


class TestBestRQSpeechBrainMasking(unittest.TestCase):
    def test_speechbrain_mask_uses_each_sample_length(self):
        torch.manual_seed(0)
        selector = _mask_selector(mask_prob=0.125, mask_length=4)
        valid_stacked = torch.arange(256)[None, :] < torch.tensor([[256], [128]])

        mask = selector.select_mask(B=2, T=256, valid_stacked=valid_stacked)

        self.assertEqual(mask.shape, (2, 256))
        self.assertTrue(mask[0, 128:].any())
        self.assertFalse(mask[1, 128:].any())
        self.assertEqual(int(mask[0].sum().item()), 128)
        self.assertEqual(int(mask[1].sum().item()), 64)


if __name__ == "__main__":
    unittest.main()
