import unittest

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None

from lstm_fusion import LSTMFusionModel


@unittest.skipUnless(torch is not None, "PyTorch is optional in the current environment")
class LSTMFusionTests(unittest.TestCase):
    def test_fuses_sequences_and_returns_multilabel_logits(self):
        model = LSTMFusionModel({"audio": 3, "visual_pose": 4, "interaction": 2}, 5)
        sequences = torch.zeros((2, 6, model.input_size))
        logits = model(sequences, lengths=[6, 4])
        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertEqual(tuple(model.probabilities(sequences).shape), (2, 5))

    def test_rejects_incorrect_sequence_width(self):
        model = LSTMFusionModel({"audio": 3, "visual": 4}, 2)
        with self.assertRaisesRegex(ValueError, "invalid_lstm_sequence_shape"):
            model(torch.zeros((1, 4, model.input_size - 1)))


if __name__ == "__main__":
    unittest.main()
