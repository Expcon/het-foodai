"""Contract guards run without the MNN SDK or a checkpoint."""
import unittest
from tools.edge_runtime.mnn_cpu import validate_contract, validate_input, validate_output


class EdgeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.contract = dict(task="classify", input_layout="NCHW", input_dtype="float32", input_shape=[1, 3, 8, 8], output_shape=[1, 2], num_classes=2, output_kind="classification-probabilities")

    def test_ambiguous_or_wrong_model_contract_rejected(self):
        for change in ({"task": "UNKNOWN"}, {"input_shape": [1, 8, 8, 3]}, {"input_layout": "NHWC"}, {"output_kind": "some-other-yolo"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_contract({**self.contract, **change})

    def test_wrong_shape_dtype_and_nonfinite_input_rejected(self):
        import numpy as np
        valid = np.zeros((1, 3, 8, 8), dtype=np.float32)  # synthetic, contract smoke only
        self.assertEqual(validate_input(valid, self.contract).shape, valid.shape)
        for invalid in (valid.astype(np.float64), valid.transpose(0, 2, 3, 1), valid + np.nan, valid - 1):
            with self.assertRaises(ValueError):
                validate_input(invalid, self.contract)

    def test_probabilities_cannot_silently_be_logits(self):
        import numpy as np
        validate_output(np.array([[0.2, 0.8]], dtype=np.float32), self.contract)
        with self.assertRaises(ValueError):
            validate_output(np.array([[2.0, 4.0]], dtype=np.float32), self.contract)

    def test_detection_contract_is_model_specific(self):
        import numpy as np
        contract = {**self.contract, "task": "detect", "num_classes": 80, "output_kind": "yolo11-detect-raw", "output_shape": [1, 84, 21]}
        validate_output(np.zeros((1, 84, 21), dtype=np.float32), contract)
        with self.assertRaises(ValueError):
            validate_output(np.zeros((1, 21, 84), dtype=np.float32), contract)


if __name__ == "__main__":
    unittest.main()
