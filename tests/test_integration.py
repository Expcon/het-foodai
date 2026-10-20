"""Temporary SYNTHETIC checkpoints: interoperability, never recognition accuracy."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foodai.contracts import class_list_hash, load_classes
from foodai.predictor import PredictionAdapter, as_contract_dict

try:
    import torch
    from PIL import Image
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Project GPU environment required")
class TestIntegratedPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from foodai.vision import get_model
        cls.temp = tempfile.TemporaryDirectory(prefix="SYNTHETIC-integration-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.classes = cls.root / "synthetic-classes.json"
        cls.classes.write_text(json.dumps({"status": "frozen", "version": "SYNTHETIC-ONLY",
            "classes": [{"id": i, "key": f"synthetic_{i}", "zh": f"合成{i}"} for i in range(16)]}), encoding="utf-8")
        config = load_classes(cls.classes, require_frozen=True)
        cls.metadata = {"model_name": "efficientnet_v2_s", "model_id": "SYNTHETIC-program-check",
            "stage": 8, "input_size": 96, "preprocessing": "foodai.vision.get_transforms/v1",
            "class_list": [{"id": c.id, "key": c.key, "zh": c.zh} for c in config.classes[:8]],
            "class_list_sha256": class_list_hash(config.classes[:8])}
        cls.checkpoint = cls.root / "synthetic.pt"
        model = get_model(8, pretrained=False)
        torch.save({"model_state": model.state_dict(), "metadata": cls.metadata}, cls.checkpoint)

    def test_sidecar_cannot_make_invalid_weights_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "best.pt").write_bytes(b"SYNTHETIC invalid weights")
            (root / "run_info.json").write_text(json.dumps(self.metadata), encoding="utf-8")
            adapter = PredictionAdapter(root / "best.pt", self.classes, device="cpu")
            self.assertEqual(adapter.status().state, "contract_error")
            self.assertFalse(adapter.is_ready)
            self.assertIsNone(adapter._model)

    def test_real_model_ready_and_top3_through_shared_loader(self):
        from foodai.checkpoint import load_classifier_checkpoint
        with patch("foodai.checkpoint.load_classifier_checkpoint", wraps=load_classifier_checkpoint) as loader:
            adapter = PredictionAdapter(self.checkpoint, self.classes, device="cpu")
            self.assertEqual(adapter.status().state, "ready", adapter.status().message)
            self.assertIsNotNone(adapter._model)
            result = adapter.predict(Image.new("RGB", (110, 100), (120, 50, 30)))
        self.assertEqual(loader.call_count, 1)
        payload = as_contract_dict(result)
        self.assertEqual(payload["model_id"], "SYNTHETIC-program-check")
        self.assertEqual(payload["class_list_sha256"], self.metadata["class_list_sha256"])
        self.assertEqual(len(payload["predictions"]), 3)
        scores = [row["score"] for row in payload["predictions"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(0 <= row["class_id"] < 8 for row in payload["predictions"]))
        self.assertEqual(payload["top1_id"], payload["predictions"][0]["class_id"])

    def test_embedded_metadata_is_authoritative(self):
        sidecar = self.root / "run_info.json"
        sidecar.write_text(json.dumps({"stage": 3, "class_list": [], "model_id": "wrong-sidecar"}), encoding="utf-8")
        try:
            adapter = PredictionAdapter(self.checkpoint, self.classes, device="cpu")
            self.assertEqual(adapter.status().state, "ready", adapter.status().message)
            self.assertEqual(adapter.status().model_id, "SYNTHETIC-program-check")
        finally:
            sidecar.unlink()

    def test_short_corrupt_pickle_is_contract_error(self):
        from foodai.checkpoint import load_classifier_checkpoint
        config = load_classes(self.classes, require_frozen=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.pt"
            for contents in (b"a", b"abc", b"\x80", b"h\x00.", b"J\x00", b"\x80\x02r\x00"):
                with self.subTest(contents=contents):
                    path.write_bytes(contents)
                    with self.assertRaises(ValueError):
                        load_classifier_checkpoint(path, config)
                    self.assertEqual(PredictionAdapter(path, self.classes).status().state, "contract_error")

    def test_valid_metadata_with_missing_weights_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.pt"
            torch.save({"metadata": self.metadata, "model_state": {}}, path)
            adapter = PredictionAdapter(path, self.classes, device="cpu")
            self.assertEqual(adapter.status().state, "contract_error")
            self.assertIsNone(adapter._model)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA-specific timing")
    def test_cuda_timing_synchronizes_before_and_after(self):
        adapter = PredictionAdapter(self.checkpoint, self.classes, device="cuda")
        self.assertEqual(adapter.status().state, "ready", adapter.status().message)
        with patch("torch.cuda.synchronize", wraps=torch.cuda.synchronize) as sync, \
                patch("foodai.predictor.time.perf_counter", side_effect=[1.0, 1.25]):
            result = adapter.predict(Image.new("RGB", (110, 100), (10, 20, 30)))
        self.assertEqual(sync.call_count, 2)
        self.assertEqual(result.latency_ms, 250.0)


if __name__ == "__main__":
    unittest.main()
