"""Acceptance-report statistics with controlled unit fixtures, never model accuracy."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import acceptance_run
from foodai.inference_contract import ClassPrediction, ModelNotReadyError, PredictionResult
from foodai.predictor import AdapterInfo, InvalidImageError


class TestAcceptance(unittest.TestCase):
    def run_unit_report(self, *, accept_corrupt=False):
        # Replacing median with last latency or extrapolating P95 must fail this test.
        results = iter(PredictionResult("UNIT", "a" * 64,
            (ClassPrediction(0, "unit", "单元夹具", .8),), 0, latency)
            for latency in (500., 10., 30., 90.))

        class UnitAdapter:
            def __init__(self, checkpoint=None, **kwargs):
                self.ready = checkpoint is not None

            def status(self):
                return AdapterInfo(state="ready" if self.ready else "not_ready", message="UNIT ONLY",
                    checkpoint="unit.pt", model_id="UNIT", num_classes=8, input_size=224,
                    class_list_sha256="a" * 64, checkpoint_sha256="b" * 64, device="cpu")

            def predict(self, image):
                if not self.ready:
                    raise ModelNotReadyError("UNIT missing weights")
                if Path(image).name == "corrupt.jpg":
                    if accept_corrupt:
                        return PredictionResult("UNIT", "a" * 64, (), 0, 0.)
                    raise InvalidImageError("UNIT corrupt input")
                return next(results)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "sample.jpg").write_bytes(b"UNIT fixture; no real image prediction")
            out = root / "new-measurement.json"
            with patch.object(acceptance_run, "PredictionAdapter", UnitAdapter), \
                    patch.object(acceptance_run.time, "perf_counter", side_effect=[0., .01, 1., 1.03, 2., 2.09]), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = acceptance_run.main(["--checkpoint", str(root / "private" / "best.pt"),
                    "--images-dir", str(root), "--repeats", "3", "--out", str(out)])
            report = json.loads(out.read_text(encoding="utf-8"))
        return code, report

    def test_report_uses_all_samples_and_observed_p95(self):
        code, report = self.run_unit_report()
        self.assertEqual(code, 0)
        row = report["images"][0]
        self.assertEqual(row["latency_ms_internal_median"], 30.)
        self.assertEqual(row["latency_ms_wall_median"], 30.)
        self.assertEqual(row["latency_ms_wall_p95"], 90.)
        self.assertEqual(row["latency_ms_internal_samples"], [10., 30., 90.])
        self.assertEqual(len(row["latency_ms_wall_samples"]), 3)
        self.assertEqual(report["checkpoint"], "best.pt")
        self.assertEqual(report["measurement"]["warmup_per_image"], 1)
        self.assertEqual(report["measurement"]["p95_method"], "nearest_rank: sorted_samples[ceil(0.95*N)-1]")

    def test_failed_error_path_returns_nonzero_and_keeps_report(self):
        code, report = self.run_unit_report(accept_corrupt=True)
        self.assertNotEqual(code, 0)
        self.assertTrue(report["error_cases"]["corrupt_image"].startswith("FAIL"))

    def test_single_repeat_p95_is_single_measurement(self):
        # One repeat is valid, and percentile must not raise StatisticsError.
        self.assertEqual(acceptance_run.percentile95([12.]), 12.)

    def test_invalid_repeats_rejected_before_model_loading(self):
        for repeats in ("0", "-1", "1.5"):
            with self.subTest(repeats=repeats), tempfile.TemporaryDirectory() as folder:
                (Path(folder) / "unit.png").write_bytes(b"UNIT")
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    acceptance_run.main(["--checkpoint", "missing.pt", "--images-dir", folder,
                                         "--repeats", repeats])
                self.assertEqual(caught.exception.code, 2)

    def test_empty_and_missing_image_directories_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for path in (folder, str(Path(folder) / "missing")):
                with self.subTest(path=path):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                        acceptance_run.main(["--checkpoint", "missing.pt", "--images-dir", path])
                    self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
