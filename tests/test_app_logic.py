"""UI 文案与表格逻辑测试：不依赖 gradio / torch。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foodai.app_logic import (adapter_banner, assert_no_accuracy_claim, banner,  # noqa: E402
                              format_top3, image_banner, latency_markdown, run_prediction)
from foodai.inference_contract import (ClassPrediction, ModelNotReadyError,  # noqa: E402
                                       PredictionResult)
from foodai.predictor import AdapterInfo  # noqa: E402


class NotReadyAdapter:
    def predict(self, _image):
        raise ModelNotReadyError("模型尚未训练／未加载。此处没有模拟识别分数。")


class TestBanners(unittest.TestCase):
    def test_not_ready_banner_disables_button(self):
        info = AdapterInfo(state="not_ready", message="模型尚未训练／未加载。此处没有模拟识别分数。")
        text, enabled = adapter_banner(info)
        self.assertFalse(enabled)
        self.assertIn("模型未就绪", text)
        self.assertIn("没有模拟识别分数", text)

    def test_contract_error_banner_disables_button(self):
        info = AdapterInfo(state="contract_error", message="类别契约不符（第 1 项）")
        text, enabled = adapter_banner(info)
        self.assertFalse(enabled)
        self.assertIn("已拒绝加载", text)

    def test_ready_banner_enables_button(self):
        info = AdapterInfo(state="ready", message="ok", model_id="effnetv2s-stage8-run001",
                           num_classes=8, input_size=224, class_list_sha256="a" * 64,
                           checkpoint_sha256="b" * 64, device="cpu")
        text, enabled = adapter_banner(info)
        self.assertTrue(enabled)
        self.assertIn("模型已就绪", text)
        self.assertIn("8 类", text)

    def test_empty_upload_is_neutral(self):
        text, ok = image_banner(None)
        self.assertFalse(ok)
        self.assertIn("等待上传", text)


class TestResults(unittest.TestCase):
    def setUp(self):
        self.result = PredictionResult(
            model_id="effnetv2s-stage8-run001",
            class_list_sha256="a" * 64,
            predictions=(ClassPrediction(1, "potato", "土豆", 0.91),
                         ClassPrediction(0, "corn", "玉米", 0.06),
                         ClassPrediction(3, "pumpkin", "南瓜", 0.02)),
            top1_id=1,
            latency_ms=24.53,
        )

    def test_top3_rows(self):
        rows = format_top3(self.result)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], [1, "土豆", "potato", "0.9100"])
        self.assertEqual(rows[2][0], 3)

    def test_latency_states_scope(self):
        text = latency_markdown(self.result)
        self.assertIn("24.5 ms", text)
        self.assertIn("不代表板端性能", text)

    def test_failure_returns_no_scores(self):
        status, rows, latency = run_prediction(NotReadyAdapter(), "whatever.jpg")
        self.assertIn("无法完成识别", status)
        self.assertEqual(rows, [])
        self.assertEqual(latency, "")

    def test_no_accuracy_wording_anywhere(self):
        status, rows, latency = run_prediction(NotReadyAdapter(), "x.jpg")
        texts = [status, latency, banner("ok", "a"), banner("error", "b", "c", "d"),
                 str(rows), str(format_top3(self.result))]
        assert_no_accuracy_claim(texts)


if __name__ == "__main__":
    unittest.main()
