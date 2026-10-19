"""Adapter 契约测试：不依赖 torch（只有元数据校验与图片校验）。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foodai.contracts import class_list_hash  # noqa: E402
from foodai.inference_contract import (ClassPrediction, ModelNotReadyError,  # noqa: E402
                                       PredictionResult, predict_unconfigured)
from foodai.predictor import (CheckpointContractError, InvalidImageError,  # noqa: E402
                              PredictionAdapter, as_contract_dict, sha256_file,
                              validate_image_file)

CLASSES = [{"id": i, "key": k, "zh": z} for i, (k, z) in enumerate([
    ("corn", "玉米"), ("potato", "土豆"), ("broccoli", "西兰花"),
    ("pumpkin", "南瓜"), ("chicken_wing", "鸡翅"), ("chicken_breast", "鸡胸肉"),
    ("sea_bass", "鲈鱼"), ("shrimp", "对虾"),
])]


class AdapterTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.classes = self.root / "classes.json"
        self.write_classes(CLASSES, status="frozen")
        self.run_dir = self.root / "runs" / "stage8-run001"
        self.run_dir.mkdir(parents=True)
        self.ckpt = self.run_dir / "best.pt"
        self.ckpt.write_bytes(b"not-a-real-checkpoint")

    def write_classes(self, classes, status="frozen"):
        self.classes.write_text(json.dumps(
            {"version": "v1", "status": status, "classes": classes},
            ensure_ascii=False), encoding="utf-8")

    def write_run_info(self, classes, sha=None, input_size=224):
        payload = {
            "stage": len(classes),
            "model_name": "efficientnet_v2_s",
            "input_size": input_size,
            "class_list": classes,
            "class_list_sha256": sha if sha is not None else _hash_of(classes),
        }
        (self.run_dir / "run_info.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def adapter(self, **kw):
        kw.setdefault("checkpoint", self.ckpt)
        kw.setdefault("classes_path", self.classes)
        return PredictionAdapter(**kw)


def _hash_of(classes):
    from foodai.contracts import ClassInfo
    return class_list_hash([ClassInfo(c["id"], c["key"], c["zh"]) for c in classes])


class TestNotReadyStates(AdapterTestBase):
    def test_unconfigured_message_is_identical_to_contract(self):
        with self.assertRaises(ModelNotReadyError) as ctx:
            predict_unconfigured(None)
        expected = str(ctx.exception)

        adapter = PredictionAdapter(checkpoint=None, classes_path=self.classes)
        with self.assertRaises(ModelNotReadyError) as ctx2:
            adapter.predict(self.root / "x.jpg")
        self.assertEqual(str(ctx2.exception), expected)

    def test_missing_checkpoint_mentions_path(self):
        missing = self.root / "runs" / "nope.pt"
        adapter = PredictionAdapter(checkpoint=missing, classes_path=self.classes)
        info = adapter.status()
        self.assertEqual(info.state, "not_ready")
        self.assertIn("nope.pt", info.message)

    def test_missing_classes_config_is_not_ready(self):
        adapter = PredictionAdapter(checkpoint=self.ckpt, classes_path=self.root / "absent.json")
        info = adapter.status()
        self.assertEqual(info.state, "not_ready")
        self.assertIn("类别配置不可用", info.message)

    def test_draft_classes_are_rejected(self):
        self.write_classes(CLASSES, status="draft")
        info = self.adapter().status()
        self.assertEqual(info.state, "not_ready")
        self.assertIn("DRAFT", info.message)


class TestContractChecks(AdapterTestBase):
    def test_wrong_class_order_rejected(self):
        swapped = [dict(CLASSES[1]), dict(CLASSES[0])] + CLASSES[2:]
        self.write_run_info(swapped, sha=_hash_of(CLASSES))
        info = self.adapter().status()
        self.assertEqual(info.state, "contract_error")
        self.assertIn("类别契约不符", info.message)

    def test_wrong_display_name_rejected(self):
        renamed = [dict(c) for c in CLASSES]
        renamed[2]["zh"] = "花椰菜"
        self.write_run_info(renamed, sha=_hash_of(CLASSES))
        info = self.adapter().status()
        self.assertEqual(info.state, "contract_error")
        self.assertIn("类别契约不符", info.message)

    def test_class_list_hash_mismatch_rejected(self):
        self.write_run_info(CLASSES, sha="0" * 64)
        info = self.adapter().status()
        self.assertEqual(info.state, "contract_error")
        self.assertIn("class_list_sha256 不符", info.message)

    def test_prefix_of_frozen_list_is_ready(self):
        """8 类配置下，3 类 checkpoint 属于合法前缀，应就绪。"""
        active = CLASSES[:3]
        self.write_run_info(active, sha=_hash_of(active))
        info = self.adapter().status()
        self.assertEqual(info.state, "ready", info.message)
        self.assertEqual(info.num_classes, 3)
        self.assertEqual(info.input_size, 224)

    def test_checkpoint_sha_prefix_mismatch(self):
        self.write_run_info(CLASSES)
        info = self.adapter(expect_checkpoint_sha256="deadbeef").status()
        self.assertEqual(info.state, "contract_error")
        self.assertIn("SHA256 前缀不符", info.message)

    def test_checkpoint_sha_prefix_match(self):
        self.write_run_info(CLASSES)
        prefix = sha256_file(self.ckpt)[:12]
        info = self.adapter(expect_checkpoint_sha256=prefix).status()
        self.assertEqual(info.state, "ready", info.message)

    def test_missing_metadata_is_contract_error(self):
        info = self.adapter().status()   # 没有 run_info.json
        self.assertEqual(info.state, "contract_error")


class TestImageValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_text_file_named_jpg_rejected(self):
        bad = self.root / "fake.jpg"
        bad.write_text("this is not an image", encoding="utf-8")
        with self.assertRaisesRegex(InvalidImageError, "无法解码"):
            validate_image_file(bad)

    def test_empty_file_rejected(self):
        bad = self.root / "empty.png"
        bad.write_bytes(b"")
        with self.assertRaisesRegex(InvalidImageError, "0 字节"):
            validate_image_file(bad)

    def test_unsupported_suffix_rejected(self):
        bad = self.root / "photo.txt"
        bad.write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(InvalidImageError, "不支持的格式"):
            validate_image_file(bad)

    def test_real_png_accepted(self):
        from PIL import Image
        good = self.root / "good.png"
        Image.new("RGB", (32, 24), (120, 60, 30)).save(good)
        self.assertEqual(validate_image_file(good), good)


class TestContractShape(unittest.TestCase):
    def test_dict_matches_contract_fields(self):
        result = PredictionResult(
            model_id="effnetv2s-stage8-run001",
            class_list_sha256="a" * 64,
            predictions=(ClassPrediction(1, "potato", "土豆", 0.91),
                         ClassPrediction(0, "corn", "玉米", 0.06)),
            top1_id=1,
            latency_ms=24.5,
        )
        payload = as_contract_dict(result)
        self.assertEqual(set(payload), {"model_id", "class_list_sha256", "predictions",
                                        "top1_id", "latency_ms", "status"})
        self.assertEqual(payload["predictions"][0],
                         {"class_id": 1, "class_key": "potato", "label_zh": "土豆", "score": 0.91})
        self.assertEqual(payload["top1_id"], 1)
        self.assertEqual(payload["status"], "ok")


if __name__ == "__main__":
    unittest.main()

