import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foodai.contracts import ContractError, class_list_hash, load_classes, validate_manifest
from foodai.inference_contract import ModelNotReadyError, predict_unconfigured

FIELDS = "image_id relative_path label_id class_key source_id group_id split license sha256".split()


class TestContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.classes = self.root / "classes.json"
        self.classes.write_text(json.dumps({
            "version": "v1", "status": "frozen",
            "classes": [{"id": 0, "key": "corn", "zh": "玉米"},
                        {"id": 1, "key": "potato", "zh": "土豆"}]
        }, ensure_ascii=False), encoding="utf-8")
        self.conf = load_classes(self.classes, require_frozen=True)
        self.manifest = self.root / "data" / "manifest.csv"
        self.manifest.parent.mkdir()

    def write_rows(self, rows):
        with self.manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def rec(self, image_id, label, group, split, sha=""):
        return {
            "image_id": image_id, "relative_path": f"data/raw/{image_id}.jpg",
            "label_id": label, "class_key": ["corn", "potato"][label],
            "source_id": "public_source", "group_id": group,
            "split": split, "license": "CC-BY-4.0", "sha256": sha,
        }

    def test_clean_split(self):
        self.write_rows([self.rec("im1", 0, "shoot1", "train"),
                         self.rec("im2", 1, "shoot2", "val"),
                         self.rec("im3", 0, "shoot3", "test")])
        recs, report = validate_manifest(self.manifest, self.conf, repo_root=self.root)
        self.assertEqual(report["images"], 3)
        self.assertEqual(report["splits"]["test"], 1)
        self.assertEqual(len(recs), 3)

    def test_group_leakage_rejected(self):
        self.write_rows([self.rec("im1", 0, "original", "train"),
                         self.rec("im2", 0, "original", "test")])
        with self.assertRaisesRegex(ContractError, "GROUP LEAKAGE"):
            validate_manifest(self.manifest, self.conf)

    def test_exact_duplicate_leakage_rejected(self):
        shared_hash = "a" * 64
        self.write_rows([self.rec("im1", 0, "first", "train", sha=shared_hash),
                         self.rec("im2", 0, "second", "test", sha=shared_hash)])
        with self.assertRaisesRegex(ContractError, "IDENTICAL IMAGE LEAKAGE"):
            validate_manifest(self.manifest, self.conf)

    def test_wrong_label_mapping_rejected(self):
        r = self.rec("im1", 0, "x", "train")
        r["class_key"] = "potato"
        self.write_rows([r])
        with self.assertRaisesRegex(ContractError, "class_key conflicts"):
            validate_manifest(self.manifest, self.conf)

    def test_path_escape_rejected(self):
        r = self.rec("im1", 0, "x", "train")
        r["relative_path"] = "../private_photo.jpg"
        self.write_rows([r])
        with self.assertRaisesRegex(ContractError, "repo-relative"):
            validate_manifest(self.manifest, self.conf)

    def test_draft_blocked(self):
        data = json.loads(self.classes.read_text(encoding="utf-8"))
        data["status"] = "draft"
        self.classes.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "DRAFT"):
            load_classes(self.classes, require_frozen=True)

    def test_class_hash_prefix_stable(self):
        original = class_list_hash(self.conf.classes)
        self.assertEqual(original, class_list_hash(self.conf.classes[:2]))

    def test_missing_model_is_explicit(self):
        with self.assertRaises(ModelNotReadyError):
            predict_unconfigured(None)


if __name__ == "__main__":
    unittest.main()
