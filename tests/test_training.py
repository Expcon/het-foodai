"""Synthetic fixtures only: pipeline checks, never food recognition scores."""
import contextlib
import csv
import io
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from foodai.contracts import REQUIRED_FIELDS, class_list_hash, hash_file, load_classes, validate_manifest
from scripts import evaluate_classifier, train_classifier

try:
    import torch
    from PIL import Image
    from foodai import vision
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Project torch/torchvision/Pillow environment required")
class TestTraining(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="foodai-synthetic-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.classes = self.root / "synthetic-classes.json"
        self.classes.write_text(json.dumps({
            "status": "frozen", "version": "SYNTHETIC-ONLY",
            "classes": [{"id": i, "key": f"synthetic_{i}", "zh": f"合成{i}"} for i in range(16)],
        }), encoding="utf-8")
        self.config = load_classes(self.classes, require_frozen=True)
        self.manifest = self.root / "manifest.csv"
        self.rows = []
        for split in ("train", "val", "test"):
            for label in range(8):
                name = f"{split}-{label}.png"
                path = self.root / name
                if split == "test":
                    # Valid manifest/hash, deliberately undecodable: training must never decode test.
                    path.write_bytes(f"SYNTHETIC held-out {label}".encode())
                else:
                    Image.new("RGB", (110, 100), (label * 23, len(split) * 30, 60)).save(path)
                self.rows.append(dict(zip(REQUIRED_FIELDS, (
                    name, name, label, f"synthetic_{label}", "synthetic", name,
                    split, "SYNTHETIC-NOT-FOR-TRAINING", hash_file(path),
                ))))
        extra = dict(self.rows[0], image_id="extra", group_id="extra")
        self.rows.append(extra)
        self.write_manifest()

    def write_manifest(self):
        with self.manifest.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=REQUIRED_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

    def run_cli(self, module, *extra):
        argv = ["synthetic-test", "--manifest", str(self.manifest), "--classes", str(self.classes), *extra]
        if module is train_classifier:
            argv.extend(["--no-pretrained", "--epochs", "1"])
        with patch.object(module, "ROOT", self.root), patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return module.main()

    def test_windows_spawn_preserves_manifest_ids(self):
        records, _ = validate_manifest(self.manifest, self.config, repo_root=self.root)
        dataset = vision.get_image_dataset([records[7], records[0]], self.root, 96, train=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=1,
                                             multiprocessing_context="spawn")
        images, labels = next(iter(loader))
        self.assertEqual(labels.tolist(), [7, 0])
        self.assertEqual(tuple(images.shape), (2, 3, 96, 96))

    def test_missing_hash_blocks_both_entrypoints(self):
        self.rows[-1]["sha256"] = ""
        self.write_manifest()
        with patch.object(vision, "get_model", lambda n, **kwargs: torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, n))):
            self.assertEqual(self.run_cli(train_classifier, "--output-dir", str(self.root / "runs")), 2)
        self.assertEqual(self.run_cli(evaluate_classifier, "--checkpoint", "absent.pt"), 2)

    def test_draft_blocks_both_entrypoints(self):
        data = json.loads(self.classes.read_text())
        data["status"] = "draft"
        self.classes.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.run_cli(train_classifier), 2)
        self.assertEqual(self.run_cli(evaluate_classifier, "--checkpoint", "absent.pt"), 2)

    def test_invalid_training_options_rejected(self):
        for option in (("--num-workers", "-1"), ("--lr", "nan"), ("--lr", "inf")):
            with self.subTest(option=option), patch.object(vision, "get_model", lambda n, **kwargs:
                    torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, n))):
                with self.assertRaises((SystemExit, ValueError)) as raised:
                    self.run_cli(train_classifier, *option, "--output-dir", str(self.root / "runs"))
                self.assertIsInstance(raised.exception, SystemExit)
                self.assertEqual(raised.exception.code, 2)

    def test_seed_repeats_augmentation_and_shuffle(self):
        self.assertTrue(callable(getattr(train_classifier, "seed_everything", None)))
        records, _ = validate_manifest(self.manifest, self.config, repo_root=self.root)
        def sample():
            train_classifier.seed_everything(17)
            dataset = vision.get_image_dataset(records[:8], self.root, 96, train=True)
            loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1,
                                                 multiprocessing_context="spawn",
                                                 generator=torch.Generator().manual_seed(17),
                                                 worker_init_fn=train_classifier.seed_worker)
            return next(iter(loader))
        first, labels = sample()
        second, other_labels = sample()
        self.assertTrue(torch.equal(labels, other_labels))
        self.assertTrue(torch.equal(first, second))

    def test_epoch_logs_include_all_samples_and_validation_confusion(self):
        # Only replace the costly network; exercise actual loaders, optimizer, AMP and log IO.
        def tiny_model(n, *, pretrained):
            self.assertFalse(pretrained)
            linear = torch.nn.Linear(3, n)
            with torch.no_grad():
                linear.weight.zero_()
                linear.bias.zero_()
                linear.bias[-1] = 10
            return torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), linear)
        with patch.object(vision, "get_model", tiny_model):
            code = self.run_cli(train_classifier,
                                "--batch-size", "4", "--image-size", "96", "--num-workers", "1",
                                "--output-dir", str(self.root / "runs"))
        self.assertEqual(code, 0)
        run = next((self.root / "runs").iterdir())
        info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["per_class_counts"][0]["train"], 2)
        self.assertEqual(info["batch_size"], 4)
        with (run / "history.csv").open() as file:
            history = list(csv.DictReader(file))
        self.assertEqual(len(history), 1)
        self.assertEqual(int(history[0]["train_samples"]), 9)
        for key in ("train_loss", "val_loss", "train_accuracy", "val_accuracy"):
            self.assertTrue(math.isfinite(float(history[0][key])))
        confusion = json.loads((run / "val_epoch_0001.json").read_text(encoding="utf-8"))
        self.assertEqual(sum(map(sum, confusion["confusion_matrix_rows_true_cols_pred"])), 8)
        self.assertEqual(confusion["confusion_matrix_rows_true_cols_pred"], [[0, 0, 0, 0, 0, 0, 0, 1]] * 8)
        self.assertEqual(confusion["val_accuracy"], 0.125)
        self.assertEqual(len(confusion["per_class_recall"]), 8)
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["metadata"]["class_list"], info["class_list"])
        self.assertFalse(info["test_used_for_selection"])

    def test_checkpoint_metadata_and_weights_are_checked(self):
        self.assertIsNotNone(importlib.util.find_spec("foodai.checkpoint"))
        from foodai import checkpoint as checkpoint_io
        info = {
            "stage": 8, "model_name": "efficientnet_v2_s", "input_size": 96,
            "class_list": [{"id": c.id, "key": c.key, "zh": c.zh} for c in self.config.classes[:8]],
            "class_list_sha256": class_list_hash(self.config.classes[:8]),
        }
        path = self.root / "synthetic.pt"
        for updates in ({"stage": True}, {"stage": "8"}, {"input_size": 0},
                        {"model_name": "other"}, {"class_list": []}, {"class_list_sha256": "wrong"}):
            torch.save({"metadata": dict(info, **updates), "model_state": {}}, path)
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                checkpoint_io.load_classifier_checkpoint(path, self.config)
        torch.save({"metadata": info, "model_state": {}}, path)
        with self.assertRaisesRegex(ValueError, "weights"):
            checkpoint_io.load_classifier_checkpoint(path, self.config)
        torch.save({"unsafe": Path("not-allowed")}, path)
        with self.assertRaises(ValueError):
            checkpoint_io.load_classifier_checkpoint(path, self.config)

    def test_failed_save_keeps_previous_checkpoint(self):
        from foodai.checkpoint import save_checkpoint
        path = self.root / "best.pt"
        torch.save({"epoch": 1}, path)
        def interrupted_save(_checkpoint, temporary):
            temporary.write_bytes(b"partial write")
            raise OSError("synthetic disk failure")
        with patch.object(torch, "save", interrupted_save), self.assertRaises(OSError):
            save_checkpoint(path, {"epoch": 2})
        self.assertEqual(torch.load(path, weights_only=True)["epoch"], 1)
        self.assertFalse(path.with_suffix(".pt.tmp").exists())

    def test_changed_manifest_blocks_held_out_evaluation(self):
        tiny = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 8))
        info = {
            "stage": 8, "model_name": "efficientnet_v2_s", "input_size": 96,
            "class_list": [{"id": c.id, "key": c.key, "zh": c.zh} for c in self.config.classes[:8]],
            "class_list_sha256": class_list_hash(self.config.classes[:8]),
            "dataset_manifest_sha256": hash_file(self.manifest),
        }
        checkpoint = self.root / "synthetic.pt"
        torch.save({"metadata": info, "model_state": tiny.state_dict()}, checkpoint)
        self.rows[0]["group_id"] = "changed-after-training"
        self.write_manifest()
        with patch.object(vision, "get_model", lambda n, **kwargs: tiny):
            self.assertEqual(self.run_cli(evaluate_classifier, "--checkpoint", str(checkpoint)), 2)


if __name__ == "__main__":
    unittest.main()
