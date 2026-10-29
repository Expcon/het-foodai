"""Run full EfficientNetV2-S GPU/AMP CLI checks on TEMPORARY SYNTHETIC data.

No network access or pretrained weights. All scratch classes, images and models
are deleted afterwards. Scores describe random pixels, never food recognition.
"""
from __future__ import annotations

import csv
import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(after_training=None):
    import torch
    from PIL import Image
    from foodai.checkpoint import load_classifier_checkpoint
    from foodai.contracts import ContractError, REQUIRED_FIELDS, hash_file, load_classes
    from foodai.vision import get_model, get_transforms

    assert torch.cuda.is_available(), "This smoke check specifically requires the project CUDA environment"
    print("SYNTHETIC ONLY — full EfficientNetV2-S GPU AMP program check; no food accuracy claim", flush=True)
    print(f"torch={torch.__version__}; GPU={torch.cuda.get_device_name(0)}", flush=True)
    (ROOT / "runs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="SYNTHETIC-", dir=ROOT / "runs") as folder:
        scratch = Path(folder)
        classes = scratch / "synthetic-classes.json"
        classes.write_text(json.dumps({
            "status": "frozen", "version": "SYNTHETIC-ONLY",
            "classes": [{"id": i, "key": f"synthetic_{i}", "zh": f"合成{i}"} for i in range(16)],
        }), encoding="utf-8")
        manifest = scratch / "synthetic-manifest.csv"
        rng = random.Random(42)
        rows = []
        for split, count in (("train", 17), ("val", 8), ("test", 8), ("external_test", 8)):
            for index in range(count):
                label = index % 8
                name = f"{split}-{index}.png"
                image = scratch / name
                Image.frombytes("RGB", (104, 104), rng.randbytes(104 * 104 * 3)).save(image)
                rows.append(dict(zip(REQUIRED_FIELDS, (
                    name, image.relative_to(ROOT).as_posix(), label, f"synthetic_{label}",
                    "synthetic", name, split, "SYNTHETIC-NOT-FOR-TRAINING", hash_file(image),
                ))))
        with manifest.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=REQUIRED_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        def run(script, *args):
            command = [sys.executable, str(ROOT / "scripts" / script), "--manifest", str(manifest),
                       "--classes", str(classes), *map(str, args)]
            print("COMMAND:", subprocess.list2cmdline(command), flush=True)
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=300)
            if result.returncode:
                print(result.stdout, result.stderr)
                raise RuntimeError(f"Synthetic {script} failed, exit={result.returncode}")
            print(f"exit={result.returncode}", flush=True)
            return result.stdout

        output = scratch / "outputs"
        print(run("train_classifier.py", "--no-pretrained", "--epochs", 2, "--batch-size", 8,
                  "--image-size", 96, "--num-workers", 1, "--output-dir", output), flush=True)
        run_dir = next(output.iterdir())
        info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
        assert info["amp"] and info["device"] == "cuda" and info["deterministic_algorithms"]
        with (run_dir / "history.csv").open() as file:
            history = list(csv.DictReader(file))
        assert len(history) == 2
        assert all(int(row["train_samples"]) == 17 and int(row["val_samples"]) == 8 for row in history)
        assert all(math.isfinite(float(row[key])) for row in history
                   for key in ("train_loss", "val_loss", "train_accuracy", "val_accuracy"))
        best = torch.load(run_dir / "best.pt", weights_only=True, map_location="cpu")
        selected = max(history, key=lambda row: float(row["val_accuracy"]))
        assert best["epoch"] == int(selected["epoch"])
        for epoch in (1, 2):
            report = json.loads((run_dir / f"val_epoch_{epoch:04d}.json").read_text(encoding="utf-8"))
            assert sum(map(sum, report["confusion_matrix_rows_true_cols_pred"])) == 8

        config = load_classes(classes, require_frozen=True)
        model, metadata = load_classifier_checkpoint(run_dir / "best.pt", config)
        assert model.classifier[-1].out_features == 8 and not model.training
        torch.testing.assert_close(model.classifier[-1].weight, best["model_state"]["classifier.1.weight"])
        with Image.open(scratch / "val-0.png") as image:
            batch = get_transforms(metadata["input_size"], train=False)(image.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            logits = model.to("cuda")(batch.to("cuda"))
        assert tuple(logits.shape) == (1, 8) and torch.isfinite(logits).all()
        scores, ids = logits.softmax(dim=1).topk(3)
        assert ids.max() < 8 and scores[0, 0] >= scores[0, 1] >= scores[0, 2]
        # A deterministic initial model distinguishes a real optimizer update from skipped AMP steps.
        torch.manual_seed(42)
        initial = get_model(8, pretrained=False)
        assert not torch.equal(initial.classifier[-1].weight, best["model_state"]["classifier.1.weight"])
        del model, initial, best
        torch.cuda.empty_cache()

        for stage in (12, 16):
            expanded = get_model(stage, pretrained=False)
            assert expanded.classifier[-1].out_features == stage
            del expanded
        for split in ("test", "external_test"):
            run("evaluate_classifier.py", "--checkpoint", run_dir / "best.pt", "--split", split,
                "--batch-size", 8)
            report = json.loads((run_dir / f"{split}_evaluation.json").read_text(encoding="utf-8"))
            assert report["total"] == 8 and math.isfinite(report["loss"])
            assert sum(map(sum, report["confusion_matrix_rows_true_cols_pred"])) == 8
            assert report["class_list_sha256"] == metadata["class_list_sha256"]
        bad_config = load_classes(classes, require_frozen=True)
        from dataclasses import replace
        swapped = list(bad_config.classes)
        swapped[0] = replace(swapped[0], zh="错误的合成标签")
        try:
            load_classifier_checkpoint(run_dir / "best.pt", replace(bad_config, classes=tuple(swapped)))
        except ContractError:
            pass
        else:
            raise AssertionError("Changed class mapping must be rejected")
        if after_training is not None:
            after_training(scratch, run_dir, classes)
    print("PASS: synthetic GPU AMP, optimizer update, logs, safe reload, RGB Top-3, held-out CLIs, 8/12/16 heads")
    print("Temporary synthetic configs/images/checkpoints deleted. Real training remains BLOCKED.")


if __name__ == "__main__":
    main()
