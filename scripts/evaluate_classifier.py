"""Locked held-out evaluation: run only AFTER model/hyperparameters are chosen on val."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foodai.contracts import ContractError, hash_file, load_classes, validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.csv")
    parser.add_argument("--classes", type=Path, default=ROOT / "configs/classes.json")
    parser.add_argument("--split", choices=["test", "external_test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")

    try:
        config = load_classes(args.classes, require_frozen=True)
        records, audit = validate_manifest(args.manifest, config, repo_root=ROOT, check_files=True)
        if audit["missing_sha256"]:
            raise ContractError("Held-out evaluation requires SHA256 for every manifest row")
    except (ContractError, OSError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from foodai.vision import get_image_dataset
    from foodai.checkpoint import load_classifier_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, info = load_classifier_checkpoint(args.checkpoint, config)
        if info.get("dataset_manifest_sha256") != audit["manifest_sha256"]:
            raise ContractError("Evaluation manifest differs from the checkpoint's frozen manifest; restore that version")
    except ContractError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    n = info["stage"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    use_rows = [r for r in records if r.split == args.split and r.label_id < n]
    class_counts = Counter(r.label_id for r in use_rows)
    if not use_rows or any(class_counts[c.id] == 0 for c in config.classes[:n]):
        print("BLOCKED: held-out split is empty or missing some active classes", file=sys.stderr)
        return 2
    model = model.to(device).eval()
    loader = DataLoader(get_image_dataset(use_rows, ROOT, info["input_size"], train=False),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)

    confusion = [[0] * n for _ in range(n)]
    total_loss = 0.0
    with torch.inference_mode():
        for imgs, targets in loader:
            logits = model(imgs.to(device))
            loss = torch.nn.functional.cross_entropy(logits, targets.to(device), reduction="sum")
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite held-out loss; evaluation report not written")
            total_loss += loss.item()
            predictions = logits.argmax(dim=1).cpu().tolist()
            for truth, pred in zip(targets.tolist(), predictions):
                confusion[truth][pred] += 1
    per_class = []
    for c in config.classes[:n]:
        total = sum(confusion[c.id])
        per_class.append({"id": c.id, "class_key": c.key, "count": total,
                          "accuracy": confusion[c.id][c.id] / total})
    correct = sum(confusion[i][i] for i in range(n))
    overall = sum(sum(row) for row in confusion)
    result = {
        "split": args.split,
        "stage": n,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hash_file(args.checkpoint),
        "class_list": info["class_list"],
        "class_list_sha256": info["class_list_sha256"],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "training_manifest_sha256": info.get("dataset_manifest_sha256"),
        "evaluation_manifest_sha256": audit["manifest_sha256"],
        "correct": correct,
        "total": overall,
        "top1_accuracy": correct / overall,
        "loss": total_loss / overall,
        "macro_recall": sum(c["accuracy"] for c in per_class) / n,
        "per_class_recall": per_class,
        "confusion_matrix_rows_true_cols_pred": confusion,
    }
    report_file = args.checkpoint.parent / f"{args.split}_evaluation.json"
    report_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Report saved: {report_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
