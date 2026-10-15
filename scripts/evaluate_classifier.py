"""Locked held-out evaluation: run only AFTER model/hyperparameters are chosen on val."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foodai.contracts import ContractError, class_list_hash, load_classes, validate_manifest


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
    except (ContractError, OSError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    import torch
    from torch.utils.data import DataLoader
    from foodai.vision import get_image_dataset, get_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Checkpoints produced by this project only; never load untrusted pickle objects.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    info = checkpoint.get("metadata", {})
    n = int(info.get("stage", 0))
    if n not in (8, 12, 16) or n > len(config.classes):
        print("BLOCKED: checkpoint stage incompatible with current class schema", file=sys.stderr)
        return 2
    if class_list_hash(config.classes[:n]) != info.get("class_list_sha256"):
        print("BLOCKED: checkpoint class mapping differs from frozen config", file=sys.stderr)
        return 2
    use_rows = [r for r in records if r.split == args.split and r.label_id < n]
    class_counts = Counter(r.label_id for r in use_rows)
    if not use_rows or any(class_counts[c.id] == 0 for c in config.classes[:n]):
        print("BLOCKED: held-out split is empty or missing some active classes", file=sys.stderr)
        return 2
    model = get_model(n, pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(device).eval()
    loader = DataLoader(get_image_dataset(use_rows, ROOT, int(info["input_size"]), train=False),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)

    confusion = [[0] * n for _ in range(n)]
    with torch.inference_mode():
        for imgs, targets in loader:
            logits = model(imgs.to(device))
            predictions = logits.argmax(dim=1).cpu().tolist()
            for truth, pred in zip(targets.tolist(), predictions):
                confusion[truth][pred] += 1
    per_class = []
    for c in config.classes[:n]:
        total = sum(confusion[c.id])
        per_class.append({"class_key": c.key, "count": total,
                          "accuracy": confusion[c.id][c.id] / total})
    correct = sum(confusion[i][i] for i in range(n))
    overall = sum(sum(row) for row in confusion)
    result = {
        "split": args.split,
        "stage": n,
        "checkpoint": str(args.checkpoint),
        "training_manifest_sha256": info.get("dataset_manifest_sha256"),
        "evaluation_manifest_sha256": audit["manifest_sha256"],
        "correct": correct,
        "total": overall,
        "top1_accuracy": correct / overall,
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
