"""EfficientNetV2-S 8/12/16-class joint training; train+val ONLY.

This script refuses draft class configs and never examines the test split for model selection.
Requires torchvision/PyTorch (GPU build for user's machine).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foodai.contracts import ContractError, class_list_hash, load_classes, validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.csv")
    parser.add_argument("--classes", type=Path, default=ROOT / "configs/classes.json")
    parser.add_argument("--num-classes", type=int, default=8, help="Cumulative categories in fixed ID order: 8, 12 or 16")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0, help="0 safer on Windows for the first run")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                        help="ImageNet weights, automatically fetched by torchvision on first use")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size < 2 or args.image_size < 96 or args.lr <= 0:
        parser.error("epochs>0, batch_size>=2, image_size>=96, lr>0 are required")
    try:
        classes = load_classes(args.classes, require_frozen=True)
        if args.num_classes not in (8, 12, 16) or args.num_classes > len(classes.classes):
            raise ContractError("Requested stage must be 8, 12, or 16 and not exceed frozen classes")
        rows, audit = validate_manifest(args.manifest, classes, repo_root=ROOT, check_files=True)
    except (ContractError, OSError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2

    active = classes.classes[:args.num_classes]
    train_rows = [r for r in rows if r.split == "train" and r.label_id < args.num_classes]
    val_rows = [r for r in rows if r.split == "val" and r.label_id < args.num_classes]
    train_counts = Counter(r.label_id for r in train_rows)
    val_counts = Counter(r.label_id for r in val_rows)
    missing_train = [c.key for c in active if train_counts[c.id] == 0]
    missing_val = [c.key for c in active if val_counts[c.id] == 0]
    if missing_train or missing_val:
        print(f"BLOCKED: class coverage missing: train={missing_train}, val={missing_val}", file=sys.stderr)
        return 2

    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from foodai.vision import get_image_dataset, get_model

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available; running a large model on CPU may be slow")

    train_dataset = get_image_dataset(train_rows, ROOT, args.image_size, train=True)
    val_dataset = get_image_dataset(val_rows, ROOT, args.image_size, train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                              drop_last=(len(train_rows) > args.batch_size and len(train_rows) % args.batch_size == 1))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    print(f"Stage={args.num_classes}; train={len(train_rows)}; val={len(val_rows)}; device={device}")
    print("Test/external_test rows are deliberately excluded from the training process.")

    model = get_model(args.num_classes, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    run_dir = args.output_dir / f"stage{args.num_classes}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "stage": args.num_classes,
        "model_name": "efficientnet_v2_s",
        "pretrained": args.pretrained,
        "class_list": [{"id": c.id, "key": c.key, "zh": c.zh} for c in active],
        "class_list_sha256": class_list_hash(active),
        "frozen_class_config_sha256": classes.file_sha256,
        "dataset_manifest_sha256": audit["manifest_sha256"],
        "dataset_train_count": len(train_rows),
        "dataset_val_count": len(val_rows),
        "input_size": args.image_size,
        "seed": args.seed,
        "device": str(device),
        "torch": str(torch.__version__),
        "epochs_requested": args.epochs,
        "validation_used_for_selection": True,
        "test_used_for_selection": False,
    }
    (run_dir / "run_info.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    history: list[dict] = []
    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_n, total_correct = 0.0, 0, 0
        for batch, labels in train_loader:
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            n = labels.size(0)
            total_loss += loss.item() * n
            total_n += n
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
        model.eval()
        val_n, val_correct, val_loss = 0, 0, 0.0
        with torch.inference_mode():
            for batch, labels in val_loader:
                batch, labels = batch.to(device), labels.to(device)
                logits = model(batch)
                loss = criterion(logits, labels)
                val_n += labels.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_loss += loss.item() * labels.size(0)
        if total_n == 0:
            raise RuntimeError("No complete training batches. Check batch-size/data counts.")
        result = {
            "epoch": epoch,
            "train_loss": total_loss / total_n,
            "train_accuracy": total_correct / total_n,
            "val_loss": val_loss / val_n,
            "val_accuracy": val_correct / val_n,
        }
        history.append(result)
        print(f"E{epoch:02d} train_acc={result['train_accuracy']:.4f} val_acc={result['val_accuracy']:.4f}")
        if result["val_accuracy"] > best_val:
            best_val = result["val_accuracy"]
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_accuracy": best_val, "metadata": metadata}, run_dir / "best.pt")

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f"DONE (validation only). Model and logs saved under: {run_dir}")
    print(f"Best validation accuracy={best_val:.4f}. Do not call it test accuracy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
