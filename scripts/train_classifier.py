"""EfficientNetV2-S 8/12/16-class joint training; train+val ONLY.

This script refuses draft class configs and never examines the test split for model selection.
Requires torchvision/PyTorch (GPU build for user's machine).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foodai.contracts import ContractError, class_list_hash, hash_file, load_classes, validate_manifest


def seed_everything(seed: int):
    # Set before CUDA context creation; strict determinism fails on unsupported operations.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(_worker_id):
    import numpy as np
    import torch
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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

    if (args.epochs <= 0 or args.batch_size < 2 or args.image_size < 96
            or not math.isfinite(args.lr) or args.lr <= 0 or args.num_workers < 0
            or not 0 <= args.seed < 2**32):
        parser.error("epochs>0, batch_size>=2, image_size>=96, finite lr>0, num_workers>=0, 0<=seed<2**32 required")
    try:
        classes = load_classes(args.classes, require_frozen=True)
        if args.num_classes not in (8, 12, 16) or args.num_classes > len(classes.classes):
            raise ContractError("Requested stage must be 8, 12, or 16 and not exceed frozen classes")
        rows, audit = validate_manifest(args.manifest, classes, repo_root=ROOT, check_files=True)
        if audit["missing_sha256"]:
            raise ContractError("Formal training requires SHA256 for every manifest row to check split leakage")
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
    import torchvision
    from torch import nn
    from torch.utils.data import DataLoader
    from foodai.vision import get_image_dataset, get_model
    from foodai.checkpoint import save_checkpoint

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available; running a large model on CPU may be slow")

    train_dataset = get_image_dataset(train_rows, ROOT, args.image_size, train=True)
    val_dataset = get_image_dataset(val_rows, ROOT, args.image_size, train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                              generator=torch.Generator().manual_seed(args.seed), worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                            generator=torch.Generator().manual_seed(args.seed + 1), worker_init_fn=seed_worker)
    print(f"Stage={args.num_classes}; train={len(train_rows)}; val={len(val_rows)}; device={device}")
    print("Test/external_test rows are deliberately excluded from the training process.")
    per_class_counts = [{"id": c.id, "key": c.key, "zh": c.zh,
                         "train": train_counts[c.id], "val": val_counts[c.id]} for c in active]
    for count in per_class_counts:
        print(f"  {count['id']:02d} {count['key']}: train={count['train']} val={count['val']}")

    model = get_model(args.num_classes, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    run_dir = args.output_dir / f"stage{args.num_classes}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "model_id": run_dir.name,
        "stage": args.num_classes,
        "model_name": "efficientnet_v2_s",
        "pretrained": args.pretrained,
        "class_list": [{"id": c.id, "key": c.key, "zh": c.zh} for c in active],
        "class_list_sha256": class_list_hash(active),
        "frozen_class_config_sha256": classes.file_sha256,
        "class_config_version": classes.version,
        "dataset_manifest_sha256": audit["manifest_sha256"],
        "dataset_train_count": len(train_rows),
        "dataset_val_count": len(val_rows),
        "per_class_counts": per_class_counts,
        "manifest": str(args.manifest.resolve()),
        "classes_config": str(args.classes.resolve()),
        "command_args": sys.argv[1:],
        "input_size": args.image_size,
        "preprocessing": "foodai.vision.get_transforms/v1",
        "seed": args.seed,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "optimizer": "AdamW",
        "optimizer_defaults": optimizer.defaults,
        "lr": args.lr,
        "weight_decay": 1e-4,
        "amp": use_amp,
        "amp_dtype": "float16" if use_amp else None,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if use_amp else platform.processor(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "training_script_sha256": hash_file(__file__),
        "vision_sha256": hash_file(Path(__file__).resolve().parents[1] / "src/foodai/vision.py"),
        "epochs_requested": args.epochs,
        "selection_metric": "val_accuracy (first epoch wins ties)",
        "validation_used_for_selection": True,
        "test_used_for_selection": False,
    }
    (run_dir / "run_info.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    history_file = run_dir / "history.csv"
    fields = ["epoch", "train_samples", "val_samples", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    with history_file.open("w", newline="", encoding="utf-8") as output:
        csv.DictWriter(output, fieldnames=fields).writeheader()
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
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}; no checkpoint saved for this epoch")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            n = labels.size(0)
            total_loss += loss.item() * n
            total_n += n
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
        model.eval()
        val_n, val_correct, val_loss = 0, 0, 0.0
        confusion = torch.zeros((args.num_classes, args.num_classes), dtype=torch.int64)
        with torch.inference_mode():
            for batch, labels in val_loader:
                batch, labels = batch.to(device), labels.to(device)
                logits = model(batch)
                loss = criterion(logits, labels)
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"Non-finite validation loss at epoch {epoch}")
                predictions = logits.argmax(dim=1)
                indices = (labels * args.num_classes + predictions).cpu()
                confusion += torch.bincount(indices, minlength=args.num_classes**2).reshape(confusion.shape)
                val_n += labels.size(0)
                val_correct += (predictions == labels).sum().item()
                val_loss += loss.item() * labels.size(0)
        if total_n == 0 or val_n == 0:
            raise RuntimeError("No complete training batches. Check batch-size/data counts.")
        result = {
            "epoch": epoch,
            "train_samples": total_n,
            "val_samples": val_n,
            "train_loss": total_loss / total_n,
            "train_accuracy": total_correct / total_n,
            "val_loss": val_loss / val_n,
            "val_accuracy": val_correct / val_n,
        }
        with history_file.open("a", newline="", encoding="utf-8") as output:
            csv.DictWriter(output, fieldnames=fields).writerow(result)
        per_class = [{"id": c.id, "class_key": c.key, "count": val_counts[c.id],
                      "accuracy": confusion[c.id, c.id].item() / val_counts[c.id]} for c in active]
        report = {**result, "split": "val", "class_list": metadata["class_list"],
                  "class_list_sha256": metadata["class_list_sha256"], "per_class_recall": per_class,
                  "macro_recall": sum(c["accuracy"] for c in per_class) / args.num_classes,
                  "confusion_matrix_rows_true_cols_pred": confusion.tolist()}
        (run_dir / f"val_epoch_{epoch:04d}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"E{epoch:02d} train_loss={result['train_loss']:.6f} train_acc={result['train_accuracy']:.4f} "
              f"val_loss={result['val_loss']:.6f} val_acc={result['val_accuracy']:.4f}", flush=True)
        if result["val_accuracy"] > best_val:
            best_val = result["val_accuracy"]
            save_checkpoint(run_dir / "best.pt", {"model_state": model.state_dict(), "epoch": epoch,
                            "val_accuracy": best_val, "metadata": metadata})
    print(f"DONE (validation only). Model and logs saved under: {run_dir}")
    print(f"Best validation accuracy={best_val:.4f}. Do not call it test accuracy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
