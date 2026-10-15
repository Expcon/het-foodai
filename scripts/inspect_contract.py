"""Inspect draft food taxonomy without installing PyTorch."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foodai.contracts import class_list_hash, load_classes

config_file = ROOT / "configs" / "classes.draft.json"
config = load_classes(config_file)
print(f"Class config: {config_file.name}, status={config.status}, count={len(config.classes)}")
for c in config.classes:
    print(f"  [{c.id}] {c.key} / {c.zh}")
print(f"Class-list hash: {class_list_hash(config.classes)}")
print("NEXT: group approval, then freeze configs/classes.json. No training data is present.")
