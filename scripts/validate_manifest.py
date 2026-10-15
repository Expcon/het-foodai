"""CLI to check manifest/class consistency and train/val/test leakage."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foodai.contracts import ContractError, load_classes, validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.csv")
    parser.add_argument("--classes", type=Path, default=ROOT / "configs/classes.json")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--require-frozen", action="store_true", help="require approved class list")
    args = parser.parse_args()
    try:
        config = load_classes(args.classes, require_frozen=args.require_frozen)
        _, report = validate_manifest(args.manifest, config, repo_root=ROOT, check_files=args.check_files)
    except (ContractError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if config.status != "frozen":
        print("WARNING: taxonomy is still DRAFT")
    if report["missing_sha256"]:
        print("WARNING: some images have no full file hash")
    if report["unverified_license"]:
        print("WARNING: some sources have unverified licenses; do not redistribute")
    if not report["splits"]["test"]:
        print("WARNING: frozen test set is missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
