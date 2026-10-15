"""Pure-standard-library contracts shared across data, training and UI.

No IO side effects outside reading files. No torch/Pillow installation needed here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

REQUIRED_FIELDS = (
    "image_id", "relative_path", "label_id", "class_key", "source_id",
    "group_id", "split", "license", "sha256",
)
SPLITS = frozenset({"train", "val", "test", "external_test"})
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """An input violates a documented dataset/class interface."""


@dataclass(frozen=True)
class ClassInfo:
    id: int
    key: str
    zh: str


@dataclass(frozen=True)
class ClassConfig:
    version: str
    status: str
    classes: tuple[ClassInfo, ...]
    file_sha256: str


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    relative_path: str
    label_id: int
    class_key: str
    source_id: str
    group_id: str
    split: str
    license: str
    sha256: str


def hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def class_list_hash(classes: tuple[ClassInfo, ...] | list[ClassInfo]) -> str:
    """Stable hash of just ordered categories; remains usable after an append-only expansion."""
    data = [{"id": c.id, "key": c.key, "zh": c.zh} for c in classes]
    content = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_classes(path: str | Path, *, require_frozen: bool = False) -> ClassConfig:
    source = Path(path)
    if not source.exists():
        raise ContractError(f"Missing class configuration: {source}")
    try:
        data: Any = json.loads(source.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ContractError(f"Invalid class JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("classes"), list):
        raise ContractError("Class JSON needs an ordered 'classes' array")
    status = data.get("status")
    if status not in ("draft", "frozen"):
        raise ContractError("Class JSON status must be 'draft' or 'frozen'")
    if require_frozen and status != "frozen":
        raise ContractError("Class list is DRAFT; obtain team approval before formal training")
    if not data["classes"]:
        raise ContractError("Class list cannot be empty")
    parsed: list[ClassInfo] = []
    seen_keys: set[str] = set()
    for idx, item in enumerate(data["classes"]):
        if not isinstance(item, dict) or type(item.get("id")) is not int:
            raise ContractError(f"classes[{idx}]: 'id' must be an integer")
        if item["id"] != idx:
            raise ContractError(f"classes[{idx}]: ID must be {idx}, got {item['id']}")
        key, zh = item.get("key"), item.get("zh")
        if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise ContractError(f"classes[{idx}]: invalid class key")
        if not isinstance(zh, str) or not zh.strip():
            raise ContractError(f"classes[{idx}]: nonblank Chinese display label required")
        if key in seen_keys:
            raise ContractError(f"Duplicate class key: {key}")
        seen_keys.add(key)
        parsed.append(ClassInfo(idx, key, zh.strip()))
    return ClassConfig(
        version=str(data.get("version", "unspecified")),
        status=status,
        classes=tuple(parsed),
        file_sha256=hash_file(source),
    )


def _valid_relative_path(value: str) -> bool:
    if not value or value.startswith(("/", "\\")) or "\\" in value or ":" in value:
        return False
    parts = PurePosixPath(value).parts
    return bool(parts) and "." not in parts and ".." not in parts


def validate_manifest(
    manifest: str | Path,
    config: ClassConfig,
    *,
    repo_root: str | Path | None = None,
    check_files: bool = False,
) -> tuple[list[ImageRecord], dict[str, Any]]:
    manifest = Path(manifest)
    if not manifest.is_file():
        raise ContractError(f"Missing manifest file: {manifest}")
    root = Path(repo_root).resolve() if repo_root else manifest.parent.parent.resolve()
    records: list[ImageRecord] = []
    by_id: set[str] = set()
    groups: dict[tuple[str, str], str] = {}
    sha_splits: dict[str, str] = {}
    counts: dict[str, int] = {s: 0 for s in sorted(SPLITS)}
    per_class: dict[str, dict[str, int]] = {}
    unknown_license = 0
    missing_hash = 0

    with manifest.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or any(field not in reader.fieldnames for field in REQUIRED_FIELDS):
            raise ContractError(f"Manifest header must include: {', '.join(REQUIRED_FIELDS)}")
        for rownum, raw in enumerate(reader, start=2):
            if None in raw:
                raise ContractError(f"Line {rownum}: more fields than header")
            row = {field: str(raw.get(field) or "").strip() for field in REQUIRED_FIELDS}
            for mandatory in ("image_id", "relative_path", "label_id", "class_key", "source_id", "group_id", "split", "license"):
                if not row[mandatory]:
                    raise ContractError(f"Line {rownum}: empty {mandatory}")
            if row["image_id"] in by_id:
                raise ContractError(f"Line {rownum}: duplicate image_id={row['image_id']}")
            by_id.add(row["image_id"])
            if not _valid_relative_path(row["relative_path"]):
                raise ContractError(f"Line {rownum}: path must be repo-relative with '/' separators")
            try:
                label_id = int(row["label_id"])
            except ValueError as exc:
                raise ContractError(f"Line {rownum}: invalid label_id") from exc
            if not 0 <= label_id < len(config.classes):
                raise ContractError(f"Line {rownum}: label ID {label_id} not found in class configuration")
            if row["class_key"] != config.classes[label_id].key:
                raise ContractError(f"Line {rownum}: class_key conflicts with label_id={label_id}")
            split = row["split"]
            if split not in SPLITS:
                raise ContractError(f"Line {rownum}: unsupported split '{split}'")
            group_key = (row["source_id"], row["group_id"])
            if group_key in groups and groups[group_key] != split:
                raise ContractError(f"Line {rownum}: GROUP LEAKAGE for source/group={group_key}: {groups[group_key]} vs {split}")
            groups[group_key] = split
            digest = row["sha256"].lower()
            if digest:
                if not HEX64.fullmatch(digest):
                    raise ContractError(f"Line {rownum}: sha256 must be 64 hexadecimal chars")
                if digest in sha_splits and sha_splits[digest] != split:
                    raise ContractError(f"Line {rownum}: IDENTICAL IMAGE LEAKAGE across splits")
                sha_splits[digest] = split
            else:
                missing_hash += 1
            if row["license"].lower() in {"unknown", "tbd", "pending"}:
                unknown_license += 1
            if check_files:
                path = (root / row["relative_path"]).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise ContractError(f"Line {rownum}: missing/outside image file {row['relative_path']}")
                if digest and hash_file(path) != digest:
                    raise ContractError(f"Line {rownum}: SHA256 does not match file bytes")
            record = ImageRecord(
                image_id=row["image_id"], relative_path=row["relative_path"],
                label_id=label_id, class_key=row["class_key"], source_id=row["source_id"],
                group_id=row["group_id"], split=split, license=row["license"], sha256=digest,
            )
            records.append(record)
            counts[split] += 1
            per_class.setdefault(record.class_key, {s: 0 for s in SPLITS})[split] += 1
    return records, {
        "images": len(records),
        "splits": counts,
        "per_class": per_class,
        "distinct_capture_groups": len(groups),
        "missing_sha256": missing_hash,
        "unverified_license": unknown_license,
        "manifest_sha256": hash_file(manifest),
    }
