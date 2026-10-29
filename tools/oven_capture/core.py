"""Offline candidate records and guarded manifest handoff; never edits MVP data."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import csv
import io
import json
import math
from pathlib import Path
import re
import sys
import time
import uuid
import warnings

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from foodai.contracts import REQUIRED_FIELDS, SPLITS, hash_file, load_classes, validate_manifest
from scripts.audit_images import BKTree, duplicate_groups, image_facts
from scripts.propose_groups import UnionFind

RAW_ROOT = ROOT / 'data/raw/oven_capture'
TOKEN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}')
MAX_IMAGE_BYTES = 64 * 1024 * 1024


class NotReady(RuntimeError):
    """Selected camera/backend is unavailable; no hardware success is implied."""


def now():
    return datetime.now(timezone.utc).isoformat()


def guard_output(path):
    """CLI storage stays in an already Git-ignored capture subtree."""
    if not Path(path).resolve().is_relative_to(RAW_ROOT.resolve()):
        raise ValueError('Output must be under data/raw/oven_capture; MVP/source paths are forbidden')


def _protect_project(path):
    if Path(path).resolve().is_relative_to(ROOT):
        guard_output(path)


def _token(value):
    if not isinstance(value, str) or not TOKEN.fullmatch(value) or value.upper() in {'UNKNOWN', 'TBD', 'NONE'}:
        raise ValueError('A nonblank pseudonymous ID is required (no paths, UNKNOWN or serial-number guesses)')
    return value


def _classes(keys, *, allow_empty=False):
    config = load_classes(ROOT / 'configs/classes.json', require_frozen=True)
    valid = {c.key for c in config.classes}
    if not isinstance(keys, list) or (not keys and not allow_empty) or any(not isinstance(k, str) for k in keys) or len(set(keys)) != len(keys) or any(k not in valid for k in keys):
        raise ValueError('class_keys must be a nonempty unique list from the current frozen class keys')
    return keys


def _metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError('Session metadata must be an object')
    result = dict(metadata)
    for field in ['camera_id', 'oven_id', 'session_id']:
        _token(result.get(field))
    _classes(result.get('class_keys'), allow_empty=True)
    samples = result.get('physical_sample_ids')
    if not isinstance(samples, list) or (not samples and result['class_keys']) or any(not isinstance(s, str) for s in samples) or len(set(samples)) != len(samples):
        raise ValueError('physical_sample_ids must identify every physical item in the frame')
    for sample in samples:
        _token(sample)
    for field in ['placement', 'lighting']:
        value = result.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > 300 or '\n' in value:
            raise ValueError(f'Missing/invalid {field}')
    try:
        captured = datetime.fromisoformat(result['captured_at'])
        if captured.tzinfo is None or captured.utcoffset() is None:
            raise ValueError('Timezone missing')
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('captured_at must be an operator-supplied ISO8601 time with timezone') from exc
    result.setdefault('license', 'unknown')
    if not isinstance(result['license'], str) or not result['license'].strip():
        raise ValueError('license must be recorded explicitly or unknown')
    result.setdefault('camera_driver', 'UNKNOWN_NOT_PROBED')
    return result


def _facts(data):
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError('Image exceeds the 64 MiB import limit')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                fmt = image.format
                if fmt not in {'JPEG', 'PNG'}:
                    raise ValueError('Only fully decoded JPEG/PNG files are accepted')
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                return {**image_facts(image), 'format': fmt}
    except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError(f'Unreadable/unsupported image: {type(exc).__name__}') from exc


def _write_json(path, row, *, replace=False):
    path = Path(path)
    _protect_project(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise ValueError('Output already exists; refusing to overwrite')
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(row, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _store(data, collection, metadata, *, representation, source_name, camera_details=None):
    metadata = _metadata(metadata)
    facts = _facts(data)
    collection = Path(collection).resolve()
    _protect_project(collection)
    _check_layout(collection)
    image_id = uuid.uuid4().hex
    suffix = '.png' if facts['format'] == 'PNG' else '.jpg'
    relative = 'images/' + image_id + suffix
    target = collection / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(data)
    row = {**metadata, **facts, 'schema_version': 'oven-capture/v1', 'image_id': image_id,
           'relative_path': relative, 'sha256': hash_file(target), 'imported_at': now(),
           'source_name': source_name, 'representation': representation, 'sensor_raw_claimed': False,
           'label_status': 'pending', 'label_source': 'operator_candidate_not_ground_truth',
           'candidate_class_keys': metadata['class_keys'], 'camera_details': camera_details or {}}
    _write_json(collection / 'records' / (image_id + '.json'), row)
    return row


def import_files(files, collection, metadata):
    """Copy encoded file bytes; timestamp is supplied by the operator, not inferred."""
    metadata = _metadata(metadata)
    if not files:
        raise ValueError('At least one explicitly selected source file is required')
    rows = []
    for source in files:
        source = Path(source)
        if not source.is_file() or source.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError('Missing/non-file/oversized source')
        rows.append(_store(source.read_bytes(), collection,
            {**metadata, 'timestamp_source': 'operator_supplied'},
            representation='imported_encoded_file', source_name=source.name))
    return rows


def capture_frames(source, collection, metadata, *, count=1, interval=0.):
    """The caller owns/explicitly selects a FrameSource; no discovery/open occurs here."""
    if type(count) is not int or not 1 <= count <= 100:
        raise ValueError('Frame count must be an integer from 1 to 100')
    if not math.isfinite(interval) or not 0 <= interval <= 60:
        raise ValueError('Interval must be finite and between 0 and 60 seconds')
    metadata = _metadata({**metadata, 'captured_at': now()})
    rows = []
    for index in range(count):
        image, details = source.read()
        captured = now()  # Host frame-read completion time, never a sensor exposure timestamp.
        if not isinstance(image, Image.Image):
            raise NotReady('FrameSource returned no decoded image')
        stream = io.BytesIO()
        image.save(stream, format='PNG')
        rows.append(_store(stream.getvalue(), collection,
            {**metadata, 'captured_at': captured, 'timestamp_source': 'host_utc_at_frame_read'},
            representation='decoded_frame_png', source_name='selected_camera_decoded_frame', camera_details=details))
        if index + 1 < count and interval:
            time.sleep(interval)
    return rows


def _image_path(collection, row):
    relative = row.get('relative_path', '')
    if not isinstance(relative, str) or not relative.startswith('images/') or '\\' in relative or ':' in relative or '..' in Path(relative).parts:
        raise ValueError('Invalid image path in metadata')
    target = (collection / relative).resolve()
    if not target.is_relative_to(collection) or not target.is_relative_to((collection / 'images').resolve()):
        raise ValueError('Image path escapes collection')
    return target


def _check_layout(collection):
    for directory in ['images', 'records']:
        if not (collection / directory).resolve().is_relative_to(collection):
            raise ValueError('Collection subdirectory symlink escapes collection')


def _records(collection):
    _check_layout(collection)
    rows, issues = [], []
    for path in sorted((collection / 'records').glob('*.json')):
        try:
            row = _metadata(json.loads(path.read_text(encoding='utf-8')))
            if row.get('schema_version') != 'oven-capture/v1' or _token(row.get('image_id')) != path.stem:
                raise ValueError('Schema/image identity mismatch')
            _image_path(collection, row)
            if row.get('label_status') not in {'pending', 'confirmed'}:
                raise ValueError('Invalid label status')
            if row['label_status'] == 'confirmed':
                _token(row.get('reviewer'))
                if row.get('label_source') != 'human_review':
                    raise ValueError('Confirmed labels need an explicit human review record')
            rows.append(row)
        except (OSError, ValueError, TypeError) as exc:
            issues.append({'code': 'invalid_metadata', 'file': path.name, 'reason': str(exc), 'severity': 'error'})
    return rows, issues


def _collections(collection):
    paths = collection if isinstance(collection, (list, tuple)) else [collection]
    paths = [Path(path).resolve() for path in paths]
    if not paths or len(set(paths)) != len(paths) or any(not path.is_dir() for path in paths):
        raise ValueError('Explicit collections must exist and be unique')
    return paths


def audit(collection, *, near_threshold=6):
    collections = _collections(collection)
    if type(near_threshold) is not int or not 0 <= near_threshold <= 16:
        raise ValueError('pHash threshold must be an integer from 0 to 16')
    located, issues = [], []
    for directory in collections:
        rows, errors = _records(directory)
        located.extend((directory, row) for row in rows)
        issues.extend(errors)
        known = {_image_path(directory, row) for row in rows}
        for path in sorted((directory / 'images').glob('*')):
            if path.is_file() and path.resolve() not in known:
                issues.append({'code': 'missing_metadata', 'file': path.name, 'severity': 'error'})
    valid, identities = [], set()
    for collection, row in located:
        path = _image_path(collection, row)
        if row['image_id'] in identities:
            issues.append({'code': 'duplicate_image_id', 'image_id': row['image_id'], 'severity': 'error'})
            continue
        identities.add(row['image_id'])
        try:
            facts = _facts(path.read_bytes())
            actual_sha = hash_file(path)
            if row.get('sha256') != actual_sha:
                issues.append({'code': 'sha256_mismatch', 'image_id': row['image_id'], 'severity': 'error'})
            if any(row.get(k) != facts[k] for k in ['width', 'height', 'format', 'pixel_sha256', 'phash']):
                issues.append({'code': 'image_metadata_mismatch', 'image_id': row['image_id'], 'severity': 'error'})
            valid.append({**row, **facts, 'sha256': actual_sha, 'source_id': 'oven_capture', 'original_split': 'unassigned'})
        except (OSError, ValueError) as exc:
            issues.append({'code': 'unreadable_image', 'image_id': row['image_id'], 'reason': str(exc), 'severity': 'error'})
        if row['label_status'] != 'confirmed':
            issues.append({'code': 'label_pending', 'image_id': row['image_id'], 'severity': 'review'})
    if not located:
        issues.append({'code': 'empty_collection', 'severity': 'error'})
    label_sets = defaultdict(set)
    for row in valid:
        if row['label_status'] == 'confirmed':
            label_sets[row['pixel_sha256']].add(tuple(sorted(row['class_keys'])))
    for digest, labels in label_sets.items():
        if len(labels) > 1:
            issues.append({'code': 'label_conflict', 'pixel_sha256': digest, 'severity': 'error'})
    union = UnionFind(row['image_id'] for row in valid)
    seen = {}
    for row in valid:
        keys = [('sample', value) for value in row['physical_sample_ids']]
        keys += [('session', row['session_id']), ('sha', row['sha256']), ('pixels', row['pixel_sha256'])]
        for key in keys:
            if key in seen:
                union.join(row['image_id'], seen[key])
            seen[key] = row['image_id']
    tree, pairs = BKTree(), []
    for row in valid:
        value = int(row['phash'], 16)
        for previous, distance in tree.find(value, near_threshold):
            pairs.append({'left': previous, 'right': row['image_id'], 'distance': distance, 'status': 'candidate_requires_review'})
            union.join(previous, row['image_id'])
        tree.add(value, row['image_id'])
    components = defaultdict(list)
    for row in valid:
        components[union.find(row['image_id'])].append(row['image_id'])
    groups = {image_id: 'oven_' + key for key, members in components.items() for image_id in members}
    return {'state': 'FAIL' if any(i['severity'] == 'error' for i in issues) else 'REVIEW_REQUIRED',
            'records': len(located), 'collections': len(collections), 'issues': issues, 'groups': groups,
            'exact_duplicates': duplicate_groups(valid, 'sha256'),
            'pixel_duplicates': duplicate_groups(valid, 'pixel_sha256'), 'near_duplicates': pairs,
            'group_scope': 'All explicitly supplied collections: transitive sample OR session OR duplicate OR pHash candidate connections; excludes external/history data',
            'near_threshold': near_threshold, 'formal_dataset_frozen': False}


def review(collection, image_id, class_keys, reviewer):
    collection = Path(collection).resolve()
    _protect_project(collection)
    _token(image_id)
    _token(reviewer)
    _classes(class_keys, allow_empty=True)
    rows, issues = _records(collection)
    if issues:
        raise ValueError('Repair invalid metadata before review')
    row = next((r for r in rows if r['image_id'] == image_id), None)
    if row is None:
        raise ValueError('Unknown image_id')
    _facts(_image_path(collection, row).read_bytes())
    row = _metadata(dict(row, class_keys=class_keys, label_status='confirmed',
                         label_source='human_review', reviewer=reviewer, reviewed_at=now()))
    _write_json(collection / 'records' / (image_id + '.json'), row, replace=True)
    return row


def export_manifest(collection, assignments, out, *, repo_root=ROOT):
    """Explicit handoff only; no split selection or modifications to frozen datasets."""
    collections = _collections(collection)
    out, repo_root = Path(out).resolve(), Path(repo_root).resolve()
    _protect_project(out)
    if not any(out.is_relative_to(path) for path in collections) or out.exists():
        raise ValueError('New manifest must stay inside a supplied collection; refusing overwrite')
    # Candidate handoff is a complete local catalog snapshot. Partial independent
    # exports would lose OR/transitive relationships across capture collections.
    capture_root = RAW_ROOT.resolve() if repo_root == ROOT else repo_root
    catalog = {path.parent.resolve() for name in ('images', 'records')
               for path in capture_root.rglob(name) if path.is_dir()}
    if any(not path.is_relative_to(capture_root) for path in set(collections) | catalog):
        raise ValueError('Collection escapes the capture catalog')
    if catalog != set(collections):
        raise ValueError('Export must include every collection in the local capture catalog; use joint --collection paths')
    result = audit(collections)
    if any(i['severity'] == 'error' for i in result['issues']):
        raise ValueError('Audit errors must be fixed before export')
    located = [(directory, row) for directory in collections for row in _records(directory)[0]]
    rows = [row for _, row in located]
    if not isinstance(assignments, dict) or set(assignments) != {r['image_id'] for r in rows}:
        raise ValueError('Explicit split assignments must cover exactly every image_id')
    config = load_classes(ROOT / 'configs/classes.json', require_frozen=True)
    by_key = {c.key: c.id for c in config.classes}
    group_splits, records = {}, []
    for collection, row in located:
        if row['label_status'] != 'confirmed' or len(row['class_keys']) != 1:
            raise ValueError('Human-confirmed single-class labels required; multi-food images have no detection boxes')
        split = assignments[row['image_id']]
        if split not in SPLITS:
            raise ValueError('Invalid split')
        group = result['groups'][row['image_id']]
        if group in group_splits and group_splits[group] != split:
            raise ValueError('Related sample/session/duplicate group cannot cross a split')
        group_splits[group] = split
        path = _image_path(collection, row)
        if not path.is_relative_to(repo_root):
            raise ValueError('Manifest image must stay inside the repository root')
        key = row['class_keys'][0]
        records.append(dict(image_id=row['image_id'], relative_path=path.relative_to(repo_root).as_posix(),
            label_id=by_key[key], class_key=key, source_id='oven_capture', group_id=group,
            split=split, license=row['license'], sha256=row['sha256']))
    temporary = out.with_name(out.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=REQUIRED_FIELDS)
            writer.writeheader()
            writer.writerows(records)
        validate_manifest(temporary, config, repo_root=repo_root, check_files=True)
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)
    return {'state': 'CANDIDATE_HANDOFF_NOT_FROZEN', 'records': len(records), 'manifest_sha256': hash_file(out)}
