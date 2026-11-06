"""Audit an oven ZIP locally and prepare an explicitly exploratory train-only set.

The source ZIP is read-only. Filename indices never establish physical identity.
Raw images and generated JSON stay in gitignored data/raw and runs directories.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import stat
import warnings
import zipfile

from PIL import Image

from scripts.audit_images import BKTree, IMAGE_EXTENSIONS, duplicate_groups, hash_file, image_facts, safe_member
from scripts.propose_groups import UnionFind

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path('runs/oven-first-round-20261106')
RAW_BASE = Path('data/raw/oven_round1_20261106')
SOURCE_ID = 'user_supplied_oven_zip_20261106'


def _checked_members(archive):
    """Bound expansion and reject Windows aliases before creating any files."""
    members = archive.infolist()
    if len(members) > 10000 or sum(m.file_size for m in members) > 2 * 1024 ** 3:
        raise ValueError('ZIP exceeds member-count or expanded-size bound')
    seen = set()
    for member in members:
        path = safe_member(member.filename)
        if not path.parts:
            raise ValueError('Empty ZIP path')
        for part in path.parts:
            if (part.endswith((' ', '.')) or any(ord(c) < 32 or c in '<>"|?*' for c in part)
                    or re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', part, re.I)):
                raise ValueError('Windows-unsafe ZIP path')
        key = path.as_posix().casefold()
        if key in seen:
            raise ValueError('Duplicate or case-colliding ZIP member')
        seen.add(key)
        if stat.S_ISLNK(member.external_attr >> 16) or member.flag_bits & 1:
            raise ValueError('Symlink or encrypted ZIP member')
        if member.file_size > 32 * 1024 ** 2 or member.file_size / max(1, member.compress_size) > 1000:
            raise ValueError('ZIP member exceeds expansion bound')
    return members


def _inside(root, path):
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError('Output escapes its allowed project directory')
    return resolved


def _group_rows(rows, pairs):
    """Unproven same-class independence is one uncertainty group, not one specimen."""
    union = UnionFind(row['image_id'] for row in rows)
    for field in ('class_key', 'sha256', 'pixel_sha256'):
        seen = {}
        for row in rows:
            value = row.get(field)
            if value:
                if value in seen:
                    union.join(row['image_id'], seen[value])
                seen[value] = row['image_id']
    for pair in pairs:
        union.join(pair['left'], pair['right'])
    components = defaultdict(list)
    for row in rows:
        components[union.find(row['image_id'])].append(row['image_id'])
    group_ids = {key: 'uncertainty_' + hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()[:24]
                 for key, ids in components.items()}
    pixel_labels = defaultdict(set)
    for row in rows:
        if row['pixel_sha256']:
            pixel_labels[row['pixel_sha256']].add(row['class_key'])
    kept = {}
    # Prefer a ZIP train copy when duplicate pixels also occur in its val/test folders.
    priority = {'train': 0, 'val': 1, 'test': 2}
    for row in sorted(rows, key=lambda r: (priority[r['original_split']], r['archive_member'])):
        row['group_id'] = group_ids[union.find(row['image_id'])]
        row['new_split'] = 'excluded'
        row['exclusion_reason'] = None
        if row['status'] != 'ok':
            row['exclusion_reason'] = 'decode_error'
        elif len(pixel_labels[row['pixel_sha256']]) > 1:
            row['exclusion_reason'] = 'conflicting_labels_for_identical_pixels'
        elif row['pixel_sha256'] in kept:
            row['exclusion_reason'] = 'duplicate_pixels'
            row['duplicate_of'] = kept[row['pixel_sha256']]
        else:
            row['new_split'] = 'train'
            kept[row['pixel_sha256']] = row['image_id']


def audit_archive(zip_path, repo_root=ROOT, *, output=None, phash_threshold=6):
    """Return (audit, rows); reject unsafe archives and never replace extracted bytes."""
    if not 0 <= phash_threshold <= 16:
        raise ValueError('pHash threshold must be in 0..16')
    repo_root = Path(repo_root).resolve()
    zip_path = Path(zip_path).resolve()
    output = Path(output) if output is not None else DEFAULT_OUTPUT
    output = output if output.is_absolute() else repo_root / output
    runs_root = _inside(repo_root, repo_root / 'runs')
    output = _inside(runs_root, output)
    zip_sha = hash_file(zip_path)
    raw_root = _inside(repo_root, repo_root / RAW_BASE)
    extraction = _inside(raw_root, raw_root / zip_sha[:16])
    rows, metadata_files = [], []
    with zipfile.ZipFile(zip_path) as archive:
        members = _checked_members(archive)
        images = []
        for member in members:
            if member.is_dir():
                continue
            path = safe_member(member.filename)
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                metadata_files.append(path.as_posix())
                continue
            if (len(path.parts) != 4 or path.parts[-3] not in {'train', 'val', 'test'}
                    or not re.fullmatch(r'[a-z][a-z0-9_]*', path.parts[-2])):
                raise ValueError('Expected images in package/{train,val,test}/class/file')
            images.append((member, path))
        if not images:
            raise ValueError('ZIP contains no supported images')
        for index, (member, path) in enumerate(sorted(images, key=lambda item: item[1].as_posix())):
            payload = archive.read(member)
            if len(payload) != member.file_size:
                raise ValueError('ZIP expanded size differs from header')
            digest = hashlib.sha256(payload).hexdigest()
            destination = _inside(extraction, extraction / path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if hash_file(destination) != digest:
                    raise FileExistsError('Extracted image differs; refusing to overwrite')
            else:
                with destination.open('xb') as stream:
                    stream.write(payload)
            image_id = hashlib.sha256((zip_sha + ':' + path.as_posix()).encode()).hexdigest()[:24]
            row = {
                'image_id': image_id, 'source_id': SOURCE_ID, 'archive_member': path.as_posix(),
                'relative_path': destination.relative_to(repo_root).as_posix(),
                'class_key': path.parts[-2], 'original_split': path.parts[-3],
                'sha256': digest, 'pixel_sha256': None, 'phash': None,
                'physical_sample_id': None, 'physical_id_evidence': 'none; filename index is not specimen identity',
                'label_status': 'source_supplied_not_independently_human_verified',
                'license': 'unknown', 'status': 'ok', 'bytes': len(payload),
            }
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error', Image.DecompressionBombWarning)
                    with Image.open(io.BytesIO(payload)) as image:
                        image.load()
                        row.update(image_facts(image))
                        row['format'] = image.format
                        row['mode'] = image.mode
                        row['metadata_keys'] = sorted(str(key) for key in image.info)
                        row['exif_tag_ids'] = sorted(image.getexif().keys())
            except (OSError, ValueError, Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
                row['status'] = 'decode_error'
                row['decode_error'] = type(error).__name__
            rows.append(row)
            if (index + 1) % 100 == 0:
                print(f'ZIP audit: {index + 1}/{len(images)} images decoded', flush=True)
    class_keys = sorted({row['class_key'] for row in rows})
    for row in rows:
        row['class_id'] = class_keys.index(row['class_key'])
    pairs, tree = [], BKTree()
    for row in rows:
        if not row['phash']:
            continue
        value = int(row['phash'], 16)
        for old, distance in tree.find(value, phash_threshold):
            pairs.append({
                'left': old['image_id'], 'right': row['image_id'], 'distance': distance,
                'same_file': old['sha256'] == row['sha256'],
                'same_pixels': old['pixel_sha256'] == row['pixel_sha256'],
                'cross_original_split': old['original_split'] != row['original_split'],
                'cross_class': old['class_key'] != row['class_key'],
                'left_member': old['archive_member'], 'right_member': row['archive_member'],
            })
        tree.add(value, row)
    exact = {field: duplicate_groups(rows, field) for field in ('sha256', 'pixel_sha256')}
    _group_rows(rows, pairs)
    class_counts = {
        key: {split: sum(r['class_key'] == key and r['original_split'] == split for r in rows)
              for split in ('train', 'val', 'test')}
        for key in class_keys
    }
    kept = [row for row in rows if row['new_split'] == 'train']
    assert len({r['pixel_sha256'] for r in kept}) == len(kept)
    summary = {
        'status': 'exploratory_fixed_configuration_only_no_formal_evaluation',
        'source': {'archive_name': zip_path.name, 'sha256': zip_sha, 'bytes': zip_path.stat().st_size,
                   'members': len(members), 'directories': sum(m.is_dir() for m in members),
                   'uncompressed_bytes': sum(m.file_size for m in members),
                   'origin': 'user-supplied ZIP; original acquisition provenance unverified',
                   'license': 'unknown', 'redistribution': 'not permitted pending rights verification'},
        'image_count': len(rows), 'class_count': len(class_keys),
        'experimental_classes': [{'id': i, 'key': key} for i, key in enumerate(class_keys)],
        'experimental_class_mapping_scope': 'this run only; does not replace configs/classes.json',
        'class_original_split_counts': class_counts,
        'matches_claimed_48_6_6_per_class': all(counts == {'train': 48, 'val': 6, 'test': 6}
                                              for counts in class_counts.values()),
        'original_split_counts': dict(sorted(Counter(r['original_split'] for r in rows).items())),
        'image_status_counts': dict(sorted(Counter(r['status'] for r in rows).items())),
        'format_counts': dict(sorted(Counter(r.get('format', 'unreadable') for r in rows).items())),
        'dimension_counts': dict(sorted(Counter(f"{r.get('width')}x{r.get('height')}" for r in rows).items())),
        'metadata_files': metadata_files,
        'image_metadata_key_counts': dict(sorted(Counter(k for r in rows for k in r.get('metadata_keys', [])).items())),
        'images_with_exif': sum(bool(r.get('exif_tag_ids')) for r in rows),
        'proven_physical_sample_count': None, 'physical_ids_available': False,
        'duplicate_counts': {field: {'groups': len(groups), 'excess_copies': sum(g['size'] - 1 for g in groups),
                                     'cross_original_split_groups': sum(g['cross_original_split'] for g in groups)}
                             for field, groups in exact.items()},
        'exact_duplicate_groups': exact, 'phash_threshold': phash_threshold,
        'near_pair_count': len(pairs),
        'near_cross_original_split_pair_count': sum(p['cross_original_split'] for p in pairs),
        'near_nonidentical_pixel_pair_count': sum(not p['same_pixels'] for p in pairs),
        'near_nonidentical_cross_original_split_pair_count': sum(not p['same_pixels'] and p['cross_original_split'] for p in pairs),
        'near_cross_class_pair_count': sum(p['cross_class'] for p in pairs),
        'near_pairs': pairs, 'uncertainty_group_count': len({r['group_id'] for r in rows}),
        'new_split_counts': dict(sorted(Counter(r['new_split'] for r in rows).items())),
        'usable_unique_pixel_counts': dict(sorted(Counter(r['class_key'] for r in kept).items())),
        'grouping_policy': 'same-class uncertainty union plus global byte/pixel duplicate and pHash candidate edges',
        'split_policy': 'train only; no validation or test; fixed exploratory fit; no tuning or formal accuracy',
        'limitations': [
            'Numeric filename suffixes are frame/file identifiers, not evidence of physical specimens.',
            'No independent physical-item count can be established from available evidence.',
            'pHash distance is a similarity candidate, not proof of shared physical identity.',
            'Conservative uncertainty groups are not verified specimen or session IDs.',
            'Source labels have not been independently human-verified.',
            'The new ZIP directory split is audited then retired for this exploratory run; old frozen MVP test is untouched.',
            'No formal held-out metric or production readiness claim is supported.',
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, data in [('data_audit.json', summary), ('candidate_rows.json', rows)]:
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--phash-threshold', type=int, default=6)
    args = parser.parse_args()
    summary, _ = audit_archive(args.zip, output=args.output, phash_threshold=args.phash_threshold)
    print(json.dumps({key: summary[key] for key in (
        'status', 'image_count', 'class_count', 'original_split_counts', 'class_original_split_counts',
        'duplicate_counts', 'near_pair_count', 'near_cross_original_split_pair_count',
        'near_nonidentical_cross_original_split_pair_count', 'new_split_counts', 'usable_unique_pixel_counts',
        'uncertainty_group_count', 'physical_ids_available')}, indent=2), flush=True)
    return 0 if all(row == 'ok' for row in summary['image_status_counts']) else 2


if __name__ == '__main__':
    raise SystemExit(main())
