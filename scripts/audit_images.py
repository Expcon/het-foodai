"""Read both source ZIPs without extraction; produce local, non-training QA assets.

Requires Pillow and NumPy in the project venv. No originals are modified.
Perceptual matches are review candidates, never automatic deletion decisions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import warnings
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
MAPPING = {
    'corn': 'corn', 'potato': 'potato', 'potatoes': 'potato',
    'broccoli': 'broccoli', 'pumpkin': 'pumpkin',
    'chicken wing': 'chicken_wing', 'chicken breast': 'chicken_breast',
}
BLOCKED = {'fish': 'sea_bass_unverified', 'shrimp': 'shrimp_unverified'}


def normalized(label):
    return ' '.join(label.lower().replace('_', ' ').split())


def safe_member(name):
    """Reject absolute, drive-qualified and traversal members even without extraction."""
    path = PurePosixPath(name.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or ':' in name:
        raise ValueError(f'Unsafe ZIP member: {name!r}')
    return path


def hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def phash(image):
    """64-bit DCT pHash (8x8 low frequencies, DC excluded from threshold)."""
    pixels = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    x = np.arange(32)
    basis = np.cos(np.pi * (2 * x[None, :] + 1) * np.arange(8)[:, None] / 64)
    values = (basis @ pixels @ basis.T).ravel()
    bits = values > np.median(values[1:])
    bits[0] = False
    return sum(int(bit) << i for i, bit in enumerate(bits))


class BKTree:
    """Hamming metric index; identical hashes share one node."""
    def __init__(self):
        self.root = None

    def add(self, value, item):
        if self.root is None:
            self.root = [value, [item], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(item)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [item], {}]
                return
            node = node[2][distance]

    def find(self, value, radius):
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                for item in node[1]:
                    yield item, distance
            pending.extend(child for edge, child in node[2].items()
                           if distance - radius <= edge <= distance + radius)


def crop_box(box, size):
    if len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
        raise ValueError('non-finite/malformed box')
    x, y, width, height = box
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > size[0] + 1e-5 or y + height > size[1] + 1e-5:
        raise ValueError('out-of-bounds/nonpositive box')
    return (math.floor(x), math.floor(y), min(size[0], math.ceil(x + width)),
            min(size[1], math.ceil(y + height)))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def image_facts(image):
    rgb = image.convert('RGB')
    gray = np.asarray(rgb.convert('L').resize((64, 64)), dtype=float)
    return {
        'width': rgb.width, 'height': rgb.height,
        'pixel_sha256': hashlib.sha256(f'{rgb.size}:'.encode() + rgb.tobytes()).hexdigest(),
        'phash': f'{phash(rgb):016x}', 'gray_std': round(float(gray.std()), 2),
        'small': min(rgb.size) < 96,
    }


def detect_pairs(rows, threshold):
    tree = BKTree()
    for row in rows:
        if not row.get('phash'):
            continue
        value = int(row['phash'], 16)
        for old, distance in tree.find(value, threshold):
            yield {'left': old['image_id'], 'right': row['image_id'],
                   'distance': distance, 'same_file': old['sha256'] == row['sha256'],
                   'same_pixels': old['pixel_sha256'] == row['pixel_sha256'],
                   'cross_source': old['source_id'] != row['source_id'],
                   'cross_original_split': (old['source_id'] == row['source_id']
                                            and old['original_split'] != row['original_split']),
                   'left_label': old.get('class_key', ''), 'right_label': row.get('class_key', '')}
        tree.add(value, row)


def duplicate_groups(rows, field):
    buckets = defaultdict(list)
    for row in rows:
        if row.get(field):
            buckets[row[field]].append(row)
    result = []
    for digest, group in buckets.items():
        if len(group) > 1:
            result.append({'digest': digest, 'size': len(group),
                           'image_ids': [r['image_id'] for r in group],
                           'source_splits': sorted({r['source_id'] + '/' + r['original_split'] for r in group}),
                           'cross_original_split': any(len({r['original_split'] for r in group if r['source_id'] == source}) > 1
                                                       for source in {r['source_id'] for r in group})})
    return result


def build_contacts(rows, root):
    """Deterministic samples spread across sources, excluding duplicate pixels."""
    buckets = defaultdict(list)
    for row in rows:
        if row.get('candidate_path'):
            buckets[row['class_key']].append(row)
    chosen_rows = []
    for label, items in sorted(buckets.items()):
        selected, seen = [], set()
        sources = sorted({r['source_id'] for r in items})
        for source in sources:
            for row in sorted((r for r in items if r['source_id'] == source), key=lambda r: r['image_id']):
                if row['pixel_sha256'] in seen:
                    continue
                seen.add(row['pixel_sha256'])
                selected.append(row)
                if sum(r['source_id'] == source for r in selected) >= max(1, 24 // len(sources)):
                    break
        sheet = Image.new('RGB', (6 * 180, 4 * 170 + 30), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f'{label} | QA CANDIDATES, NOT ACCEPTED TRAINING DATA', fill='black')
        for index, row in enumerate(selected[:24]):
            with Image.open(ROOT / row['candidate_path']) as image:
                thumb = ImageOps.contain(image.convert('RGB'), (172, 130))
                x, y = index % 6 * 180 + 4, index // 6 * 170 + 30
                sheet.paste(thumb, (x, y))
                draw.text((x, y + 131), f'{index + 1}: {row["image_id"][:12]}', fill='black')
                draw.text((x, y + 146), row['source_id'][:24], fill='black')
            chosen_rows.append({'sheet': label, 'cell': index + 1, **row})
        sheet.save(root / f'contact_{label}.jpg', quality=90)
    write_csv(root / 'contact_index.csv', chosen_rows)


def run(args):
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'data/cache'):
        raise ValueError('Audit output must stay under project data/cache')
    output.mkdir(parents=True, exist_ok=True)
    candidates = ROOT / 'data/processed/audit_candidates_v1'
    rows, samples, issues, packages, annotation_counts = [], [], [], [], []
    for source, path in [('kaggle_vegetable_v1', args.kaggle), ('roboflow_fridge_genie_v1', args.roboflow)]:
        print(f'Auditing {source}: {path.name}', flush=True)
        package = {'source_id': source, 'archive': str(path.relative_to(ROOT)),
                   'sha256': hash_file(path), 'bytes': path.stat().st_size}
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(set(names)) != len(names):
                raise ValueError('Duplicate ZIP member names')
            for name in names:
                safe_member(name)
            package['members'] = len(names)
            package['uncompressed_bytes'] = sum(i.file_size for i in archive.infolist())
            package['readme'] = {}
            metadata = {}
            annotations = defaultdict(list)
            for name in names:
                if name.lower().endswith(('.txt', '.md')):
                    package['readme'][name] = archive.read(name).decode('utf-8', errors='replace')
                elif name.endswith('.coco.json'):
                    data = json.loads(archive.read(name))
                    split = PurePosixPath(name).parent.as_posix()
                    categories = {c['id']: c['name'] for c in data['categories']}
                    image_map = {i['id']: i for i in data['images']}
                    package.setdefault('coco', []).append({'split': split, 'images': len(image_map),
                                                          'annotations': len(data['annotations']),
                                                          'categories': data['categories'], 'licenses': data.get('licenses'),
                                                          'info': data.get('info')})
                    by_category = defaultdict(list)
                    for ann in data['annotations']:
                        if ann['image_id'] not in image_map:
                            raise ValueError('COCO annotation refers to missing image')
                        label = categories[ann['category_id']]
                        by_category[label].append(ann)
                        member = str(PurePosixPath(split) / image_map[ann['image_id']]['file_name'])
                        annotations[member].append({**ann, 'source_label': label})
                    for label, group in by_category.items():
                        annotation_counts.append({'source_id': source, 'split': split, 'label': label,
                                                  'boxes': len(group), 'images': len({a['image_id'] for a in group})})
                    for image in data['images']:
                        member = str(PurePosixPath(split) / image['file_name'])
                        if member not in names:
                            raise ValueError(f'Missing COCO image {member}')
                        metadata[member] = image
            image_names = sorted(n for n in names if PurePosixPath(n).suffix.lower() in IMAGE_EXTENSIONS)
            for index, name in enumerate(image_names):
                image_id = hashlib.sha256((source + ':' + name).encode()).hexdigest()[:24]
                parts = PurePosixPath(name).parts
                label = parts[-2] if source.startswith('kaggle') else ''
                key = MAPPING.get(normalized(label), '')
                row = {'image_id': image_id, 'source_id': source, 'archive_member': name,
                       'original_split': parts[-3] if source.startswith('kaggle') else parts[0],
                       'source_label': label, 'class_key': key, 'sha256': '', 'status': 'unread'}
                try:
                    content = archive.read(name)  # CRC checked by ZipFile on full read.
                    row['sha256'] = hashlib.sha256(content).hexdigest()
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', Image.DecompressionBombWarning)
                        with Image.open(io.BytesIO(content)) as original:
                            original.load()
                            image = original.convert('RGB')
                    row.update(image_facts(image))
                    row['status'] = 'readable'
                    if source.startswith('kaggle') and key:
                        destination = candidates / key / f'{image_id}.jpg'
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if destination.exists() and destination.read_bytes() != content:
                            raise ValueError('Candidate output collision')
                        destination.write_bytes(content)
                        samples.append({**row, 'parent_image_id': image_id, 'candidate_path': destination.relative_to(ROOT).as_posix(),
                                        'review_status': 'needs_semantic_review', 'decision': '', 'review_note': ''})
                    if name in metadata:
                        m = metadata[name]
                        if (m['width'], m['height']) != image.size:
                            issues.append({'image_id': image_id, 'reason': 'COCO dimension mismatch'})
                            rows.append(row)
                            continue
                        for ann in annotations[name]:
                            source_label = ann['source_label']
                            mapped = MAPPING.get(normalized(source_label)) or BLOCKED.get(normalized(source_label))
                            if not mapped:
                                continue
                            try:
                                bounds = crop_box(ann['bbox'], image.size)
                            except ValueError as error:
                                issues.append({'image_id': image_id, 'annotation_id': ann['id'], 'reason': str(error)})
                                continue
                            crop = image.crop(bounds)
                            crop_id = hashlib.sha256(f'{image_id}:{ann["id"]}:{bounds}'.encode()).hexdigest()[:24]
                            destination = candidates / mapped / f'{crop_id}.png'
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            crop.save(destination)
                            samples.append({**row, **image_facts(crop), 'image_id': crop_id, 'parent_image_id': image_id,
                                            'source_label': source_label, 'class_key': mapped, 'bbox': json.dumps(ann['bbox']),
                                            'sha256': hash_file(destination), 'annotation_id': ann['id'],
                                            'candidate_path': destination.relative_to(ROOT).as_posix(),
                                            'review_status': 'blocked_taxonomy' if mapped in BLOCKED.values() else 'needs_semantic_review',
                                            'decision': '', 'review_note': ''})
                except (OSError, ValueError, zipfile.BadZipFile, Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
                    row['status'] = 'error'
                    issues.append({'image_id': image_id, 'reason': f'{type(error).__name__}: {error}'})
                rows.append(row)
                if (index + 1) % 2000 == 0:
                    print(f'  {index + 1}/{len(image_names)} decoded', flush=True)
            package['images'] = len(image_names)
            packages.append(package)
    write_csv(output / 'images.csv', rows)
    write_csv(output / 'human_review.csv', samples)
    write_csv(output / 'issues.csv', issues, ['image_id', 'annotation_id', 'reason'])
    write_csv(output / 'annotation_counts.csv', annotation_counts)
    print('Finding original-image perceptual matches (review candidates only)...', flush=True)
    pair_fields = ['left', 'right', 'distance', 'same_file', 'same_pixels', 'cross_source',
                   'cross_original_split', 'left_label', 'right_label']
    pair_counts = Counter()
    # Stream large pair files rather than materializing quadratic duplicate groups.
    for category, collection in [('original', rows), ('candidate', samples)]:
        with (output / f'{category}_near_pairs.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=pair_fields)
            writer.writeheader()
            for pair in detect_pairs(collection, args.phash_threshold):
                writer.writerow(pair)
                pair_counts[category + '_pairs'] += 1
                if pair['cross_original_split']:
                    pair_counts[category + '_cross_original_split_pairs'] += 1
                if pair['cross_source']:
                    pair_counts[category + '_cross_source_pairs'] += 1
    exact = {field: duplicate_groups(rows, field) for field in ['sha256', 'pixel_sha256']}
    (output / 'original_exact_groups.json').write_text(json.dumps(exact, indent=2), encoding='utf-8')
    summary = {'status': 'audit_only_not_training_ready', 'packages': packages,
               'image_counts_by_source': dict(Counter(r['source_id'] for r in rows)),
               'image_status': dict(Counter(r['status'] for r in rows)),
               'kaggle_class_counts': dict(Counter(r['source_label'] for r in rows if r['source_id'].startswith('kaggle'))),
               'candidate_counts': dict(Counter(r['class_key'] for r in samples)),
               'candidate_unique_pixels': {k: len({r['pixel_sha256'] for r in samples if r['class_key'] == k})
                                           for k in sorted({r['class_key'] for r in samples})},
               'candidate_parent_images': {k: len({r['parent_image_id'] for r in samples if r['class_key'] == k})
                                            for k in sorted({r['class_key'] for r in samples})},
               'issues': len(issues), 'phash_threshold': args.phash_threshold,
               'pair_counts': dict(pair_counts),
               'original_exact_group_counts': {k: len(v) for k, v in exact.items()},
               'original_exact_excess_copies': {k: sum(g['size'] - 1 for g in v) for k, v in exact.items()},
               'original_exact_cross_split_groups': {k: sum(g['cross_original_split'] for g in v) for k, v in exact.items()},
               'limitations': ['No automatic semantic acceptance; all candidates require review.',
                               'Perceptual similarity is not proof of same original.',
                               'Independent capture groups cannot be inferred from filenames alone.',
                               'No frozen classes, split or formal manifest was generated.']}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    build_contacts(samples, output)
    print(json.dumps({k: v for k, v in summary.items() if k not in ['packages', 'limitations']}, indent=2), flush=True)
    print(f'QA outputs: {output}', flush=True)
    return 0 if not any(r['status'] == 'error' for r in rows) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kaggle', type=Path, default=ROOT / 'data/raw/kaggle_vegetable/archive.zip')
    parser.add_argument('--roboflow', type=Path, default=ROOT / 'data/raw/roboflow_fridge_genie/Fridge Genie.v1i.coco.zip')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/cache/audit_v1')
    parser.add_argument('--phash-threshold', type=int, default=6)
    args = parser.parse_args()
    if not 0 <= args.phash_threshold <= 16:
        parser.error('phash threshold must be in 0..16')
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
