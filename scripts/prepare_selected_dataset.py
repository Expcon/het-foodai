"""Audit human-selected images and materialize a reproducible grouped 80/10/10 dataset.

Originals stay untouched. Known pre-augmented fish photos are kept in the input,
but excluded from this base split; augmentation belongs to training only.
Taxonomy stays proposed until the team approves the shared configuration.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import shutil
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from audit_images import image_facts, hash_file, phash, write_csv
from propose_groups import UnionFind

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodai.contracts import load_classes, validate_manifest, REQUIRED_FIELDS

CLASSES = [('corn', '玉米'), ('potato', '土豆'), ('broccoli', '西兰花'),
           ('pumpkin', '南瓜'), ('chicken_wing', '鸡翅'), ('chicken_breast', '鸡胸肉'),
           ('sea_bass', '鲈鱼'), ('shrimp', '虾')]
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def near_pairs(rows, threshold=6):
    """Exact Hamming-radius retrieval using threshold+1 disjoint bit blocks.

    Pigeonhole principle: at most threshold changed bits leave one block equal.
    This avoids quadratic all-pairs comparison while retaining every radius hit.
    """
    blocks = threshold + 1
    widths = [64 // blocks + (i < 64 % blocks) for i in range(blocks)]
    buckets = [defaultdict(list) for _ in widths]
    previous = []
    for row in rows:
        value = int(row['canonical_phash'], 16)
        offset = 0
        keys = []
        hits = set()
        for index, width in enumerate(widths):
            key = (value >> offset) & ((1 << width) - 1)
            keys.append(key)
            hits.update(buckets[index].get(key, []))
            offset += width
        for index in sorted(hits):
            other_value, other = previous[index]
            distance = (value ^ other_value).bit_count()
            if distance <= threshold:
                yield other['image_id'], row['image_id'], distance
        for index, key in enumerate(keys):
            buckets[index][key].append(len(previous))
        previous.append((value, row))


def stratified_groups(rows, seed=42):
    """Assign global multi-class groups, then improve integer class balance."""
    groups = defaultdict(list)
    totals = Counter(r['class_key'] for r in rows)
    for row in rows:
        groups[row['group_id']].append(row)
    vectors = {g: Counter(r['class_key'] for r in members) for g, members in groups.items()}
    splits = ['train', 'val', 'test']
    targets = {s: {k: totals[k] * ratio for k in totals} for s, ratio in zip(splits, [.8, .1, .1])}
    counts = {s: Counter() for s in splits}
    rng = random.Random(seed)
    tie = {g: rng.random() for g in sorted(groups)}
    ordered = sorted(groups, key=lambda g: (-max(n / totals[k] for k, n in vectors[g].items()), -len(groups[g]), tie[g]))

    def delta(split, vector, sign):
        return sum(((counts[split][key] + sign * n - targets[split][key]) ** 2 -
                    (counts[split][key] - targets[split][key]) ** 2) / max(1, targets[split][key])
                   for key, n in vector.items())

    assignments = {}
    for group in ordered:
        vector = vectors[group]
        choice = min(splits, key=lambda s: (delta(s, vector, 1), splits.index(s)))
        assignments[group] = choice
        counts[choice].update(vector)
    for _ in range(30):
        improved = False
        for group in ordered:
            old = assignments[group]
            vector = vectors[group]
            # Preserve existing per-class coverage in every split.
            if any(counts[old][key] - n <= 0 for key, n in vector.items()):
                continue
            choices = [(delta(old, vector, -1) + delta(new, vector, 1), new)
                       for new in splits if new != old]
            change, new = min(choices)
            if change < -1e-9:
                counts[old].subtract(vector)
                counts[new].update(vector)
                assignments[group] = new
                improved = True
        if not improved:
            break
    missing = [(s, k) for s in splits for k in totals if counts[s][k] == 0]
    if missing:
        raise ValueError(f'Not enough separable groups for coverage: {missing}')
    return assignments


def source_index(cache):
    index = defaultdict(list)
    old_rows = read_csv(ROOT / 'data/cache/audit_v1/images.csv')
    old_union = UnionFind(r['image_id'] for r in old_rows)
    for r in old_rows:
        index[r['sha256']].append({'source_id': r['source_id'], 'source_member': r['archive_member'],
                                  'origin_id': r['image_id'], 'role': 'original',
                                  'license': 'CC-BY-SA-4.0' if r['source_id'].startswith('kaggle') else 'CC-BY-4.0'})
    for pair in read_csv(ROOT / 'data/cache/audit_v1/original_near_pairs.csv'):
        old_union.join(pair['left'], pair['right'])
    for values in index.values():
        for item in values:
            item['origin_group'] = 'prior_' + old_union.find(item['origin_id'])
    fish_zip = next((ROOT / 'data/raw/large_scale_fish').glob('*.zip'))
    digest = hash_file(fish_zip)
    fish_cache = cache / 'fish_source_index.json'
    if fish_cache.exists() and (stored := json.loads(fish_cache.read_text(encoding='utf-8')))['zip_sha256'] == digest:
        fish_items = stored['items']
    else:
        fish_items = []
        with zipfile.ZipFile(fish_zip) as archive:
            names = [n for n in archive.namelist() if PurePosixPath(n).suffix.lower() in EXTENSIONS
                     and ('/Sea Bass/' in n or '/Shrimp/' in n)]
            for number, name in enumerate(names):
                role = 'nonaugmented_candidate' if name.startswith('NA_Fish_Dataset/') else (
                    'segmentation_mask' if PurePosixPath(name).parent.name.endswith(' GT') else 'preaugmented')
                fish_items.append({'sha256': hashlib.sha256(archive.read(name)).hexdigest(),
                                   'source_id': 'large_scale_fish', 'source_member': name,
                                   'role': role, 'license': 'CC-BY-4.0'})
                if (number + 1) % 1000 == 0:
                    print(f'Fish source index {number + 1}/{len(names)}', flush=True)
        fish_cache.write_text(json.dumps({'zip_sha256': digest, 'items': fish_items}, indent=2), encoding='utf-8')
    for item in fish_items:
        index[item['sha256']].append(item)
    return index, digest


def deduplicate(rows):
    """Quarantine conflicting labels; retain one representative per exact pixel set."""
    buckets = defaultdict(list)
    for row in rows:
        if row['status'] == 'candidate':
            buckets[row['pixel_sha256']].append(row)
    for members in buckets.values():
        if len({r['class_key'] for r in members}) > 1:
            for row in members:
                row['status'] = 'label_conflict'
        else:
            for row in sorted(members, key=lambda r: r['input_path'])[1:]:
                row['status'] = 'exact_duplicate'
    return [r for r in rows if r['status'] == 'candidate']


def run(args):
    selected = args.input.resolve()
    output = (ROOT / 'data/processed' / args.version).resolve()
    if not output.is_relative_to(ROOT / 'data/processed') or output.exists():
        raise ValueError('Use a new, project-local dataset output version; existing output will not be overwritten')
    if (ROOT / 'data/manifest.csv').exists():
        raise ValueError('A manifest already exists; do not silently replace a previous split')
    cache = ROOT / 'data/cache' / args.version
    cache.mkdir(parents=True, exist_ok=True)
    index, fish_zip_hash = source_index(cache)
    rows = []
    for label_id, (key, zh) in enumerate(CLASSES):
        print(f'Auditing selected class {key}', flush=True)
        for path in sorted((selected / zh).rglob('*')):
            if not path.is_file():
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(selected):
                raise ValueError(f'Input path escapes selected root: {path}')
            relative = path.relative_to(ROOT).as_posix()
            row = {'image_id': hashlib.sha256(relative.encode()).hexdigest()[:24], 'input_path': relative,
                   'label_id': label_id, 'class_key': key, 'status': 'candidate', 'warning': ''}
            if path.suffix.lower() not in EXTENSIONS:
                row['status'] = 'not_image'
                rows.append(row)
                continue
            row['sha256'] = hash_file(path)
            matches = index.get(row['sha256'], [])
            priority = {'nonaugmented_candidate': 0, 'original': 1, 'preaugmented': 2, 'segmentation_mask': 3}
            provenance = min(matches, key=lambda m: (priority[m['role']], m['source_id'], m['source_member'])) if matches else {}
            row.update({field: provenance.get(field, '') for field in ['source_member', 'origin_group', 'role']})
            row['source_id'] = provenance.get('source_id', 'manual_source_unresolved')
            row['license'] = provenance.get('license', 'unknown')
            row['all_source_matches'] = json.dumps(matches, ensure_ascii=False)
            try:
                with Image.open(path) as original:
                    original.load()
                    image = ImageOps.exif_transpose(original).convert('RGB')
                row.update(image_facts(image))
                if provenance.get('role') == 'segmentation_mask':
                    row['status'] = 'segmentation_mask'
                elif provenance.get('role') == 'preaugmented':
                    row['status'] = 'preaugmented_held_out_of_base_split'
                else:
                    variants = [image, image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)]
                    hashes = [phash(im.transpose(rotation)) for im in variants for rotation in
                              [Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270]]
                    hashes.extend(phash(im) for im in variants)
                    row['canonical_phash'] = f'{min(hashes):016x}'
                if min(image.size) < 96:
                    row['warning'] = 'short_side_under_96_review'
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                row['status'] = 'unreadable'
                row['warning'] = f'{type(error).__name__}: {error}'
            rows.append(row)
    kept = deduplicate(rows)
    if any(not any(r['class_key'] == key for r in kept) for key, _ in CLASSES):
        write_csv(cache / 'audit.csv', rows)
        raise ValueError('Some classes have no usable non-augmented images; see audit.csv')
    union = UnionFind(r['image_id'] for r in kept)
    prior = {}
    for row in kept:
        if row['origin_group']:
            if row['origin_group'] in prior:
                union.join(row['image_id'], prior[row['origin_group']])
            prior[row['origin_group']] = row['image_id']
    pairs = []
    print(f'Grouping {len(kept)} base candidates', flush=True)
    for left, right, distance in near_pairs(kept):
        union.join(left, right)
        pairs.append({'left': left, 'right': right, 'distance': distance})
    components = defaultdict(list)
    for row in kept:
        components[union.find(row['image_id'])].append(row['image_id'])
    group_ids = {key: 'group_' + hashlib.sha256('\n'.join(sorted(values)).encode()).hexdigest()[:24]
                 for key, values in components.items()}
    for row in kept:
        row['group_id'] = group_ids[union.find(row['image_id'])]
    assignments = stratified_groups(kept, args.seed)
    for row in kept:
        row['split'] = assignments[row['group_id']]
    output.mkdir(parents=True)
    for split in ['train', 'val', 'test']:
        for _, zh in CLASSES:
            (output / split / zh).mkdir(parents=True)
    manifest = []
    for row in kept:
        source = ROOT / row['input_path']
        destination = output / row['split'] / CLASSES[row['label_id']][1] / (row['image_id'] + source.suffix.lower())
        shutil.copy2(source, destination)
        if hash_file(destination) != row['sha256']:
            raise ValueError('Copied image hash changed')
        manifest.append({field: (destination.relative_to(ROOT).as_posix() if field == 'relative_path' else row[field])
                         for field in REQUIRED_FIELDS})
    classes_file = output / 'classes.proposed.json'
    classes_file.write_text(json.dumps({'version': args.version, 'status': 'draft',
                                       'notes': 'User approved general shrimp; ordered class IDs unchanged. Team freeze pending.',
                                       'classes': [{'id': i, 'key': key, 'zh': zh} for i, (key, zh) in enumerate(CLASSES)]},
                                      ensure_ascii=False, indent=2), encoding='utf-8')
    write_csv(output / 'manifest.csv', manifest, list(REQUIRED_FIELDS))
    _, validation = validate_manifest(output / 'manifest.csv', load_classes(classes_file), repo_root=ROOT, check_files=True)
    # Independently verify global groups across sources, stronger than source/group contract.
    for field in ['group_id', 'sha256']:
        values = defaultdict(set)
        for row in manifest:
            values[row[field]].add(row['split'])
        if any(len(splits) > 1 for splits in values.values()):
            raise ValueError(f'Cross-split leakage: {field}')
    lookup = {r['image_id']: r for r in kept}
    assert all(lookup[p['left']]['split'] == lookup[p['right']]['split'] for p in pairs)
    shutil.copy2(output / 'manifest.csv', ROOT / 'data/manifest.csv')
    write_csv(cache / 'audit.csv', rows)
    write_csv(cache / 'excluded.csv', [r for r in rows if r['status'] != 'candidate'])
    write_csv(cache / 'near_pairs.csv', pairs, ['left', 'right', 'distance'])
    write_csv(output / 'provenance.csv', [{**r} for r in kept])
    class_groups = {key: len({r['group_id'] for r in kept if r['class_key'] == key}) for key, _ in CLASSES}
    summary = {'version': args.version, 'seed': args.seed, 'target_ratio': [.8, .1, .1],
               'status': 'split_generated_taxonomy_team_freeze_pending', 'inputs': len(rows),
               'input_sha256': hashlib.sha256('\n'.join(sorted(r['input_path'] + ':' + r.get('sha256', '') for r in rows)).encode()).hexdigest(),
               'fish_zip_sha256': fish_zip_hash, 'audit_status': dict(Counter(r['status'] for r in rows)),
               'counts': validation, 'class_groups': class_groups, 'near_pairs': len(pairs),
               'global_cross_split_group_leakage': 0, 'detected_near_pair_cross_split_leakage': 0,
               'source_counts': dict(Counter(r['source_id'] for r in kept)),
               'quality_warnings': sum(bool(r['warning']) for r in kept),
               'limitations': ['Perceptual grouping conservative; unknown capture groups may be missed.',
                               'Preaugmented images retained in input but excluded from base split.',
                               'Manual class labels accepted, not a biological species verification.',
                               'No external-source independent test, no model training or accuracy claim.',
                               'Formal taxonomy freeze requires team approval.']}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    for key, zh in CLASSES:
        members = sorted((r for r in kept if r['class_key'] == key), key=lambda r: r['image_id'])[:12]
        sheet = Image.new('RGB', (720, 400), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(members):
            with Image.open(ROOT / row['input_path']) as im:
                thumb = ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'), (170, 105))
                x, y = i % 4 * 180 + 4, i // 4 * 130 + 4
                sheet.paste(thumb, (x, y))
                draw.text((x, y + 107), f'{i+1}: {row["image_id"][:12]}', fill='black')
        sheet.save(cache / f'contact_{key}.jpg', quality=90)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/raw/人工筛选')
    parser.add_argument('--version', default='selected_dataset_v1')
    parser.add_argument('--seed', type=int, default=42)
    sys.exit(run(parser.parse_args()))
