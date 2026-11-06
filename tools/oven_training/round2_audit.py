"""Read-only second-round oven audit; local outputs are exploratory, train-only.

Directory labels and capture scene identifiers are proposals, not verified truth.
No source image or metadata is copied or modified. Per-image outputs stay in runs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import warnings

from PIL import Image, ImageDraw, ImageOps

from scripts.audit_images import BKTree, IMAGE_EXTENSIONS, hash_file, image_facts, safe_member
from scripts.propose_groups import UnionFind
from tools.oven_training.audit import _inside

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path('runs/oven-round2-20261107/audit')
DEFAULT_OLD_ROWS = Path('runs/oven-first-round-20261106/candidate_rows.json')
CLASS_KEYS = ('asparagus', 'bacon', 'broccoli', 'chicken_wing', 'croissant',
              'french_fries', 'potato', 'salmon', 'toast', 'corn', 'sausage', 'steak')
FOLDER_CLASSES = {'芦笋': 'asparagus', '培根': 'bacon', '西兰花': 'broccoli',
                  '鸡翅': 'chicken_wing', '可颂': 'croissant', '薯条': 'french_fries',
                  '土豆': 'potato', '三文鱼': 'salmon', '吐司': 'toast',
                  '玉米': 'corn', '香肠': 'sausage', '牛排': 'steak'}


def safe_child(root, name):
    """Validate indexed paths before reads, including symlink/junction escapes."""
    root = Path(root).resolve()
    path = safe_member(name)
    if not path.parts:
        raise ValueError('Empty indexed path')
    result = root.joinpath(*path.parts).resolve()
    if result == root or not result.is_relative_to(root):
        raise ValueError('Indexed path escapes its source root')
    return result


def label_evidence(class_key, relative_path, sidecar, organizer_reviewed_photos):
    context = sidecar.get('camera_context', {})
    candidate = context.get('candidate_class_key_unreviewed')
    confirmed = context.get('confirmed_class_key', sidecar.get('confirmed_class_key'))
    manual = (context.get('human_label_confirmed') is True and confirmed == class_key
              and sidecar.get('label_status') in {'human_confirmed', 'human_reviewed', 'confirmed', 'verified'})
    photo = (organizer_reviewed_photos and class_key == 'toast'
             and re.fullmatch(r'photo (mid|up)_\d{4}\.png', Path(relative_path).name) is not None)
    return {
        'trusted_label': bool(manual or photo),
        'label_status': ('human_reviewed_from_organizer' if photo else
                         'human_confirmed_sidecar' if manual else 'candidate_directory'),
        'label_evidence': ('Organizer README states each of the 23 photo mid/up images was reviewed as toast; '
                           'source sidecar remains pending; not a formal test label' if photo else
                           'Explicit matching confirmed class and human confirmation in reviewed sidecar' if manual else
                           'Directory mapping and organizer representative review only; exploratory weak label'),
        'source_label_status': sidecar.get('label_status', 'missing'),
        'source_human_label_confirmed': context.get('human_label_confirmed'),
        'candidate_class_key_unreviewed': candidate,
        'candidate_disagrees_with_directory': candidate is not None and candidate != class_key,
    }


def scene_evidence(sidecar, filename):
    context = sidecar.get('camera_context', {})
    filename_prefix = re.sub(r'_\d+\.[^.]+$', '', filename)
    prefix = sidecar.get('sample_prefix') or filename_prefix
    raw = {'sample_prefix': sidecar.get('sample_prefix'), 'filename_prefix': filename_prefix}
    tokens = set(re.split(r'[-_\s]+', (prefix + ' ' + filename_prefix).lower()))
    foil_values = {'absent' if 'without_foil' in text.lower() else 'present'
                   for text in (prefix, filename_prefix)
                   if 'without_foil' in text.lower() or 'with_foil' in text.lower()}
    foil = next(iter(foil_values)) if len(foil_values) == 1 else 'unknown'
    levels = tokens & {'high', 'mid', 'low', 'up'}
    shelf = next(iter(levels)) if len(levels) == 1 else 'unknown'
    physical = context.get('physical_sample_id') or 'unknown'
    return {
        'physical_sample_id': physical, 'session_id': context.get('session_id') or 'unknown',
        'oven_id': context.get('oven_id') or 'unknown',
        'physical_identity_verified': False,
        'physical_id_evidence': 'Unverified capture scene identifier; changing shelf/foil prefix does not establish another object',
        'session_id_evidence': 'Source camera_context.session_id' if context.get('session_id') else 'missing',
        'oven_id_evidence': 'Source explicit oven_id' if context.get('oven_id') else 'No oven identity; camera alias is not oven identity',
        'sample_prefix_raw': prefix,
        'foil': {'value': foil, 'raw': raw,
                 'evidence': 'Explicit filename token only' if foil != 'unknown' else 'have/no tokens are ambiguous; no verified foil annotation'},
        'shelf': {'value': shelf, 'raw': raw,
                  'evidence': 'Filename/sidecar level token, unverified' if shelf != 'unknown' else
                              'conflicting level tokens' if len(levels) > 1 else 'missing or ambiguous level token'},
        'angle': {'value': 'unknown', 'raw': context.get('angle'), 'evidence': 'No verified viewing angle'},
        'lighting': {'value': context.get('lighting') or 'unknown', 'raw': context.get('lighting'),
                     'evidence': 'Source camera_context; not calibrated' if context.get('lighting') else 'missing'},
    }


def read_image(path, row):
    row.update({'sha256': hash_file(path), 'bytes': path.stat().st_size, 'status': 'ok',
                'pixel_sha256': None, 'phash': None, 'width': None, 'height': None, 'format': None})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image.load()
                row.update(image_facts(image))
                row.update({'format': image.format, 'mode': image.mode,
                            'metadata_keys': sorted(str(k) for k in image.info),
                            'exif_tag_ids': sorted(image.getexif().keys())})
    except (OSError, ValueError, Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        row.update({'status': 'decode_error', 'decode_error': type(error).__name__})


def group_rows(rows, pairs):
    """Join actual graph edges; unknown same-class independence stays conservative."""
    union = UnionFind(r['image_id'] for r in rows)
    edge_counts = Counter()
    seen = {}
    for row in rows:
        session = row.get('session_id', 'unknown')
        physical = row.get('physical_sample_id', 'unknown')
        keys = [('sha256', row.get('sha256')), ('pixel_sha256', row.get('pixel_sha256'))]
        if session != 'unknown':
            keys.append(('session', session))
        if session != 'unknown' and physical != 'unknown':
            keys.append(('session_physical', (session, physical)))
        if not row.get('physical_identity_verified', False):
            keys.append(('unknown_same_class', row['class_key']))
        for kind, value in keys:
            if value:
                key = (kind, value)
                if key in seen:
                    union.join(row['image_id'], seen[key])
                    edge_counts[kind] += 1
                else:
                    seen[key] = row['image_id']
    for pair in pairs:
        union.join(pair['left'], pair['right'])
        edge_counts['phash_candidate'] += 1
    components, pixel_labels = defaultdict(list), defaultdict(set)
    for row in rows:
        components[union.find(row['image_id'])].append(row['image_id'])
        if row.get('pixel_sha256'):
            pixel_labels[row['pixel_sha256']].add(row['class_key'])
    ids = {key: 'uncertainty_' + hashlib.sha256('\n'.join(sorted(members)).encode()).hexdigest()[:24]
           for key, members in components.items()}
    kept = {}
    for row in sorted(rows, key=lambda r: (not r.get('trusted_label', False), r['source'] != 'new', r['relative_path'])):
        row.update({'group_id': ids[union.find(row['image_id'])], 'new_split': 'excluded',
                    'exclusion_reason': None, 'duplicate_of': None})
        if row['status'] != 'ok':
            row['exclusion_reason'] = 'decode_error'
        elif row.get('integrity_errors'):
            row['exclusion_reason'] = 'source_integrity_mismatch'
        elif len(pixel_labels[row['pixel_sha256']]) > 1:
            row['exclusion_reason'] = 'conflicting_labels_for_identical_pixels'
        elif row['pixel_sha256'] in kept:
            row['exclusion_reason'] = 'duplicate_pixels'
            row['duplicate_of'] = kept[row['pixel_sha256']]
        else:
            row['new_split'] = 'train'
            kept[row['pixel_sha256']] = row['image_id']
    return {'group_count': len(components), 'group_sizes': sorted(map(len, components.values()), reverse=True),
            'edge_counts': dict(sorted(edge_counts.items())), 'verified_physical_sample_count': None,
            'formal_holdout_available': False,
            'policy': 'Same session, session-scoped physical strings, byte/pixel duplicates, pHash<=6, '
                      'and unverified same-class old/new independence form one graph. Train-only; no formal validation/test.'}


def duplicate_summary(rows, field, cross_source=False):
    buckets = defaultdict(list)
    for row in rows:
        if row.get(field):
            buckets[row[field]].append(row)
    groups = [group for group in buckets.values() if len(group) > 1
              and (not cross_source or {r['source'] for r in group} == {'new', 'round1'})]
    return {'groups': len(groups), 'excess_copies': sum(len(group) - 1 for group in groups),
            'pair_count': sum(sum(r['source'] == 'new' for r in group) * sum(r['source'] == 'round1' for r in group)
                              if cross_source else len(group) * (len(group) - 1) // 2 for group in groups),
            'image_ids': [[r['image_id'] for r in group] for group in groups]}


def row_summary(rows):
    return {'image_count': len(rows), 'class_counts': dict(sorted(Counter(r['class_key'] for r in rows).items())),
            'image_status_counts': dict(sorted(Counter(r['status'] for r in rows).items())),
            'format_counts': dict(sorted(Counter(r['format'] or 'unreadable' for r in rows).items())),
            'dimension_counts': dict(sorted(Counter(f"{r['width']}x{r['height']}" for r in rows).items())),
            'label_status_counts': dict(sorted(Counter(r['label_status'] for r in rows).items())),
            'trusted_label_count': sum(r['trusted_label'] for r in rows),
            'new_split_counts': dict(sorted(Counter(r['new_split'] for r in rows).items())),
            'train_class_counts': dict(sorted(Counter(r['class_key'] for r in rows if r['new_split'] == 'train').items())),
            'exclusion_counts': dict(sorted(Counter(r['exclusion_reason'] for r in rows if r['exclusion_reason']).items()))}


def audit_directory(input_root, repo_root=ROOT, *, output=None, old_rows_path=DEFAULT_OLD_ROWS, phash_threshold=6):
    if not 0 <= phash_threshold <= 16:
        raise ValueError('pHash threshold must be in 0..16')
    input_root, repo_root = Path(input_root).resolve(), Path(repo_root).resolve()
    output = Path(output) if output is not None else DEFAULT_OUTPUT
    output = _inside(repo_root / 'runs', output if output.is_absolute() else repo_root / output)
    if output.is_relative_to(input_root):
        raise ValueError('Audit outputs may not be written inside input root')
    output.mkdir(parents=True, exist_ok=True)
    with safe_child(input_root, '分类清单.csv').open(encoding='utf-8-sig', newline='') as stream:
        index = list(csv.DictReader(stream))
    indexed = {}
    for record in index:
        path = safe_child(input_root, record['destination'])
        relative = path.relative_to(input_root).as_posix()
        if relative.casefold() in indexed:
            raise ValueError('Duplicate or case-colliding index destination')
        indexed[relative.casefold()] = record
    verification = json.loads(safe_child(input_root, '分类核验.json').read_text(encoding='utf-8-sig'))
    organizer_text = safe_child(input_root, '整理说明.md').read_text(encoding='utf-8-sig')
    reviewed_photos = '23张photo mid/photo up图片已逐张查看，归入吐司' in organizer_text
    files = sorted(safe_child(input_root, p.relative_to(input_root).as_posix())
                   for p in input_root.rglob('*') if p.is_file())
    metadata = [p for p in files if p.suffix.lower() not in IMAGE_EXTENSIONS]
    preservation = [{'relative_path': p.relative_to(input_root).as_posix(), 'sha256': hash_file(p),
                     'bytes': p.stat().st_size} for p in metadata]
    hashes = {item['relative_path']: item['sha256'] for item in preservation}
    parsed, metadata_errors = {}, []
    for path in metadata:
        if path.suffix.lower() == '.json':
            try:
                parsed[path.relative_to(input_root).as_posix()] = json.loads(path.read_text(encoding='utf-8-sig'))
            except (ValueError, UnicodeError) as error:
                metadata_errors.append({'relative_path': path.relative_to(input_root).as_posix(), 'error': type(error).__name__})
    new_rows = []
    images = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
    for number, path in enumerate(images, 1):
        relative = path.relative_to(input_root).as_posix()
        if len(path.relative_to(input_root).parts) != 2 or path.parent.name not in FOLDER_CLASSES:
            raise ValueError('Image is not directly inside an explicitly mapped class folder')
        class_key = FOLDER_CLASSES[path.parent.name]
        sidecar_name = str(Path(relative).with_suffix('.json')).replace('\\', '/')
        sidecar = parsed.get(sidecar_name, {})
        record = indexed.get(relative.casefold(), {})
        row = {'image_id': 'new_' + hashlib.sha256(relative.encode()).hexdigest()[:24],
               'source': 'new', 'source_id': 'user_supplied_classified_oven_round2',
               'relative_path': relative, 'class_key': class_key, 'class_id': CLASS_KEYS.index(class_key),
               'original_split': 'unassigned', 'license': 'unknown',
               'index_source_raw': record.get('source'), 'metadata_relative_path': sidecar_name if sidecar else None,
               'source_collection': (record.get('source') or 'unknown').split('/')[0],
               'metadata_sha256': hashes.get(sidecar_name), 'integrity_errors': [], 'metadata_warnings': [],
               'source_image_file_raw': sidecar.get('image_file')}
        read_image(path, row)
        row.update(label_evidence(class_key, relative, sidecar, reviewed_photos))
        row.update(scene_evidence(sidecar, path.name))
        if sidecar:
            if sidecar.get('sha256') != row['sha256']:
                row['integrity_errors'].append('sidecar_sha256_mismatch')
            if sidecar.get('image_file') != path.name:
                row['metadata_warnings'].append('sidecar_filename_differs_but_sha256_matches'
                    if sidecar.get('sha256') == row['sha256'] else 'sidecar_filename_and_sha256_mismatch')
            if row['status'] == 'ok' and (sidecar.get('width'), sidecar.get('height')) != (row['width'], row['height']):
                row['integrity_errors'].append('sidecar_dimensions_mismatch')
        new_rows.append(row)
        if number % 100 == 0 or number == len(images):
            print(f'New image audit: {number}/{len(images)} decoded', flush=True)
    actual = {p.relative_to(input_root).as_posix(): p for p in files}
    all_hashes = {**hashes, **{r['relative_path']: r['sha256'] for r in new_rows}}
    index_errors = []
    for record in index:
        name = safe_child(input_root, record['destination']).relative_to(input_root).as_posix()
        path = actual.get(name)
        errors = []
        if path is None:
            errors.append('missing_file')
        else:
            if all_hashes[name] != record['sha256']:
                errors.append('sha256_mismatch')
            if path.stat().st_size != int(record['size_bytes']):
                errors.append('size_mismatch')
            if record['category'] != Path(name).parts[0]:
                errors.append('category_mismatch')
            if (record['file_type'] == 'image') != (path.suffix.lower() in IMAGE_EXTENSIONS):
                errors.append('file_type_mismatch')
        if errors:
            index_errors.append({'relative_path': name, 'errors': errors})
    by_name = {r['relative_path']: r for r in new_rows}
    for error in index_errors:
        if error['relative_path'] in by_name:
            by_name[error['relative_path']]['integrity_errors'].extend(error['errors'])
    unindexed = sorted(name for name in actual if '/' in name and name.casefold() not in indexed)
    for name in unindexed:
        if name in by_name:
            by_name[name]['integrity_errors'].append('missing_index_record')
    old_rows = []
    if old_rows_path is not None:
        old_path = Path(old_rows_path)
        old_path = old_path if old_path.is_absolute() else repo_root / old_path
        for number, original in enumerate(json.loads(old_path.read_text(encoding='utf-8')), 1):
            row = dict(original)
            previous_sha = row['sha256']
            row.update({'source': 'round1', 'image_id': 'round1_' + row['image_id'],
                        'class_id': CLASS_KEYS.index(row['class_key']), 'trusted_label': False,
                        'label_status': 'candidate_directory',
                        'label_evidence': 'Round1 source class, not independently human confirmed',
                        'integrity_errors': [], 'metadata_warnings': [], 'source_collection': 'round1_archive'})
            row.update(scene_evidence({}, Path(row['relative_path']).name))
            read_image(safe_child(repo_root, row['relative_path']), row)
            if row['sha256'] != previous_sha:
                row['integrity_errors'].append('round1_previous_sha256_mismatch')
            old_rows.append(row)
            if number % 100 == 0:
                print(f'Round1 comparison audit: {number} images decoded', flush=True)
    rows = new_rows + old_rows
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Image IDs must be unique')
    tree, pairs = BKTree(), []
    for row in rows:
        if row['phash']:
            value = int(row['phash'], 16)
            for previous, distance in tree.find(value, phash_threshold):
                pairs.append({'left': previous['image_id'], 'right': row['image_id'], 'distance': distance,
                              'scope': 'new_new' if previous['source'] == row['source'] == 'new' else
                                       'round1_round1' if previous['source'] == row['source'] else 'new_round1',
                              'same_file': previous['sha256'] == row['sha256'],
                              'same_pixels': previous['pixel_sha256'] == row['pixel_sha256'],
                              'cross_class': previous['class_key'] != row['class_key']})
            tree.add(value, row)
    grouping = group_rows(rows, pairs)
    for record in preservation:
        record['unchanged_after_audit'] = hash_file(safe_child(input_root, record['relative_path'])) == record['sha256']
    claimed_counts = verification.get('image_counts', {})
    summary = {
        'status': 'exploratory_candidates_only_no_formal_holdout',
        'source': {'origin': 'User supplied reorganized oven capture folder; original acquisition rights unverified',
                   'license': 'unknown', 'redistribution': 'not authorized pending rights verification',
                   'source_images_modified_or_copied': False},
        'experimental_classes': [{'id': i, 'key': key, 'zh': next(folder for folder, value in FOLDER_CLASSES.items() if value == key)}
                                 for i, key in enumerate(CLASS_KEYS)],
        'class_mapping_scope': 'Oven experiment only; preserves round1 IDs and appends corn/sausage/steak; not configs/classes.json',
        'new': row_summary(new_rows), 'round1': row_summary(old_rows), 'combined': row_summary(rows),
        'index': {'record_count': len(index), 'file_type_counts': dict(Counter(r['file_type'] for r in index)),
                  'sha_mismatch_count': sum('sha256_mismatch' in e['errors'] for e in index_errors),
                  'error_count': len(index_errors), 'errors': index_errors, 'unindexed_files': unindexed,
                  'declared_image_count_matches': verification.get('image_count') == len(new_rows),
                  'declared_class_counts_match': claimed_counts == dict(Counter(Path(r['relative_path']).parts[0] for r in new_rows)),
                  'source_prefix_counts': dict(sorted(Counter((r['index_source_raw'] or 'unknown').split('/')[0] for r in new_rows).items()))},
        'metadata': {'per_image_sidecar_count': sum(r['metadata_relative_path'] is not None for r in new_rows),
                     'json_file_count': sum(p.suffix.lower() == '.json' for p in metadata),
                     'preserved_nonimage_file_count': len(preservation),
                     'all_metadata_unchanged_after_audit': all(r['unchanged_after_audit'] for r in preservation),
                     'parse_errors': metadata_errors,
                     'missing_sidecars': [r['relative_path'] for r in new_rows if r['metadata_relative_path'] is None],
                     'source_label_status_counts': dict(Counter(r['source_label_status'] for r in new_rows)),
                     'human_label_confirmed_counts': dict(Counter(str(r['source_human_label_confirmed']) for r in new_rows)),
                     'candidate_class_counts': dict(Counter(str(r['candidate_class_key_unreviewed']) for r in new_rows)),
                     'candidate_directory_disagreement_count': sum(r['candidate_disagrees_with_directory'] for r in new_rows),
                     'metadata_warning_counts': dict(Counter(w for r in rows for w in r['metadata_warnings'])),
                     'row_integrity_error_count': sum(bool(r['integrity_errors']) for r in rows),
                     'session_count': len({r['session_id'] for r in new_rows} - {'unknown'}),
                     'physical_identifier_string_count': len({r['physical_sample_id'] for r in new_rows} - {'unknown'}),
                     'physical_ids_are_verified_specimens': False,
                     'images_with_exif': sum(bool(r.get('exif_tag_ids')) for r in new_rows),
                     'scene_value_counts': {field: dict(Counter(r[field]['value'] for r in new_rows))
                                            for field in ('foil', 'shelf', 'angle', 'lighting')},
                     'unknown_oven_id_count': sum(r['oven_id'] == 'unknown' for r in new_rows)},
        'duplicates': {scope: {field: duplicate_summary(subset, field, scope == 'new_round1')
                              for field in ('sha256', 'pixel_sha256')}
                       for scope, subset in [('new_new', new_rows), ('new_round1', rows), ('round1_round1', old_rows)]},
        'phash_threshold': phash_threshold,
        'near_pairs': {scope: {'pair_count': sum(p['scope'] == scope for p in pairs),
                              'nonidentical_pixel_pair_count': sum(p['scope'] == scope and not p['same_pixels'] for p in pairs),
                              'cross_class_pair_count': sum(p['scope'] == scope and p['cross_class'] for p in pairs)}
                       for scope in ('new_new', 'new_round1', 'round1_round1')},
        'grouping': grouping,
    }
    for name, value in [('new_rows.json', new_rows), ('combined_rows.json', rows),
                        ('data_audit.json', summary), ('near_pairs.json', pairs), ('source_metadata_hashes.json', preservation)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return summary, rows


def contact_sheet(rows, input_root, output):
    """45 local thumbnails spread over class scene prefixes; no human label claim."""
    candidates = defaultdict(list)
    for row in rows:
        if row['source'] == 'new' and row['status'] == 'ok':
            candidates[row['class_key']].append(row)
    chosen = []
    for index, key in enumerate(CLASS_KEYS):
        seen = set()
        items = []
        for row in candidates[key]:
            if row['sample_prefix_raw'] not in seen:
                items.append(row)
                seen.add(row['sample_prefix_raw'])
        count = 4 if index < 9 else 3
        chosen.extend(items[round(i * (len(items) - 1) / max(1, min(count, len(items)) - 1))]
                      for i in range(min(count, len(items))))
    sheet = Image.new('RGB', (5 * 260, 9 * 185 + 30), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 7), '45 organizer-directory candidates | visual audit only | no formal labels or holdout', fill='black')
    for index, row in enumerate(chosen):
        x, y = (index % 5) * 260 + 4, (index // 5) * 185 + 30
        with Image.open(safe_child(input_root, row['relative_path'])) as image:
            sheet.paste(ImageOps.contain(image.convert('RGB'), (252, 145)), (x, y))
        draw.text((x, y + 146), f'{index + 1}: {row["class_key"]} [{row["label_status"][:9]}]', fill='black')
        draw.text((x, y + 161), row['sample_prefix_raw'][:37], fill='black')
    sheet.save(output / 'representatives_45.jpg', quality=90)
    (output / 'representatives_45_index.json').write_text(json.dumps(chosen, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--old-rows', type=Path, default=DEFAULT_OLD_ROWS)
    parser.add_argument('--contacts', action='store_true')
    args = parser.parse_args()
    summary, rows = audit_directory(args.input_root, output=args.output, old_rows_path=args.old_rows)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.contacts:
        contact_sheet(rows, args.input_root, output)
    print(json.dumps({key: summary[key] for key in ('new', 'round1', 'combined', 'near_pairs', 'grouping')}, indent=2), flush=True)
    return 0 if (summary['index']['error_count'] == 0 and summary['metadata']['row_integrity_error_count'] == 0
                 and summary['metadata']['all_metadata_unchanged_after_audit']
                 and all(row['status'] == 'ok' for row in rows)) else 2


if __name__ == '__main__':
    raise SystemExit(main())
