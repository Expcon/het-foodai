"""Read-only v2 error audit. This entry point has no training or export action."""
import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from scripts.audit_images import IMAGE_EXTENSIONS, BKTree, image_facts, write_csv
from .export_desk import compare_probabilities, write_json
from .model import ROOT, PREPROCESS, class_digest, configure_sdk, digest, load_model, output_dir, transform
from .round2 import V2_KEYS, resolve_image
from .round2_audit import FOLDER_CLASSES, safe_child, scene_evidence

BASE = ROOT / 'runs/oven-round2-20261107'
TARGETS = {'salmon', 'sausage', 'corn', 'potato'}
EXTRA_CORN = {'白玉米', '黄玉米', '玉米3'}
CHECKPOINT_SHA = '44878014afe8381055ed2a26fffff6269982e3f5b57c2da0fbedc596127770d5'
ONNX_SHA = '8c696ce5dc63a25053198f1781504b4a53cb801f2d971e3bceeba52eed789689'


def snapshot(root):
    return {p.relative_to(root).as_posix(): digest(safe_child(root, p.relative_to(root).as_posix()))
            for p in sorted(root.rglob('*')) if p.is_file()}


def field_candidate(name):
    # Filename/user report supplies a proposed label, never human confirmation.
    if name.startswith('三文鱼'): return 'salmon'
    if name.startswith('香肠'): return 'sausage'
    if name.startswith('potato'): return 'potato'
    return 'unknown'


def image_record(path, row):
    row = dict(row)
    row.update(sha256=digest(path), decode_status='OK', format=None)
    try:
        with Image.open(path) as image:
            image.load()
            row.update(image_facts(ImageOps.exif_transpose(image)))
            row['format'] = image.format
        row['content_224_size'] = list(ImageOps.contain(Image.new('RGB', (row['width'], row['height'])), (224, 224)).size)
    except (OSError, ValueError) as exc:
        row.update(decode_status='FAIL', decode_error=str(exc))
    return row


def path_for(row, source_root, field_root):
    if row['source'] == 'field': return safe_child(field_root, row['relative_path'])
    return resolve_image(row, source_root)


def check_mapping(metadata, contract, rows):
    expected = class_digest(V2_KEYS)
    if metadata['class_keys'] != V2_KEYS or contract['class_keys'] != V2_KEYS:
        raise ValueError('Ordered keys differ')
    if metadata['class_list_sha256'] != expected or contract['class_list_sha256'] != expected:
        raise ValueError('Ordered class SHA differs')
    if contract['classes'] != [{'id': i, 'key': key} for i, key in enumerate(V2_KEYS)]:
        raise ValueError('Export class IDs differ')
    if metadata['config']['class_keys'] != V2_KEYS or contract['preprocess'] != PREPROCESS:
        raise ValueError('Training keys or preprocessing differs')
    if contract['source_checkpoint_sha256'] != CHECKPOINT_SHA or contract['models']['onnx']['sha256'] != ONNX_SHA:
        raise ValueError('Contract is bound to different model bytes')
    counts = Counter()
    for row in rows:
        if row['class_id'] != V2_KEYS.index(row['class_key']):
            raise ValueError('Training row ID differs')
        folder = Path(row['relative_path']).parent.name
        if (FOLDER_CLASSES.get(folder, folder)) != row['class_key']:
            raise ValueError('Training directory differs from row class')
        if row['new_split'] == 'train': counts[row['class_key']] += 1
    if dict(counts) != metadata['config']['provenance']['per_class_samples']:
        raise ValueError('Training class counts differ')
    return {'status': 'PASS', 'class_keys': V2_KEYS, 'class_list_sha256': expected,
            'train_counts': dict(counts), 'rows_checked': len(rows)}


def contacts(rows, out, source_root, field_root):
    directory = out / 'contacts'
    directory.mkdir()
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    font = ImageFont.truetype(str(font_path), 13) if font_path.exists() else ImageFont.load_default()
    groups = defaultdict(list)
    for row in rows: groups[row['gallery']].append(row)
    pages, index = [], []
    for group, members in sorted(groups.items()):
        for start in range(0, len(members), 24):
            page = f'{group}_{start // 24 + 1:02d}.jpg'
            batch = members[start:start+24]
            height = ((len(batch)+3)//4)*254 + 32
            canvas = Image.new('RGB', (1440, height), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 6), f'{group} | all original images | label review pending', fill='black', font=font)
            for slot, row in enumerate(batch):
                x, y = slot % 4 * 360, slot // 4 * 254 + 32
                if row['decode_status'] == 'OK':
                    with Image.open(path_for(row, source_root, field_root)) as image:
                        tile = ImageOps.contain(ImageOps.exif_transpose(image).convert('RGB'), (356, 201))
                        canvas.paste(tile, (x, y))
                draw.text((x+3, y+203), f"{start+slot+1}: {Path(row['relative_path']).name}"[:48], fill='black', font=font)
                draw.text((x+3, y+222), f"{row['class_key']} | {row['source']} | {row['image_id'][-12:]}", fill='black', font=font)
                row['contact_page'] = 'contacts/' + page
                row['contact_slot'] = slot + 1
                index.append({k: row[k] for k in ['image_id', 'relative_path', 'class_key', 'contact_page', 'contact_slot']})
            canvas.save(directory / page, quality=92)
            pages.append((group, page))
    body = ['<!doctype html><meta charset="utf-8"><title>v2 error audit</title>',
            '<style>body{font-family:sans-serif}img{width:min(100%,1440px)}a{display:block;margin:12px}</style>',
            '<h1>原图联系表 · 标签待人工核验</h1><p>图片仅作本地审核。页内数字为图片序号，不能作为实物数量。</p>']
    for group, page in pages:
        body.append(f'<a href="contacts/{page}">{html.escape(group)} / {page}</a><img loading="lazy" src="contacts/{page}">')
    (out/'contact_index.html').write_text('\n'.join(body), encoding='utf-8')
    write_csv(out/'contact_index.csv', index)
    template = [{**entry, 'reviewed_class': '', 'human_confirmed': '', 'food_color': '',
                 'variety': 'unknown', 'maturity': 'unknown', 'mixed_food': '',
                 'physical_identity': 'unknown', 'notes': ''} for entry in index]
    write_csv(out/'human_review_template.csv', template)
    return len(pages)


def inventory(source_root, field_root, training, out):
    by_path = {r['relative_path']: r for r in training if r['source'] == 'new'}
    selected = []
    for path in sorted(source_root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS: continue
        rel = path.relative_to(source_root).as_posix()
        folder = Path(rel).parts[0]
        if FOLDER_CLASSES.get(folder) not in TARGETS and folder not in EXTRA_CORN: continue
        row = by_path.get(rel)
        if row is None:
            row = {'image_id': 'additional_' + hashlib.sha256(rel.encode()).hexdigest()[:24],
                   'source': 'new', 'relative_path': rel, 'class_key': 'corn', 'trusted_label': False,
                   'new_split': 'not_in_training_version', **scene_evidence({}, path.name)}
        expected = row.get('sha256')
        row = image_record(safe_child(source_root, rel), row)
        row['original_training_hash_matches'] = expected is None or expected == row['sha256']
        row['gallery'] = 'new_' + (folder if folder in EXTRA_CORN else row['class_key'])
        selected.append(row)
    for original in training:
        if original['source'] == 'round1' and original['class_key'] in TARGETS:
            row = image_record(resolve_image(original, source_root), original)
            row.update(gallery='old_' + row['class_key'], original_training_hash_matches=row['sha256'] == original['sha256'])
            selected.append(row)
    for path in sorted(field_root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS: continue
        rel = path.relative_to(field_root).as_posix()
        sidecar = safe_child(field_root, Path(rel).with_suffix('.json').as_posix())
        metadata = json.loads(sidecar.read_text(encoding='utf-8-sig')) if sidecar.is_file() else {}
        candidate = field_candidate(path.name)
        row = {'image_id': 'field_' + hashlib.sha256(rel.encode()).hexdigest()[:24],
               'source': 'field', 'relative_path': rel, 'class_key': candidate,
               'trusted_label': False, 'new_split': 'field_diagnostic_only',
               'gallery': 'field_' + candidate, 'captured_at': metadata.get('created_host_local'),
               **scene_evidence(metadata, path.name)}
        row = image_record(safe_child(field_root, rel), row)
        row['sidecar_hash_matches'] = metadata.get('sha256') == row['sha256'] if metadata else None
        selected.append(row)
    train_bytes = defaultdict(list)
    train_pixels = defaultdict(list)
    for row in training:
        if row['new_split'] == 'train':
            train_bytes[row['sha256']].append(row['image_id'])
            train_pixels[row['pixel_sha256']].append(row['image_id'])
    for row in selected:
        row['training_byte_matches'] = train_bytes[row['sha256']]
        row['training_pixel_matches'] = train_pixels[row.get('pixel_sha256')]
    tree = BKTree()
    pairs = []
    for row in selected:
        if row.get('phash') is None: continue
        value = int(row['phash'], 16)
        for prior, distance in tree.find(value, 6):
            pairs.append({'left': prior['image_id'], 'right': row['image_id'], 'distance': distance,
                          'same_pixels': prior['pixel_sha256'] == row['pixel_sha256'],
                          'cross_class': prior['class_key'] != row['class_key']})
        tree.add(value, row)
    write_json(out/'near_candidates.json', pairs)
    pages = contacts(selected, out, source_root, field_root)
    write_json(out/'inventory.json', selected)
    return selected, pages


def compare(rows, network, engine, source_root, field_root, out):
    import numpy as np
    import torch
    from tools.oven_desk.inference_engine import prepare_input, read_image_unicode
    pre = transform(False)
    results = []
    for index, row in enumerate(rows):
        if row['decode_status'] != 'OK': continue
        path = path_for(row, source_root, field_root)
        if digest(path) != row['sha256']: raise ValueError('Input changed during audit')
        with Image.open(path) as image: tensor = pre(image).unsqueeze(0)
        frame = read_image_unicode(path, 'oven_round2')
        desktop, _ = prepare_input(frame, 'oven_round2', 224, 224)
        with torch.inference_mode(): p = network(tensor)[0].numpy()[0]
        prediction = engine.predict(frame)
        q = np.array(prediction['probabilities'], dtype=np.float32)
        result = compare_probabilities(p, q)
        result['preprocess_exact'] = bool(np.array_equal(tensor.numpy(), desktop))
        result['pass'] = result['pass'] and result['preprocess_exact']
        result.update({key: row.get(key) for key in ['image_id', 'source', 'relative_path', 'class_key', 'gallery',
                      'contact_page', 'contact_slot', 'sha256', 'physical_sample_id', 'session_id', 'foil', 'shelf']})
        result.update(pytorch_probabilities=p.tolist(), onnx_probabilities=q.tolist(), top3=prediction['top3'],
                      directory_agreement=prediction['top3'][0]['label'] == row['class_key'],
                      in_training_bytes=bool(row['training_byte_matches']), human_confirmed=False)
        results.append(result)
        if (index+1) % 100 == 0: print(f'Inference-only comparison {index+1}/{len(rows)}', flush=True)
    write_json(out/'pytorch_onnx.json', results)
    flat = [{**{k: r[k] for k in ['image_id', 'source', 'relative_path', 'class_key', 'gallery', 'contact_page', 'contact_slot']},
             'top1': r['top3'][0]['label'], 'score': r['top3'][0]['score'], 'top3': json.dumps(r['top3']),
             'max_abs_error': r['max_abs_error'], 'preprocess_exact': r['preprocess_exact'], 'in_training_bytes': r['in_training_bytes'],
             'human_confirmed': False} for r in results]
    write_csv(out/'predictions.csv', flat)
    for key in TARGETS:
        errors = [r for r in flat if r['class_key'] == key and r['top1'] != key]
        write_csv(out/f'{key}_directory_disagreements.csv', errors, list(flat[0]))
        write_csv(out/f'{key}_training_disagreements.csv', [r for r in errors if r['in_training_bytes']], list(flat[0]))
    groups = {}
    for group in sorted({r['gallery'] for r in results}):
        members = [r for r in results if r['gallery'] == group]
        groups[group] = {'images': len(members), 'directory_agreement': sum(r['directory_agreement'] for r in members),
                         'predicted_counts': dict(Counter(r['top3'][0]['label'] for r in members))}
    summary = {'status': 'PASS' if results and all(r['pass'] for r in results) else 'FAIL',
               'images': len(results), 'top1_agree': sum(r['top1_agree'] for r in results),
               'ordered_top3_agree': sum(r['top3_agree'] for r in results),
               'preprocess_exact': sum(r['preprocess_exact'] for r in results),
               'max_abs_error': max(r['max_abs_error'] for r in results), 'groups': groups,
               'accuracy_scope': 'Filename/directory proposals only; no formal accuracy or independent specimen count'}
    write_json(out/'comparison_summary.json', summary)
    return summary


def make_engine():
    import onnxruntime as ort
    from tools.oven_desk.inference_engine import ModelEngine
    model_path = BASE/'onnx/oven12.onnx'
    engine = ModelEngine(model_path, BASE/'onnx/classes.json', 'oven_round2', 224)
    # Bound audit CPU parallelism to match PyTorch and the original export check.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    engine.session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
    return engine


def run(args):
    import torch
    source_root = args.source_root.resolve(strict=True)
    field_root = args.field_root.resolve(strict=True)
    out = output_dir(args.out)
    if any(out.is_relative_to(root) for root in [source_root, field_root]): raise ValueError('Output overlaps originals')
    out.mkdir(parents=True, exist_ok=False)
    configure_sdk(out/'sdk-settings')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    training_path = BASE/'audit/combined_rows.json'
    training = json.loads(training_path.read_text(encoding='utf-8'))
    protected = json.loads((BASE/'protected_before.json').read_text())
    for path in [BASE/'yolo11n-12cls-fixed30/last.pt', BASE/'onnx/oven12.onnx', BASE/'onnx/classes.json',
                 BASE/'yolo11n-12cls-fixed30/run_info.json', training_path]:
        protected[str(path)] = digest(path)
    before = {'dataset': snapshot(source_root), 'field': snapshot(field_root),
              'protected': {name: digest(name) for name in protected}}
    write_json(out/'preservation_before.json', before)
    network, meta = load_model(BASE/'yolo11n-12cls-fixed30/last.pt', 'cpu', CHECKPOINT_SHA)
    if digest(BASE/'onnx/oven12.onnx') != ONNX_SHA: raise ValueError('Original ONNX SHA differs')
    contract = json.loads((BASE/'onnx/classes.json').read_text())
    mapping = check_mapping(meta, contract, training)
    if digest(training_path) != meta['config']['rows_sha256']: raise ValueError('Training manifest SHA differs')
    engine = make_engine()
    custom = engine.session.get_modelmeta().custom_metadata_map
    if custom['class_list_sha256'] != mapping['class_list_sha256'] or json.loads(custom['preprocess']) != PREPROCESS:
        raise ValueError('ONNX metadata differs')
    mapping.update(checkpoint_sha256=CHECKPOINT_SHA, onnx_sha256=ONNX_SHA,
                   checkpoint_head_shape=list(network.model[-1].linear.weight.shape),
                   onnx_input=engine.session.get_inputs()[0].shape, onnx_output=engine.session.get_outputs()[0].shape)
    write_json(out/'mapping.json', mapping)
    print('V2 checkpoint / training directories / class contract / ONNX mapping: PASS', flush=True)
    rows, pages = inventory(source_root, field_root, training, out)
    write_json(out/'inventory_summary.json', {'images': len(rows), 'galleries': dict(Counter(r['gallery'] for r in rows)),
        'decode_failures': [r['image_id'] for r in rows if r['decode_status'] != 'OK'], 'contact_pages': pages,
        'additional_images_seen_in_training': sum(bool(r['training_pixel_matches']) for r in rows if r.get('new_split') in {'not_in_training_version', 'field_diagnostic_only'})})
    summary = compare(rows, network, engine, source_root, field_root, out)
    after = {'dataset': snapshot(source_root), 'field': snapshot(field_root),
             'protected': {name: digest(name) for name in protected}}
    write_json(out/'preservation_after.json', after)
    preservation = {'status': 'PASS' if before == after else 'FAIL',
                    'dataset_files': len(before['dataset']), 'field_files': len(before['field']),
                    'protected_files': len(protected)}
    write_json(out/'preservation.json', preservation)
    print(json.dumps({'mapping': mapping['status'], 'inventory': len(rows), 'contacts': pages,
                      'comparison': summary['status'], 'max_abs_error': summary['max_abs_error'], 'preservation': preservation}), flush=True)
    clean = all(r['decode_status'] == 'OK' and r.get('original_training_hash_matches', True) for r in rows)
    return 0 if summary['status'] == 'PASS' and preservation['status'] == 'PASS' and clean else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--field-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    return run(parser.parse_args())


if __name__ == '__main__': raise SystemExit(main())
