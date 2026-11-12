"""Read-only real-image Desk v2 parity; no camera, training or accuracy claim."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools/oven_desk'))

import numpy as np
from inference_engine import ModelEngine, file_sha256, prepare_input, read_image_unicode
from tools.oven_training.export_desk import compare_probabilities, summarize,read_alignment_rows,resolve_alignment_image


def run(args):
    target = args.out.resolve()
    source_root = args.source_root.resolve(strict=True)
    runs = (ROOT/'runs').resolve()
    if (not runs.is_relative_to(ROOT) or target == runs or not target.is_relative_to(runs)
            or target.is_relative_to(source_root) or target.exists()):
        raise ValueError('Use a new output file inside repository runs/, outside the photo source')
    if not source_root.is_dir():
        raise ValueError('Source root must be an existing photo directory')
    report = {'summary': {'status': 'FAIL'}, 'images': []}
    try:
        bound = {key: file_sha256(getattr(args, key)) for key in ('model', 'classes', 'alignment', 'rows')}
        diagnostic_rows=getattr(args,'diagnostic_rows',None)
        if diagnostic_rows is not None:bound['diagnostic_rows']=file_sha256(diagnostic_rows)
        evidence = json.loads(args.alignment.read_text(encoding='utf-8'))
        rows = read_alignment_rows(args.rows,diagnostic_rows)
        summary, references = evidence['summary'], evidence['images']
        if (summary['status'] != 'PASS' or summary['onnx_sha256'] != bound['model']
                or summary['rows_sha256'] != bound['rows'] or not rows
                or summary['images'] != len(rows) or len(rows) != len(references)):
            raise ValueError('Exporter evidence must PASS and bind this model and complete rows')
        if diagnostic_rows is not None and summary.get('diagnostic_rows_sha256')!=bound['diagnostic_rows']:
            raise ValueError('Diagnostic rows differ from exporter evidence')
        engine = ModelEngine(args.model, args.classes, 'oven_round2', 224)
        if engine.contract['source_checkpoint_sha256'] != summary['checkpoint_sha256']:
            raise ValueError('Checkpoint differs between contract and exporter evidence')
        seen = set()
        for index, (row, reference) in enumerate(zip(rows, references)):
            image_id = row['image_id']
            if (image_id in seen or image_id != reference['image_id']
                    or row['sha256'] != reference['sha256'] or reference.get('pass') is not True):
                raise ValueError(f'Duplicate, reordered or unapproved image evidence: {image_id}')
            seen.add(image_id)
            path = resolve_alignment_image(row,source_root,getattr(args,'r2_source_root',None),getattr(args,'diagnostic_root',None))
            if row['source'] == 'round1' and not path.is_relative_to((ROOT/'data/raw/oven_round1_20261106').resolve()):
                raise ValueError('Round1 image escapes its original raw directory')
            if file_sha256(path) != row['sha256']:
                raise ValueError(f'Original image changed: {image_id}')
            frame = read_image_unicode(path, mode='oven_round2')
            tensor, _ = prepare_input(frame, 'oven_round2', 224, 224)
            tensor_sha = hashlib.sha256(tensor.tobytes()).hexdigest()
            predicted = engine.predict(frame)
            actual = np.asarray(predicted['probabilities'], dtype=np.float32)
            record = compare_probabilities(reference['reference_probabilities'], actual)
            tensor_matches = tensor_sha == reference['input_tensor_sha256']
            top3_matches = [entry['id'] for entry in predicted['top3']] == record['actual_top3']
            record['pass'] = record['pass'] and tensor_matches and top3_matches
            record.update(image_id=image_id, sha256=row['sha256'], input_tensor_sha256=tensor_sha,
                          preprocessing_exact=tensor_matches, desk_top3_consistent=top3_matches,
                          actual_probabilities=actual.tolist())
            if file_sha256(path) != row['sha256']:
                raise ValueError(f'Original image changed during inference: {image_id}')
            report['images'].append(record)
            if (index+1) % 50 == 0:
                print(f'Desk compared {index+1}/{len(rows)}', flush=True)
        for key, sha in bound.items():
            if file_sha256(getattr(args, key)) != sha:
                raise ValueError(f'{key} changed during alignment')
        versions = {key: importlib.metadata.version(key) for key in ('numpy', 'Pillow', 'onnxruntime', 'opencv-python')}
        report['summary'] = summarize(report['images'], versions)
        report['summary'].update(mode='oven_round2', independent_validation=False, camera_opened=False,
                                 preprocessing_tensors_exact=sum(r['preprocessing_exact'] for r in report['images']),
                                 python_executable=sys.executable, input_sha256=bound,
                                 source_checkpoint_sha256=engine.contract['source_checkpoint_sha256'])
    except Exception as exc:
        report['summary'].update(status='FAIL', error=f'{type(exc).__name__}: {exc}',
                                 images_checked=len(report['images']), camera_opened=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)
    return 0 if report['summary']['status'] == 'PASS' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'classes', 'alignment', 'rows', 'source-root', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('r2-source-root','diagnostic-rows','diagnostic-root'):parser.add_argument('--'+name,type=Path)
    try:
        return run(parser.parse_args())
    except (OSError, ValueError) as exc:
        print(f'Alignment rejected: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
