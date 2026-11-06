"""Local distribution-shift diagnostic for the supplied oven ZIP, never a holdout evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from foodai.contracts import hash_file, load_classes
from foodai.predictor import PredictionAdapter, as_contract_dict

BASELINE_SHA256 = '41056b3ea90d5fa718a18f48cf5182373d1010c5185cb426ba5c949b1ec1f4c5'
SOURCE_ID = 'user_supplied_oven_zip_20261106'
SOURCE_PREFIX = 'data/raw/oven_round1_20261106/'
OVEN_KEYS = {'asparagus', 'bacon', 'broccoli', 'chicken_wing', 'croissant',
             'french_fries', 'potato', 'salmon', 'toast'}
OVERLAP = {'broccoli', 'potato'}
LIMITATIONS = [
    'Descriptive diagnostic only: no independent holdout or formal generalization estimate.',
    'Original ZIP splits are not independent; physical specimen/session identity is unavailable.',
    'Labels are source supplied and not independently human verified.',
    'Only broccoli and potato overlap the frozen eight-class checkpoint.',
    'Seven unsupported labels are outside the checkpoint vocabulary; no nine-class accuracy is computed.',
    'Softmax confidence is not an OOD detector and does not establish label correctness.',
    'No training, relabeling, threshold tuning, or old frozen-test evaluation is performed.',
]


def load_rows(rows_path: Path, root: Path = ROOT) -> list[dict]:
    rows = json.loads(Path(rows_path).read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Rows must be a nonempty JSON list')
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f'Row {index}: expected object')
        rel = row.get('relative_path')
        if (not isinstance(rel, str) or not rel.startswith(SOURCE_PREFIX)
                or '\\' in rel or ':' in rel or '..' in PurePosixPath(rel).parts
                or not (root / rel).resolve().is_relative_to(root.resolve())):
            raise ValueError(f'Row {index}: path must remain inside the supplied oven source')
        if rel in seen:
            raise ValueError(f'Row {index}: duplicate relative_path')
        seen.add(rel)
        if (row.get('source_id') != SOURCE_ID or row.get('class_key') not in OVEN_KEYS
                or row.get('original_split') not in {'train', 'val', 'test'}
                or row.get('new_split') not in {'train', 'excluded'}
                or not isinstance(row.get('group_id'), str) or not row['group_id'].strip()):
            raise ValueError(f'Row {index}: invalid source, class, split or group metadata')
        for field in ('sha256', 'pixel_sha256'):
            if not isinstance(row.get(field), str) or not re.fullmatch('[0-9a-f]{64}', row[field]):
                raise ValueError(f'Row {index}: invalid {field}')
        if hash_file(root / rel) != row['sha256']:
            raise ValueError(f'Row {index}: image SHA256 differs from audited source')
    return rows


def build_summary(rows: list[dict], predictions: list[dict], failures: list[dict]) -> dict:
    def summarize(selected_rows, selected_predictions):
        per_class = {}
        for key in sorted({row['class_key'] for row in selected_rows}):
            records = [item for item in selected_predictions if item['class_key'] == key]
            supported = key in OVERLAP
            per_class[key] = {
                'input_images': sum(row['class_key'] == key for row in selected_rows),
                'successful_images': len(records), 'supported_label': supported,
                'top1_prediction_counts': dict(sorted(Counter(
                    item['predictions'][0]['class_key'] for item in records).items())),
                'top1_correct': sum(item['top1_matches_source_label'] for item in records) if supported else None,
                'top3_contains_source_label': sum(item['top3_contains_source_label'] for item in records) if supported else None,
            }
        overlap = [item for item in selected_predictions if item['class_key'] in OVERLAP]
        unsupported = [item for item in selected_predictions if item['class_key'] not in OVERLAP]
        return {
            'input_images': len(selected_rows), 'successful_images': len(selected_predictions),
            'unique_byte_sha256': len({row['sha256'] for row in selected_rows}),
            'unique_pixel_sha256': len({row['pixel_sha256'] for row in selected_rows}),
            'uncertainty_groups': len({row['group_id'] for row in selected_rows}),
            'original_split_counts': dict(sorted(Counter(row['original_split'] for row in selected_rows).items())),
            'per_class': per_class, 'unsupported_images': len(unsupported),
            'unsupported_top1_score_ge_0_90': sum(item['predictions'][0]['score'] >= .9 for item in unsupported),
            'overlap_raw_diagnostic': {
                'successful_images': len(overlap),
                'top1_correct': sum(item['top1_matches_source_label'] for item in overlap),
                'top3_contains_source_label': sum(item['top3_contains_source_label'] for item in overlap),
            },
        }

    latencies = sorted(item['latency_ms'] for item in predictions)
    return {
        'scope': 'oven_source_distribution_shift_diagnostic_only',
        'limitations': LIMITATIONS,
        'all_supplied_rows': summarize(rows, predictions),
        'retained_train_rows': summarize([row for row in rows if row['new_split'] == 'train'],
                                         [item for item in predictions if item['new_split'] == 'train']),
        'prediction_error_count': len(failures),
        'internal_latency_ms': {
            'samples': len(latencies), 'min': min(latencies) if latencies else None,
            'median': statistics.median(latencies) if latencies else None,
            'p95_nearest_rank': latencies[math.ceil(.95 * len(latencies)) - 1] if latencies else None,
            'max': max(latencies) if latencies else None,
            'scope': 'Existing PredictionAdapter CUDA-synchronized preprocessing/H2D/forward/softmax/Top3/D2H; excludes decoding and model load.',
            'warmup': 'Adapter readiness forward only; all source-image predictions included, including the first.',
        },
    }


def run_diagnostic(rows_path: Path, checkpoint: Path, out: Path, device: str, root: Path = ROOT) -> dict:
    rows_path, checkpoint, out, root = map(Path, (rows_path, checkpoint, out, root))
    if out.exists():
        raise FileExistsError('Output directory already exists; choose a new runs/ subdirectory')
    if not out.resolve().is_relative_to((root / 'runs').resolve()) or out.resolve() == (root / 'runs').resolve():
        raise ValueError('Output must be a new directory underneath this repository runs/')
    if not checkpoint.is_file():
        raise FileNotFoundError('Baseline checkpoint is missing; no predictions were generated')
    if hash_file(checkpoint) != BASELINE_SHA256:
        raise ValueError('Baseline checkpoint SHA256 does not match the approved unchanged checkpoint')
    rows = load_rows(rows_path, root)
    classes_path = root / 'configs/classes.json'
    classes = load_classes(classes_path, require_frozen=True)
    if OVEN_KEYS.intersection(item.key for item in classes.classes) != OVERLAP:
        raise ValueError('Frozen MVP class overlap changed; this diagnostic scope must be reviewed')
    adapter = PredictionAdapter(checkpoint=checkpoint, classes_path=classes_path,
                                expect_checkpoint_sha256=BASELINE_SHA256, device=device)
    info = adapter.status()
    if info.state != 'ready':
        raise RuntimeError(f'Baseline model not ready ({info.state}): {info.message}')
    out.mkdir(parents=True, exist_ok=False)
    predictions, failures = [], []
    print(f'BASELINE ready device={info.device} model={info.model_id} input_images={len(rows)}', flush=True)
    with (out / 'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for index, row in enumerate(rows, 1):
            record = {key: row[key] for key in (
                'relative_path', 'class_key', 'original_split', 'new_split',
                'sha256', 'pixel_sha256', 'group_id')}
            record['supported_label'] = row['class_key'] in OVERLAP
            try:
                record.update(as_contract_dict(adapter.predict(root / row['relative_path'])))
                keys = [item['class_key'] for item in record['predictions']]
                record['top1_matches_source_label'] = keys[0] == row['class_key'] if record['supported_label'] else None
                record['top3_contains_source_label'] = row['class_key'] in keys if record['supported_label'] else None
                predictions.append(record)
            except (OSError, ValueError, RuntimeError) as exc:
                record.update(status='error', error_type=type(exc).__name__,
                              error=str(exc).replace(str(root), '<repo>'))
                failures.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            if index % 50 == 0 or index == len(rows):
                print(f'PROGRESS {index}/{len(rows)} successful={len(predictions)} errors={len(failures)}', flush=True)
    summary = build_summary(rows, predictions, failures)
    import torch
    summary.update(
        completed_utc=datetime.now(timezone.utc).isoformat(),
        rows_sha256=hash_file(rows_path), checkpoint_sha256=info.checkpoint_sha256,
        class_list_sha256=info.class_list_sha256, model_id=info.model_id,
        device=info.device, torch_version=torch.__version__,
        device_name=torch.cuda.get_device_name(torch.device(info.device)) if info.device.startswith('cuda') else 'CPU',
    )
    # Mismatches are local inspection candidates, never automatic label corrections.
    inspection = [item for item in predictions if item['supported_label'] and not item['top1_matches_source_label']]
    for name, payload in [('summary.json', summary), ('prediction_errors.json', failures), ('overlap_mismatches.json', inspection)]:
        (out / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'scope': summary['scope'], 'overlap_raw_diagnostic': summary['all_supplied_rows']['overlap_raw_diagnostic'],
                      'prediction_error_count': len(failures), 'internal_latency_ms': summary['internal_latency_ms']}, ensure_ascii=False), flush=True)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'cuda:0'), required=True)
    args = parser.parse_args(argv)
    try:
        summary = run_diagnostic(args.rows, args.checkpoint, args.out, args.device)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'DIAGNOSTIC ERROR: {exc}', file=sys.stderr)
        return 2
    return 1 if summary['prediction_error_count'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
