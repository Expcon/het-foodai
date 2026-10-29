"""SYNTHETIC CLI chain only: no cameras, food accuracy, SDK or real data required."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/phase2-20261029/oven-offline-smoke.json')
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT / 'runs/phase2-20261029'):
        parser.error('Smoke evidence must stay in its own ignored runs/phase2-20261029 directory')
    raw = ROOT / 'data/raw/oven_capture'
    raw.mkdir(parents=True, exist_ok=True)
    checks = []
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-smoke-', dir=raw) as directory:
        root = Path(directory)
        collection = root / 'collection'
        image = Image.new('RGB', (108, 97))
        rng = random.Random(29)
        image.putdata([tuple(rng.randrange(256) for _ in range(3)) for _ in range(108 * 97)])
        a, b, c, bad = [root / name for name in ['SYNTHETIC-a.png', 'SYNTHETIC-b.png', 'SYNTHETIC-c.png', 'SYNTHETIC-bad.jpg']]
        image.save(a)
        b.write_bytes(a.read_bytes())
        image.save(c, compress_level=0)
        bad.write_text('SYNTHETIC damaged image')
        metadata = root / 'session.json'
        metadata.write_text(json.dumps(dict(camera_id='SYNTHETIC-camera', oven_id='SYNTHETIC-oven',
            physical_sample_ids=['SYNTHETIC-physical'], session_id='SYNTHETIC-session', class_keys=['carrot'],
            placement='SYNTHETIC-front', lighting='SYNTHETIC-light', license='unknown',
            captured_at='2026-10-29T10:00:00+08:00')))

        def run(command, expected=0, *, validator=False):
            entry = ['scripts/validate_manifest.py'] if validator else ['-m', 'tools.oven_capture']
            result = subprocess.run([sys.executable, *entry, *map(str, command)], cwd=ROOT,
                                    capture_output=True, text=True, encoding='utf-8', timeout=45)
            if result.returncode != expected:
                raise AssertionError((command[0], result.returncode, result.stdout, result.stderr))
            checks.append({'case': 'shared_validator' if validator else str(command[0]),
                           'exit_code': result.returncode, 'expected_exit_code': expected})
            return result

        run(['import', '--files', a, b, c, '--metadata', metadata, '--output', collection])
        run(['import', '--files', bad, '--metadata', metadata, '--output', collection], 1)
        (collection / 'images/SYNTHETIC-orphan.png').write_bytes(a.read_bytes())
        run(['audit', '--collection', collection, '--report', root / 'audit-orphan.json'], 1)
        (collection / 'images/SYNTHETIC-orphan.png').unlink()
        run(['audit', '--collection', collection, '--report', root / 'audit-pending.json'])
        audit = json.loads((root / 'audit-pending.json').read_text(encoding='utf-8'))
        assert audit['exact_duplicates'] and audit['pixel_duplicates'] and audit['near_duplicates']
        rows = [json.loads(p.read_text(encoding='utf-8')) for p in (collection / 'records').glob('*.json')]
        assert len(rows) == 3 and all(r['label_status'] == 'pending' for r in rows)
        assignments = root / 'assignments.json'
        assignments.write_text(json.dumps({r['image_id']: 'train' for r in rows}))
        manifest = collection / 'candidate-manifest.csv'
        run(['export', '--collection', collection, '--assignments', assignments, '--out', manifest], 1)
        for row in rows:
            run(['review', '--collection', collection, '--image-id', row['image_id'],
                 '--class-keys', 'carrot', '--reviewer', 'SYNTHETIC-reviewer'])
        cross = {r['image_id']: 'train' for r in rows}
        cross[rows[0]['image_id']] = 'test'
        assignments.write_text(json.dumps(cross))
        run(['export', '--collection', collection, '--assignments', assignments, '--out', manifest], 1)
        assignments.write_text(json.dumps({r['image_id']: 'train' for r in rows}))
        run(['export', '--collection', collection, '--assignments', assignments, '--out', manifest])
        run(['--manifest', manifest, '--classes', 'configs/classes.json', '--check-files'], validator=True)
        other = root / 'second-collection'
        run(['import', '--files', a, '--metadata', metadata, '--output', other])
        extra = json.loads(next((other / 'records').glob('*.json')).read_text(encoding='utf-8'))
        run(['review', '--collection', other, '--image-id', extra['image_id'],
             '--class-keys', 'carrot', '--reviewer', 'SYNTHETIC-reviewer'])
        joint = collection / 'joint-candidate-manifest.csv'
        run(['export', '--collection', collection, '--assignments', assignments, '--out', joint], 1)
        run(['audit', '--collection', collection, other, '--report', root / 'audit-joint.json'])
        cross[extra['image_id']] = 'test'
        assignments.write_text(json.dumps(cross))
        run(['export', '--collection', collection, other, '--assignments', assignments, '--out', joint], 1)
        assignments.write_text(json.dumps({r['image_id']: 'train' for r in [*rows, extra]}))
        run(['export', '--collection', collection, other, '--assignments', assignments, '--out', joint])
        run(['--manifest', joint, '--classes', 'configs/classes.json', '--check-files'], validator=True)
        proof = {'state': 'PASS_SOFTWARE_ONLY', 'synthetic_only': True, 'opened_camera': False,
            'simulated_operator_review_not_ground_truth': True, 'tests': checks,
            'candidate_images': 4, 'exact_pixel_near_duplicates_detected': True,
            'missing_metadata_rejected': True, 'corrupt_image_rejected': True,
            'unreviewed_and_cross_split_export_rejected': True, 'shared_nine_field_validator': 'PASS',
            'partial_catalog_and_cross_collection_split_rejected': True,
            'formal_dataset_created': False, 'temporary_synthetic_files_removed_on_exit': True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(proof, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
