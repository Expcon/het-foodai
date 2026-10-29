"""Candidate-only oven capture CLI. Camera commands require explicit selection."""
import argparse
import json
from pathlib import Path
import sys

from . import core
from .camera import OpenCVSource, inventory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    devices = commands.add_parser('devices', help='Passive inventory only; never opens cameras')
    devices.add_argument('--out', type=Path, required=True)
    importing = commands.add_parser('import', help='Copy explicit offline JPG/PNG files without changing bytes')
    importing.add_argument('--files', nargs='+', type=Path, required=True)
    importing.add_argument('--metadata', type=Path, required=True)
    importing.add_argument('--output', type=Path, required=True)
    auditing = commands.add_parser('audit')
    auditing.add_argument('--collection', type=Path, nargs='+', required=True)
    auditing.add_argument('--report', type=Path, required=True)
    auditing.add_argument('--near-threshold', type=int, default=6)
    reviewing = commands.add_parser('review', help='Operator confirms labels after viewing each image')
    reviewing.add_argument('--collection', type=Path, required=True)
    reviewing.add_argument('--image-id', required=True)
    reviewing.add_argument('--class-keys', nargs='*', required=True)
    reviewing.add_argument('--reviewer', required=True)
    exporting = commands.add_parser('export', help='Explicit guarded nine-field candidate handoff, never freeze')
    exporting.add_argument('--collection', type=Path, nargs='+', required=True)
    exporting.add_argument('--assignments', type=Path, required=True)
    exporting.add_argument('--out', type=Path, required=True)
    for name in ['probe', 'capture']:
        camera = commands.add_parser(name)
        camera.add_argument('--device', required=True)
        camera.add_argument('--backend', choices=['dshow', 'msmf', 'v4l2'], required=True)
        camera.add_argument('--output', type=Path, required=True)
        camera.add_argument('--cold-confirmed', action='store_true', required=True)
        camera.add_argument('--width', type=int)
        camera.add_argument('--height', type=int)
        if name == 'capture':
            camera.add_argument('--metadata', type=Path, required=True)
            camera.add_argument('--count', type=int, default=1)
            camera.add_argument('--interval', type=float, default=0.)
    args = parser.parse_args(argv)
    try:
        if args.command == 'devices':
            core.guard_output(args.out)
            result = inventory()
            core._write_json(args.out, result)
            print(json.dumps({'state': result['state'], 'enumerated_devices': len(result['devices']),
                              'opened_any_device': False, 'details': 'saved locally; device IDs are not for Git'}, ensure_ascii=False))
            return 2 if result['state'] == 'BLOCKED' else 0
        if args.command == 'import':
            core.guard_output(args.output)
            rows = core.import_files(args.files, args.output, json.loads(args.metadata.read_text(encoding='utf-8-sig')))
            result = {'state': 'CANDIDATES_PENDING_HUMAN_REVIEW', 'images': [r['image_id'] for r in rows], 'count': len(rows)}
        elif args.command == 'audit':
            for collection in args.collection:
                core.guard_output(collection)
            core.guard_output(args.report)
            result = core.audit(args.collection, near_threshold=args.near_threshold)
            core._write_json(args.report, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1 if result['state'] == 'FAIL' else 0
        elif args.command == 'review':
            core.guard_output(args.collection)
            row = core.review(args.collection, args.image_id, args.class_keys, args.reviewer)
            result = {'state': 'HUMAN_LABEL_CONFIRMED_NOT_DATASET_FROZEN', 'image_id': row['image_id']}
        elif args.command == 'export':
            for collection in args.collection:
                core.guard_output(collection)
            core.guard_output(args.out)
            result = core.export_manifest(args.collection, json.loads(args.assignments.read_text(encoding='utf-8-sig')), args.out)
        else:
            core.guard_output(args.output)
            core._check_layout(args.output.resolve())
            metadata = None
            if args.command == 'capture':
                import math
                if not 1 <= args.count <= 100 or not math.isfinite(args.interval) or not 0 <= args.interval <= 60:
                    raise ValueError('count must be 1..100; interval must be finite 0..60 seconds')
                metadata = core._metadata({**json.loads(args.metadata.read_text(encoding='utf-8-sig')), 'captured_at': core.now()})
            source = OpenCVSource(args.device, args.backend, width=args.width, height=args.height)
            try:
                if args.command == 'probe':
                    image, details = source.read()
                    result = {'state': 'SELECTED_CAMERA_FRAME_READ_NOT_OVEN_ACCEPTED', 'observed_at': core.now(),
                              'device': args.device, 'frame_size': list(image.size), 'details': details}
                    core._write_json(args.output / 'probe.json', result)
                else:
                    rows = core.capture_frames(source, args.output, metadata, count=args.count, interval=args.interval)
                    result = {'state': 'CANDIDATES_PENDING_HUMAN_REVIEW', 'count': len(rows), 'images': [r['image_id'] for r in rows]}
            finally:
                source.close()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, core.NotReady) as exc:
        print(json.dumps({'state': 'BLOCKED' if isinstance(exc, core.NotReady) else 'FAIL',
                          'reason': str(exc), 'formal_dataset_frozen': False}, ensure_ascii=False), file=sys.stderr)
        return 2 if isinstance(exc, core.NotReady) else 1


if __name__ == '__main__':
    raise SystemExit(main())
