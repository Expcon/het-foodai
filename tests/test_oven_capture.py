"""SYNTHETIC offline fixtures: software checks, never camera/food accuracy evidence."""
import csv
import contextlib
from datetime import datetime, timezone
import json
import io
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from foodai.contracts import load_classes, validate_manifest
from tools.oven_capture import core


class TestOvenCapture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='SYNTHETIC-oven-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.collection = self.root / 'collection'
        rng = random.Random(83)
        self.image = self.root / 'SYNTHETIC.png'
        image = Image.new('RGB', (108, 97))
        image.putdata([tuple(rng.randrange(256) for _ in range(3)) for _ in range(108 * 97)])
        image.save(self.image)
        self.metadata = dict(captured_at='2026-10-30T09:00:00+08:00', camera_id='SYNTHETIC-camera',
            oven_id='SYNTHETIC-oven', physical_sample_ids=['SYNTHETIC-sample'], session_id='SYNTHETIC-session',
            class_keys=['carrot'], placement='SYNTHETIC-front', lighting='SYNTHETIC-light', license='unknown')

    def ingest(self, **changes):
        return core.import_files([self.image], self.collection, {**self.metadata, **changes})[0]

    def test_import_preserves_bytes_size_and_requires_review(self):
        row = self.ingest()
        self.assertEqual((self.collection / row['relative_path']).read_bytes(), self.image.read_bytes())
        self.assertEqual((row['width'], row['height'], row['format']), (108, 97, 'PNG'))
        self.assertEqual(row['label_status'], 'pending')
        self.assertEqual(row['representation'], 'imported_encoded_file')
        self.assertFalse(row['sensor_raw_claimed'])
        self.assertIn('imported_at', row)

    def test_missing_naive_or_unknown_identity_metadata_fails_before_write(self):
        for updates in [dict(camera_id=''), dict(captured_at='2026-10-30T09:00:00'),
                        dict(physical_sample_ids=[]), dict(session_id='UNKNOWN'), dict(class_keys=['fake_food'])]:
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.ingest(**updates)
        self.assertFalse(self.collection.exists())

    def test_corrupt_input_is_rejected_without_saving(self):
        self.image.write_bytes(b'SYNTHETIC damaged image')
        with self.assertRaises(ValueError):
            self.ingest()
        self.assertFalse(self.collection.exists())

    def test_wrong_metadata_container_types_fail_as_validation_errors(self):
        for updates in [dict(class_keys=[{}]), dict(physical_sample_ids=[{}])]:
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.ingest(**updates)

    def test_identical_pixels_with_conflicting_confirmed_labels_block_export(self):
        a, b = self.ingest(), self.ingest()
        core.review(self.collection, a['image_id'], ['carrot'], 'SYNTHETIC-reviewer')
        core.review(self.collection, b['image_id'], ['potato'], 'SYNTHETIC-reviewer')
        self.assertIn('label_conflict', {i['code'] for i in core.audit(self.collection)['issues']})
        with self.assertRaises(ValueError):
            core.export_manifest(self.collection, {a['image_id']: 'train', b['image_id']: 'train'},
                                 self.collection / 'manifest.csv', repo_root=self.root)

    def test_symlinked_image_directory_cannot_escape_collection(self):
        outside = self.root / 'outside'
        outside.mkdir()
        self.collection.mkdir()
        try:
            (self.collection / 'images').symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest('OS does not permit test-directory symlinks')
        with self.assertRaises(ValueError):
            self.ingest()
        self.assertEqual(list(outside.iterdir()), [])

    def test_audit_finds_exact_pixel_near_and_missing_metadata(self):
        first = self.ingest()
        second = self.ingest(physical_sample_ids=['other'], session_id='other-session')
        Image.open(self.image).save(self.root / 'reencoded.png', compress_level=0)
        third = core.import_files([self.root / 'reencoded.png'], self.collection,
                                 {**self.metadata, 'session_id': 'third', 'physical_sample_ids': ['third']})[0]
        (self.collection / 'images' / 'orphan.png').write_bytes(self.image.read_bytes())
        result = core.audit(self.collection)
        self.assertTrue(result['exact_duplicates'])
        self.assertTrue(result['pixel_duplicates'])
        self.assertTrue(result['near_duplicates'])
        self.assertTrue(any(i['code'] == 'missing_metadata' for i in result['issues']))
        self.assertEqual(len(set(result['groups'].values())), 1)

    def test_audit_flags_tamper_and_malformed_sidecar(self):
        row = self.ingest()
        (self.collection / row['relative_path']).write_bytes(b'broken')
        (self.collection / 'records' / 'bad.json').write_text('{', encoding='utf-8')
        codes = {i['code'] for i in core.audit(self.collection)['issues']}
        self.assertIn('unreadable_image', codes)
        self.assertIn('invalid_metadata', codes)

    def test_sample_session_connections_are_transitive(self):
        # Different image bytes eliminate accidental duplicate edges in this grouping check.
        rows = []
        for index, (sample, session) in enumerate([('A', 'one'), ('B', 'one'), ('B', 'two')]):
            rng = random.Random(index + 300)
            image = Image.new('RGB', (130, 100))
            image.putdata([tuple(rng.randrange(256) for _ in range(3)) for _ in range(13000)])
            image.save(self.image)
            rows.append(self.ingest(physical_sample_ids=[sample], session_id=session))
        result = core.audit(self.collection, near_threshold=0)
        self.assertEqual(len({result['groups'][r['image_id']] for r in rows}), 1)

    def test_export_requires_review_and_rejects_cross_split_group(self):
        a, b = self.ingest(), self.ingest()
        out = self.collection / 'manifest.csv'
        assignments = {a['image_id']: 'train', b['image_id']: 'test'}
        with self.assertRaises(ValueError):
            core.export_manifest(self.collection, assignments, out, repo_root=self.root)
        for row in [a, b]:
            core.review(self.collection, row['image_id'], ['carrot'], 'SYNTHETIC-reviewer')
        with self.assertRaisesRegex(ValueError, 'split'):
            core.export_manifest(self.collection, assignments, out, repo_root=self.root)
        self.assertFalse(out.exists())

    def test_reviewed_export_matches_existing_nine_field_contract(self):
        row = self.ingest()
        core.review(self.collection, row['image_id'], ['carrot'], 'SYNTHETIC-reviewer')
        out = self.collection / 'manifest.csv'
        core.export_manifest(self.collection, {row['image_id']: 'train'}, out, repo_root=self.root)
        config = load_classes(ROOT / 'configs/classes.json', require_frozen=True)
        records, report = validate_manifest(out, config, repo_root=self.root, check_files=True)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].class_key, 'carrot')
        with out.open(encoding='utf-8-sig') as stream:
            self.assertEqual(len(next(csv.reader(stream))), 9)

    def test_multi_food_record_cannot_be_exported_as_single_class(self):
        row = self.ingest(class_keys=['carrot', 'potato'], physical_sample_ids=['A', 'B'])
        core.review(self.collection, row['image_id'], ['carrot', 'potato'], 'SYNTHETIC-reviewer')
        with self.assertRaises(ValueError):
            core.export_manifest(self.collection, {row['image_id']: 'train'},
                                 self.collection / 'manifest.csv', repo_root=self.root)

    def test_empty_scene_is_recordable_but_never_a_classification_label(self):
        row = self.ingest(class_keys=[], physical_sample_ids=[])
        self.assertEqual(row['label_status'], 'pending')
        core.review(self.collection, row['image_id'], [], 'SYNTHETIC-reviewer')
        with self.assertRaises(ValueError):
            core.export_manifest(self.collection, {row['image_id']: 'train'},
                                 self.collection / 'manifest.csv', repo_root=self.root)

    def test_review_cannot_confirm_food_without_physical_identity_or_change_record(self):
        row = self.ingest(class_keys=[], physical_sample_ids=[])
        path = self.collection / 'records' / (row['image_id'] + '.json')
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'physical_sample_ids'):
            core.review(self.collection, row['image_id'], ['carrot'], 'SYNTHETIC-reviewer')
        self.assertEqual(path.read_bytes(), before)
        self.assertNotEqual(core.audit(self.collection)['state'], 'FAIL')

    def test_absent_license_is_normalized_to_unknown_for_handoff(self):
        row = self.ingest()
        core.review(self.collection, row['image_id'], ['carrot'], 'SYNTHETIC-reviewer')
        path = self.collection / 'records' / (row['image_id'] + '.json')
        data = json.loads(path.read_text())
        del data['license']
        path.write_text(json.dumps(data))
        out = self.collection / 'manifest.csv'
        core.export_manifest(self.collection, {row['image_id']: 'train'}, out, repo_root=self.root)
        with out.open(encoding='utf-8') as stream:
            self.assertEqual(next(csv.DictReader(stream))['license'], 'unknown')

    def test_joint_export_checks_cross_collection_groups_and_rejects_partial_catalog(self):
        first = self.ingest()
        other = self.root / 'second-collection'
        # Different pixels ensure that the physical/session metadata is tested.
        Image.new('RGB', (108, 97), 'blue').save(self.image)
        second = core.import_files([self.image], other, self.metadata)[0]
        for collection, row in [(self.collection, first), (other, second)]:
            core.review(collection, row['image_id'], ['carrot'], 'SYNTHETIC-reviewer')
        out = self.collection / 'joint-manifest.csv'
        with self.assertRaisesRegex(ValueError, 'catalog'):
            core.export_manifest(self.collection, {first['image_id']: 'train'}, out, repo_root=self.root)
        collections = [self.collection, other]
        assignments = {first['image_id']: 'train', second['image_id']: 'test'}
        with self.assertRaisesRegex(ValueError, 'split'):
            core.export_manifest(collections, assignments, out, repo_root=self.root)
        assignments[second['image_id']] = 'train'
        core.export_manifest(collections, assignments, out, repo_root=self.root)
        with out.open(encoding='utf-8') as stream:
            records = list(csv.DictReader(stream))
        self.assertEqual(len(records), 2)
        self.assertEqual(len({r['group_id'] for r in records}), 1)

    def test_interrupted_images_only_collection_cannot_escape_catalog_audit(self):
        row = self.ingest()
        core.review(self.collection, row['image_id'], ['carrot'], 'SYNTHETIC-reviewer')
        interrupted = self.root / 'interrupted'
        (interrupted / 'images').mkdir(parents=True)
        (interrupted / 'images' / 'orphan.png').write_bytes(self.image.read_bytes())
        out = self.collection / 'manifest.csv'
        with self.assertRaisesRegex(ValueError, 'catalog'):
            core.export_manifest(self.collection, {row['image_id']: 'train'}, out, repo_root=self.root)
        with self.assertRaisesRegex(ValueError, 'Audit errors'):
            core.export_manifest([self.collection, interrupted], {row['image_id']: 'train'}, out, repo_root=self.root)
        self.assertFalse(out.exists())

    def test_traversal_sidecar_cannot_read_or_review_outside_collection(self):
        row = self.ingest()
        path = self.collection / 'records' / (row['image_id'] + '.json')
        data = json.loads(path.read_text())
        data['relative_path'] = '../SYNTHETIC.png'
        path.write_text(json.dumps(data))
        self.assertTrue(core.audit(self.collection)['issues'])
        with self.assertRaises(ValueError):
            core.review(self.collection, '../escape', ['carrot'], 'SYNTHETIC-reviewer')

    def test_cli_output_guard_rejects_protected_dataset_and_source_paths(self):
        for p in [ROOT / 'data/manifest.csv', ROOT / 'configs', ROOT / 'data/processed/selected_dataset_v1.1.0']:
            with self.subTest(p=p), self.assertRaises(ValueError):
                core.guard_output(p)
        core.guard_output(ROOT / 'data/raw/oven_capture/SYNTHETIC')

    def test_finite_frame_source_is_replaceable_and_saves_decoded_png(self):
        class SyntheticSource:
            calls = 0
            def read(self):
                self.calls += 1
                return Image.new('RGB', (192, 108), (self.calls, 30, 40)), {'backend': 'SYNTHETIC', 'reported_fourcc': 'TEST'}
        source = SyntheticSource()
        rows = core.capture_frames(source, self.collection, self.metadata, count=2)
        self.assertEqual(source.calls, 2)
        self.assertTrue(all(r['representation'] == 'decoded_frame_png' for r in rows))
        self.assertTrue(all(not r['sensor_raw_claimed'] for r in rows))
        self.assertTrue(all((r['width'], r['height']) == (192, 108) for r in rows))
        with self.assertRaises(ValueError):
            core.capture_frames(source, self.collection, self.metadata, count=0)

    def test_invalid_cli_frame_count_cannot_open_a_camera(self):
        from tools.oven_capture import __main__ as cli
        with patch.object(cli, 'OpenCVSource', side_effect=AssertionError('Camera must remain unopened')), \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            code = cli.main(['capture', '--device', '0', '--backend', 'dshow', '--cold-confirmed',
                '--output', str(core.RAW_ROOT / 'SYNTHETIC-unused'), '--metadata', 'not-read.json', '--count', '0'])
        self.assertEqual(code, 1)
        self.assertIn('count', errors.getvalue())

    def test_backend_exception_is_not_ready_and_releases_failed_initialization(self):
        from tools.oven_capture.camera import OpenCVSource
        class SyntheticBackendError(Exception):
            pass
        class SyntheticCapture:
            closed = False
            def isOpened(self): return True
            def set(self, *args): raise SyntheticBackendError('UNIT driver failure')
            def release(self): self.closed = True
        cap = SyntheticCapture()
        cv2 = SimpleNamespace(VideoCapture=lambda *args: cap, error=SyntheticBackendError,
            CAP_DSHOW=1, CAP_MSMF=2, CAP_V4L2=3, CAP_PROP_FRAME_WIDTH=4, CAP_PROP_FRAME_HEIGHT=5)
        with patch.dict(sys.modules, {'cv2': cv2}), self.assertRaises(core.NotReady):
            OpenCVSource('0', 'dshow', width=1920, height=1080)
        self.assertTrue(cap.closed)

    def test_unreported_camera_properties_remain_unknown_not_requested_resolution(self):
        import numpy as np
        from tools.oven_capture.camera import OpenCVSource
        cap = SimpleNamespace(isOpened=lambda: True, release=lambda: None,
            read=lambda: (True, np.zeros((2, 3, 3), dtype=np.uint8)), get=lambda key: float('nan'))
        cv2 = SimpleNamespace(VideoCapture=lambda *args: cap, error=RuntimeError, cvtColor=lambda a, code: a[..., ::-1],
            CAP_DSHOW=1, CAP_MSMF=2, CAP_V4L2=3, CAP_PROP_FRAME_WIDTH=4, CAP_PROP_FRAME_HEIGHT=5,
            CAP_PROP_FPS=6, CAP_PROP_FOURCC=7, COLOR_BGR2RGB=8)
        with patch.dict(sys.modules, {'cv2': cv2}):
            source = OpenCVSource('0', 'dshow')
            image, details = source.read()
            source.close()
        self.assertEqual(image.size, (3, 2))
        self.assertEqual(details['actual_decoded_resolution'], [3, 2])
        self.assertEqual(details['reported_resolution'], [None, None])
        self.assertIsNone(details['reported_fps_not_measured'])
        self.assertEqual(details['reported_fourcc'], 'UNKNOWN')


if __name__ == '__main__':
    unittest.main()
