"""Synthetic-only ZIP fixtures: never counted as real food data."""
import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
import zipfile

from PIL import Image, PngImagePlugin

from tools.oven_training.audit import audit_archive


def synthetic_png(seed, note=None):
    randomizer = random.Random(seed)
    image = Image.frombytes('RGB', (32, 24), randomizer.randbytes(32 * 24 * 3))
    stream = io.BytesIO()
    metadata = PngImagePlugin.PngInfo()
    if note:
        metadata.add_text('synthetic_note', note)
    image.save(stream, format='PNG', pnginfo=metadata)
    return stream.getvalue()


class OvenRound1AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'synthetic.zip'

    def make_zip(self, files):
        with zipfile.ZipFile(self.source, 'w') as archive:
            for name, payload in files:
                archive.writestr(name, payload)

    def test_pixel_duplicate_cross_split_is_excluded_without_claiming_independent_ids(self):
        self.make_zip([
            ('synthetic/train/toast/toast_0001.png', synthetic_png(1)),
            ('synthetic/test/toast/toast_0048.png', synthetic_png(1, 'different encoding')),
            ('synthetic/val/toast/toast_0060.png', synthetic_png(2)),
        ])
        original_digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        summary, rows = audit_archive(self.source, self.root, phash_threshold=0)
        self.assertEqual(summary['original_split_counts'], {'test': 1, 'train': 1, 'val': 1})
        self.assertEqual(summary['duplicate_counts']['sha256']['groups'], 0)
        self.assertEqual(summary['duplicate_counts']['pixel_sha256']['cross_original_split_groups'], 1)
        self.assertEqual(summary['new_split_counts'], {'excluded': 1, 'train': 2})
        self.assertEqual(len({row['group_id'] for row in rows}), 1)
        self.assertTrue(all(row['physical_sample_id'] is None for row in rows))
        self.assertEqual({row['class_id'] for row in rows}, {0})
        excluded = next(row for row in rows if row['new_split'] == 'excluded')
        self.assertEqual(excluded['exclusion_reason'], 'duplicate_pixels')
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), original_digest)
        output = self.root / 'runs/oven-first-round-20261106'
        self.assertEqual(json.loads((output / 'candidate_rows.json').read_text()), rows)
        self.assertTrue(all((self.root / row['relative_path']).is_file() for row in rows))

    def test_duplicate_pixels_with_conflicting_labels_are_all_excluded_and_groups_joined(self):
        self.make_zip([
            ('synthetic/train/toast/a.png', synthetic_png(5)),
            ('synthetic/test/bacon/b.png', synthetic_png(5)),
            ('synthetic/train/bacon/c.png', synthetic_png(6)),
        ])
        summary, rows = audit_archive(self.source, self.root, phash_threshold=0)
        self.assertEqual(summary['new_split_counts'], {'excluded': 2, 'train': 1})
        self.assertEqual(len({row['group_id'] for row in rows}), 1)
        self.assertEqual({r['exclusion_reason'] for r in rows if r['new_split'] == 'excluded'},
                         {'conflicting_labels_for_identical_pixels'})
        self.assertEqual(summary['experimental_classes'], [{'id': 0, 'key': 'bacon'}, {'id': 1, 'key': 'toast'}])

    def test_unreadable_image_is_retained_as_excluded(self):
        self.make_zip([('synthetic/train/toast/broken.png', b'not an image')])
        summary, rows = audit_archive(self.source, self.root)
        self.assertEqual(summary['image_status_counts'], {'decode_error': 1})
        self.assertEqual(rows[0]['new_split'], 'excluded')
        self.assertEqual(rows[0]['exclusion_reason'], 'decode_error')
        self.assertIsNone(rows[0]['pixel_sha256'])

    def test_unsafe_names_and_case_collisions_are_rejected_before_extraction(self):
        for names in [
            ['../escaped.png'], ['C:/escaped.png'], ['synthetic/train/toast/NUL.png'],
            ['synthetic/train/toast/a.png', 'synthetic/train/toast/A.png'],
        ]:
            with self.subTest(names=names):
                self.make_zip([(name, synthetic_png(3)) for name in names])
                with self.assertRaises(ValueError):
                    audit_archive(self.source, self.root)
                self.assertFalse((self.root / 'data/raw').exists())

    def test_symlink_and_oversized_zip_members_are_rejected_before_extraction(self):
        link = zipfile.ZipInfo('synthetic/train/toast/link.png')
        link.create_system = 3
        link.external_attr = (0o120777 << 16)
        self.make_zip([(link, b'/tmp/target')])
        with self.assertRaises(ValueError):
            audit_archive(self.source, self.root)
        self.make_zip([('synthetic/train/toast/large.png', b'x' * (32 * 1024 * 1024 + 1))])
        with self.assertRaises(ValueError):
            audit_archive(self.source, self.root)
        self.assertFalse((self.root / 'data/raw').exists())

    def test_existing_extraction_is_reused_but_never_overwritten(self):
        self.make_zip([('synthetic/train/toast/a.png', synthetic_png(4))])
        _, rows = audit_archive(self.source, self.root)
        destination = self.root / rows[0]['relative_path']
        _, repeated = audit_archive(self.source, self.root)
        self.assertEqual(rows, repeated)
        destination.write_bytes(b'local modification')
        with self.assertRaises(FileExistsError):
            audit_archive(self.source, self.root)
        self.assertEqual(destination.read_bytes(), b'local modification')

    def test_output_outside_runs_is_rejected(self):
        self.make_zip([('synthetic/train/toast/a.png', synthetic_png(4))])
        with self.assertRaises(ValueError):
            audit_archive(self.source, self.root, output=self.root / 'docs')


if __name__ == '__main__':
    unittest.main()
