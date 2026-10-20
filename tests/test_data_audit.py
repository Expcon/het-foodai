"""Synthetic QA fixtures only; no fixtures count as real dataset samples."""
import importlib.util
import contextlib
import csv
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

from PIL import Image

SPEC = importlib.util.spec_from_file_location('audit_images', Path(__file__).resolve().parents[1] / 'scripts/audit_images.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
GROUP_SPEC = importlib.util.spec_from_file_location('propose_groups', Path(__file__).resolve().parents[1] / 'scripts/propose_groups.py')
grouping = importlib.util.module_from_spec(GROUP_SPEC)
GROUP_SPEC.loader.exec_module(grouping)


class TestDataAudit(unittest.TestCase):
    def test_bktree_matches_bruteforce_with_duplicate_hashes(self):
        randomizer = random.Random(77)
        values = [randomizer.getrandbits(64) for _ in range(80)]
        values += [values[0], values[0] ^ 7, values[1] ^ 63]
        tree = audit.BKTree()
        for index, value in enumerate(values):
            tree.add(value, index)
        for query in values:
            expected = sorted((i, (query ^ value).bit_count()) for i, value in enumerate(values)
                              if (query ^ value).bit_count() <= 6)
            self.assertEqual(sorted(tree.find(query, 6)), expected)

    def test_rejects_zip_traversal_and_windows_absolute_paths(self):
        for name in ['../photo.jpg', '/photo.jpg', 'C:/photo.jpg', '..\\photo.jpg']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit.safe_member(name)
        self.assertEqual(str(audit.safe_member('train/photo.jpg')), 'train/photo.jpg')

    def test_box_bounds_and_nan_are_rejected(self):
        for box in [[-1, 0, 10, 10], [0, 0, 101, 10], [0, 0, 0, 10], [0, 0, float('nan'), 10]]:
            with self.subTest(box=box), self.assertRaises(ValueError):
                audit.crop_box(box, (100, 100))
        self.assertEqual(audit.crop_box([0.5, 1.5, 20, 30], (100, 100)), (0, 1, 21, 32))

    def test_byte_and_pixel_duplicate_leakage_are_distinguished(self):
        rows = [dict(image_id='synthetic-a', source_id='s', original_split='train', sha256='a', pixel_sha256='p'),
                dict(image_id='synthetic-b', source_id='s', original_split='test', sha256='b', pixel_sha256='p')]
        self.assertEqual(audit.duplicate_groups(rows, 'sha256'), [])
        groups = audit.duplicate_groups(rows, 'pixel_sha256')
        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0]['cross_original_split'])

    def test_taxonomy_does_not_turn_generic_fish_into_sea_bass(self):
        self.assertNotIn('fish', audit.MAPPING)
        self.assertNotIn('shrimp', audit.MAPPING)
        self.assertEqual(audit.MAPPING[audit.normalized('chicken_breast')], 'chicken_breast')
        self.assertEqual(audit.BLOCKED['fish'], 'sea_bass_unverified')

    def test_phash_survives_lossless_encoding(self):
        randomizer = random.Random(31)
        image = Image.new('RGB', (100, 80))
        image.putdata([tuple(randomizer.randrange(256) for _ in range(3)) for _ in range(8000)])
        stream = io.BytesIO()
        image.save(stream, format='PNG')
        with Image.open(io.BytesIO(stream.getvalue())) as decoded:
            self.assertEqual(audit.phash(image), audit.phash(decoded))

    def test_groups_join_cross_source_duplicates_parents_and_transitive_edges(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = [dict(image_id=key, sha256='same' if key in 'ab' else key, pixel_sha256=key)
                      for key in 'abcde']
            samples = [dict(image_id=key, parent_image_id=parent, source_id='synthetic_' + parent,
                            class_key='synthetic_food', review_status='needs_review')
                       for key, parent in [('a1', 'a'), ('a2', 'a'), ('b1', 'b'), ('c1', 'c'), ('d1', 'd'), ('e1', 'e')]]
            for filename, rows in [('images.csv', images), ('human_review.csv', samples),
                                   ('original_near_pairs.csv', [dict(left='a', right='c')]),
                                   ('candidate_near_pairs.csv', [dict(left='c1', right='d1')])]:
                with (root / filename).open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            with contextlib.redirect_stdout(io.StringIO()):
                grouping.propose(root)
            result = grouping.read_csv(root / 'group_proposals.csv')
            groups = {r['image_id']: r['proposed_group_id'] for r in result}
            self.assertEqual(len({groups[key] for key in ['a1', 'a2', 'b1', 'c1', 'd1']}), 1)
            self.assertNotEqual(groups['a1'], groups['e1'])
            self.assertTrue(all('split' not in row for row in result))
            summary = json.loads((root / 'group_summary.json').read_text())
            self.assertEqual(summary['candidate_groups'], 2)


if __name__ == '__main__':
    unittest.main()
