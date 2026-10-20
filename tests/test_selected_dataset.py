"""Synthetic verification for the human-selected data pipeline."""
import importlib.util
from pathlib import Path
import random
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('prepare_selected_dataset', SCRIPTS / 'prepare_selected_dataset.py')
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class TestSelectedDataset(unittest.TestCase):
    def test_block_index_matches_brute_force_radius(self):
        rng = random.Random(12)
        values = [rng.getrandbits(64) for _ in range(80)]
        values += [values[0], values[1] ^ 63, values[2] ^ 127, values[3] ^ (1 << 63)]
        rows = [dict(image_id=str(i), canonical_phash=f'{value:016x}') for i, value in enumerate(values)]
        actual = set(pipeline.near_pairs(rows))
        expected = {(str(i), str(j), (values[i] ^ values[j]).bit_count()) for j in range(len(values))
                    for i in range(j) if (values[i] ^ values[j]).bit_count() <= 6}
        self.assertEqual(actual, expected)

    def test_grouped_ratio_reproducible_and_covers_every_class(self):
        rows = [dict(image_id=f'{key}{i}', group_id=f'g{i}', class_key=key)
                for key in ['synthetic_a', 'synthetic_b'] for i in range(100)]
        assigned = pipeline.stratified_groups(rows)
        self.assertEqual(assigned, pipeline.stratified_groups(list(reversed(rows))))
        for key in ['synthetic_a', 'synthetic_b']:
            counts = {s: sum(assigned[r['group_id']] == s for r in rows if r['class_key'] == key)
                      for s in ['train', 'val', 'test']}
            self.assertEqual(counts, {'train': 80, 'val': 10, 'test': 10})
        self.assertEqual(len(assigned), 100)

    def test_label_conflicts_are_quarantined_and_duplicates_have_one_representative(self):
        rows = [dict(image_id='a', input_path='a', status='candidate', pixel_sha256='x', class_key='corn'),
                dict(image_id='b', input_path='b', status='candidate', pixel_sha256='x', class_key='potato'),
                dict(image_id='c', input_path='c', status='candidate', pixel_sha256='y', class_key='shrimp'),
                dict(image_id='d', input_path='d', status='candidate', pixel_sha256='y', class_key='shrimp'),
                dict(image_id='e', input_path='e', status='preaugmented_held_out_of_base_split', pixel_sha256='z', class_key='shrimp')]
        kept = pipeline.deduplicate(rows)
        self.assertEqual([r['image_id'] for r in kept], ['c'])
        self.assertEqual([r['status'] for r in rows], ['label_conflict', 'label_conflict', 'candidate',
                                                    'exact_duplicate', 'preaugmented_held_out_of_base_split'])

    def test_unsplittable_class_fails_instead_of_leaking_group(self):
        rows = [dict(class_key='synthetic_food', group_id='one_capture') for _ in range(100)]
        with self.assertRaisesRegex(ValueError, 'Not enough separable groups'):
            pipeline.stratified_groups(rows)


if __name__ == '__main__':
    unittest.main()
