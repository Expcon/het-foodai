"""Synthetic report/input fixtures only; no food-model quality claims."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    from tools.oven_training.diagnose_baseline import build_summary, load_rows, run_diagnostic
except ModuleNotFoundError:
    build_summary = load_rows = run_diagnostic = None


class OvenBaselineDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(load_rows, 'Baseline diagnostic is not implemented')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'data/raw/oven_round1_20261106/synthetic.png'
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(b'synthetic input-validation bytes, not a real image')
        self.row = {
            'relative_path': self.path.relative_to(self.root).as_posix(),
            'source_id': 'user_supplied_oven_zip_20261106',
            'class_key': 'broccoli', 'original_split': 'test',
            'sha256': hashlib.sha256(self.path.read_bytes()).hexdigest(),
            'pixel_sha256': 'a' * 64, 'group_id': 'synthetic_group', 'new_split': 'train',
        }
        self.rows_file = self.root / 'rows.json'

    def write_rows(self, rows):
        self.rows_file.write_text(json.dumps(rows), encoding='utf-8')

    def test_changed_image_bytes_fail_before_prediction(self):
        self.write_rows([self.row])
        self.assertEqual(load_rows(self.rows_file, self.root), [self.row])
        self.path.write_bytes(b'changed synthetic bytes')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            load_rows(self.rows_file, self.root)

    def test_outside_source_and_old_frozen_dataset_are_rejected(self):
        for path in ('../secret.png', '/absolute.png', 'data/processed/selected_dataset_v1.1.0/x.png'):
            with self.subTest(path=path):
                self.write_rows([dict(self.row, relative_path=path)])
                with self.assertRaises(ValueError):
                    load_rows(self.rows_file, self.root)

    def test_missing_checkpoint_and_existing_output_do_not_overwrite(self):
        self.write_rows([self.row])
        with self.assertRaises(FileNotFoundError):
            run_diagnostic(self.rows_file, self.root / 'missing.pt', self.root / 'runs/new', 'cpu', self.root)
        output = self.root / 'runs/existing'
        output.mkdir(parents=True)
        marker = output / 'keep.txt'
        marker.write_text('keep', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            run_diagnostic(self.rows_file, self.path, output, 'cpu', self.root)
        self.assertEqual(marker.read_text(encoding='utf-8'), 'keep')

    def test_unsupported_labels_and_duplicate_exclusion_never_enter_overlap_denominator(self):
        rows = [self.row, dict(self.row, class_key='toast'), dict(self.row, new_split='excluded')]
        records = [
            dict(row, predictions=[{'class_key': top1, 'score': .99}, {'class_key': 'potato', 'score': .01}],
                 latency_ms=latency, supported_label=row['class_key'] == 'broccoli',
                 top1_matches_source_label=row['class_key'] == top1,
                 top3_contains_source_label=row['class_key'] == top1)
            for row, top1, latency in zip(rows, ('broccoli', 'potato', 'potato'), (10., 20., 30.))
        ]
        summary = build_summary(rows, records, [])
        self.assertEqual(summary['all_supplied_rows']['overlap_raw_diagnostic'], {
            'successful_images': 2, 'top1_correct': 1, 'top3_contains_source_label': 1,
        })
        self.assertEqual(summary['retained_train_rows']['overlap_raw_diagnostic']['successful_images'], 1)
        self.assertIsNone(summary['all_supplied_rows']['per_class']['toast']['top1_correct'])
        self.assertEqual(summary['all_supplied_rows']['unsupported_images'], 1)
        self.assertNotIn('accuracy', summary)
        self.assertEqual(summary['internal_latency_ms']['median'], 20.)


if __name__ == '__main__':
    unittest.main()
