"""Synthetic contract checks, not food accuracy evidence."""
import unittest
from tools.oven_training.model import check_rows, validate_metadata, class_digest, PREPROCESS


class OvenRound1ModelTests(unittest.TestCase):
    def test_exploratory_training_refuses_validation_or_unknown_classes(self):
        rows=[dict(class_key='potato',new_split='train',group_id='g',relative_path='data/raw/x.png',sha256='a'*64)]
        self.assertEqual(check_rows(rows,['potato']),rows)
        for updates in [dict(new_split='test'),dict(new_split='val'),dict(class_key='unknown'),dict(group_id='')]:
            with self.assertRaises(ValueError):
                check_rows([{**rows[0],**updates}],['potato'])

    def test_checkpoint_rejects_order_preprocess_and_formal_metric_claim(self):
        meta=dict(format='oven-yolo-exploratory/v1',architecture='yolo11n-cls',class_keys=['potato','toast'],
            class_list_sha256=class_digest(['potato','toast']),preprocess=PREPROCESS,independent_validation=False,
            ultralytics_version='8.3.230',epoch=30)
        validate_metadata(meta)
        for updates in [dict(class_keys=['toast','potato']),dict(preprocess={}),dict(independent_validation=True),dict(epoch=0)]:
            with self.assertRaises(ValueError): validate_metadata({**meta,**updates})

    def test_checkpoint_rejects_duplicate_class_keys_and_traversal_rows(self):
        with self.assertRaises(ValueError):class_digest(['potato','potato'])
        with self.assertRaises(ValueError):
            check_rows([dict(class_key='potato',new_split='train',group_id='g',relative_path='../x.png',sha256='a'*64)],['potato'])

    def test_batch_summary_does_not_invent_holdout_or_unlabelled_accuracy(self):
        from tools.oven_training.predict import summarize
        records=[dict(expected='potato',top3=[dict(class_key='toast')]),dict(expected=None,top3=[dict(class_key='potato')])]
        result=summarize(records,['potato','toast'])
        self.assertEqual(result['labelled_images'],1)
        self.assertEqual(result['correct'],0)
        self.assertIsNone(result['independent_validation_accuracy'])
        self.assertIsNone(summarize(records[1:],['potato','toast'])['resubstitution_accuracy'])

    def test_malformed_checkpoint_fails_before_sdk_or_prediction(self):
        import tempfile
        from pathlib import Path
        import torch
        from tools.oven_training.model import load_model
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-checkpoint-') as directory:
            path=Path(directory)/'invalid.pt'
            path.write_bytes(b'SYNTHETIC corrupt checkpoint')
            with self.assertRaises(ValueError):load_model(path)
            torch.save([],path)
            with self.assertRaises(ValueError):load_model(path)
            torch.save({'metadata':{},'model_state':{}},path)
            with self.assertRaises(ValueError):load_model(path)
