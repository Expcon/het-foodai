"""Synthetic audit fixtures. These tests never load or train a network."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from tools.oven_training.error_audit import CHECKPOINT_SHA, ONNX_SHA, check_mapping, field_candidate, image_record
from tools.oven_training.model import PREPROCESS, class_digest
from tools.oven_training.round2 import V2_KEYS


class OvenErrorAuditTests(unittest.TestCase):
    def fixture(self):
        rows = [{'class_id': 7, 'class_key': 'salmon', 'relative_path': '三文鱼/a.jpg', 'new_split': 'train'}]
        metadata = {'class_keys': V2_KEYS, 'class_list_sha256': class_digest(V2_KEYS),
                    'config': {'class_keys': V2_KEYS, 'provenance': {'per_class_samples': {'salmon': 1}}}}
        contract = {'class_keys': V2_KEYS, 'class_list_sha256': class_digest(V2_KEYS), 'preprocess': PREPROCESS,
                    'classes': [{'id': i, 'key': key} for i, key in enumerate(V2_KEYS)],
                    'source_checkpoint_sha256': CHECKPOINT_SHA, 'models': {'onnx': {'sha256': ONNX_SHA}}}
        return metadata, contract, rows

    def test_mapping_rejects_wrong_ids_directory_or_checkpoint(self):
        metadata, contract, rows = self.fixture()
        self.assertEqual(check_mapping(metadata, contract, rows)['status'], 'PASS')
        for field, value in [('class_id', 1), ('relative_path', '培根/a.jpg')]:
            with self.assertRaises(ValueError): check_mapping(metadata, contract, [{**rows[0], field: value}])
        altered = deepcopy(contract)
        altered['class_keys'] = list(reversed(V2_KEYS))
        with self.assertRaises(ValueError): check_mapping(metadata, altered, rows)
        altered = deepcopy(contract)
        altered['source_checkpoint_sha256'] = 'a'*64
        with self.assertRaises(ValueError): check_mapping(metadata, altered, rows)

    def test_filename_is_only_a_candidate_and_does_not_invent_label(self):
        self.assertEqual(field_candidate('三文鱼识别成培根_0001.png'), 'salmon')
        self.assertEqual(field_candidate('香肠成鸡翅_0001.png'), 'sausage')
        self.assertEqual(field_candidate('potato1_0100.png'), 'potato')
        self.assertEqual(field_candidate('test_0001.png'), 'unknown')

    def test_real_decoder_geometry_and_corrupt_input_leave_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'SYNTHETIC.png'
            Image.new('RGB', (1920,1080), (100,120,130)).save(path)
            before = path.read_bytes()
            row = image_record(path, {'source': 'synthetic', 'human_confirmed': False})
            self.assertEqual(row['content_224_size'], [224,126])
            self.assertEqual(row['decode_status'], 'OK')
            self.assertFalse(row['human_confirmed'])
            self.assertEqual(path.read_bytes(), before)
            path.write_bytes(b'SYNTHETIC corrupt negative fixture')
            self.assertEqual(image_record(path, {})['decode_status'], 'FAIL')
