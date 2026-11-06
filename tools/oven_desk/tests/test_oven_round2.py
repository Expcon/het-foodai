"""Synthetic v2 boundary tests only; no training, accuracy or hardware evidence."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps

from inference_engine import ModelContractError, ModelEngine, prepare_input, read_image_unicode
import oven_capture_gui as gui


KEYS = ['asparagus', 'bacon', 'broccoli', 'chicken_wing', 'croissant',
        'french_fries', 'potato', 'salmon', 'toast', 'corn', 'sausage', 'steak']
CLASS_SHA = 'eeb5260a718f2a1bd42a4d600c8cb2e5ee290c98a5a24e38aa1822e7391827f7'


class SyntheticRound2Session:
    """Replace only ONNX Runtime execution; never use as a demo prediction."""
    shape = [1, 3, 224, 224]
    output_shape = [1, 12]
    output = np.array([[.01] * 11 + [.89]], dtype=np.float32)
    metadata = {'het_contract_format': 'het-oven-classifier/v2'}

    def __init__(self, *args, **kwargs):
        self.inputs_seen = []

    def get_inputs(self):
        return [types.SimpleNamespace(name='images', shape=self.shape, type='tensor(float)')]

    def get_outputs(self):
        return [types.SimpleNamespace(name='output', shape=self.output_shape, type='tensor(float)')]

    def get_modelmeta(self):
        return types.SimpleNamespace(custom_metadata_map=self.metadata)

    def run(self, names, feed):
        self.inputs_seen.append(feed['images'].copy())
        return [self.output.copy()]


class OvenRound2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='SYNTHETIC-oven-round2-')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.model = self.folder/'synthetic.onnx'
        self.model.write_bytes(b'SYNTHETIC boundary fixture, not an ONNX model')
        self.labels = self.folder/'classes.json'
        self.contract = dict(format='het-oven-classifier/v2', class_keys=KEYS,
            class_list_sha256=CLASS_SHA, classes=[dict(id=i, key=k) for i, k in enumerate(KEYS)],
            preprocess=dict(size=224, color='RGB', resize='bilinear-contain-center-pad',
                            pad=114, scale='1/255', layout='NCHW'),
            source_checkpoint_sha256='b'*64,
            models={'onnx': {'sha256': hashlib.sha256(self.model.read_bytes()).hexdigest()}},
            input_shape=[1, 3, 224, 224], output_shape=[1, 12],
            output_kind='classification-probabilities', independent_validation=False)
        self.write_contract()
        runtime = patch.dict(sys.modules, {'onnxruntime': types.SimpleNamespace(InferenceSession=SyntheticRound2Session)})
        runtime.start()
        self.addCleanup(runtime.stop)

    def write_contract(self):
        self.labels.write_text(json.dumps(self.contract), encoding='utf-8')

    def engine(self):
        return ModelEngine(self.model, self.labels, 'oven_round2')

    def test_ready_executes_forward_and_preserves_twelve_class_order(self):
        engine = self.engine()
        self.assertEqual(engine.labels, KEYS)
        self.assertEqual(len(engine.session.inputs_seen), 1)
        np.testing.assert_array_equal(engine.session.inputs_seen[0], np.zeros((1,3,224,224), np.float32))
        result = engine.predict(np.zeros((30,90,3), np.uint8))
        self.assertEqual(result['top3'][0]['id'], 11)
        self.assertEqual(result['top3'][0]['label'], 'steak')
        self.assertEqual(len(result['probabilities']), 12)
        self.assertFalse(result['independent_validation'])
        self.assertNotIn('accuracy', json.dumps(result).lower())

    def test_changed_order_hash_contract_or_model_is_rejected(self):
        changes = [dict(format='het-oven-classifier/v1'), dict(class_keys=KEYS[::-1]),
                   dict(class_list_sha256='0'*64),
                   dict(classes=[dict(id=i, key=k) for i,k in enumerate(KEYS[::-1])]),
                   dict(preprocess={}), dict(source_checkpoint_sha256='missing'),
                   dict(models={}), dict(input_shape=[2,3,224,224]), dict(output_shape=[1,9]),
                   dict(output_kind='logits'), dict(independent_validation=True)]
        for change in changes:
            with self.subTest(change=change):
                self.labels.write_text(json.dumps({**self.contract, **change}), encoding='utf-8')
                with self.assertRaises(ModelContractError): self.engine()
        self.write_contract()
        self.model.write_bytes(b'SYNTHETIC altered fixture')
        with self.assertRaises(ModelContractError): self.engine()

    def test_runtime_requires_static_twelve_probabilities(self):
        for shape in [[2,3,224,224], [1,3,'h','w'], [1,3,320,320]]:
            with self.subTest(shape=shape), patch.object(SyntheticRound2Session, 'shape', shape):
                with self.assertRaises(ModelContractError): self.engine()
        with patch.object(SyntheticRound2Session, 'output_shape', [1,9]):
            with self.assertRaises(ModelContractError): self.engine()
        for value in [np.ones((1,12), np.float32), np.zeros(12, np.float32),
                      np.full((1,12), np.nan, np.float32), np.ones((1,12), np.float64)/12,
                      np.ones((1,9), np.float32)/9,
                      np.array([[-.01, .01] + [0]*9 + [1]], np.float32)]:
            with self.subTest(output=value), patch.object(SyntheticRound2Session, 'output', value):
                with self.assertRaises(ModelContractError): self.engine()

    def test_wrong_explicit_mode_and_generic_modes_reject_v2_contract(self):
        for mode in ['oven_round1', 'yolo_classify', 'efficientnet_imagenet', 'yolo_detect']:
            with self.subTest(mode=mode), self.assertRaises(ModelContractError):
                ModelEngine(self.model, self.labels, mode)
        plain = self.folder/'plain.txt'
        plain.write_text('unrelated\nlabels', encoding='utf-8')
        with self.assertRaises(ModelContractError):
            ModelEngine(self.model, plain, 'yolo_classify')
        self.labels.unlink()
        plain.write_text('\n'.join(KEYS), encoding='utf-8')
        with patch.object(SyntheticRound2Session, 'metadata', {}):
            with self.assertRaises(ModelContractError):
                ModelEngine(self.model, plain, 'yolo_classify')

    def test_generic_modes_reject_v2_metadata_without_sidecar(self):
        self.labels.unlink()
        plain = self.folder/'plain.txt'
        plain.write_text('unrelated\nlabels', encoding='utf-8')
        for mode in ['yolo_classify', 'efficientnet_imagenet', 'yolo_detect']:
            with self.subTest(mode=mode), self.assertRaisesRegex(ModelContractError, 'oven_round2'):
                ModelEngine(self.model, plain, mode)

    def test_v2_mode_rejects_v1_onnx_metadata(self):
        with patch.object(SyntheticRound2Session, 'metadata', {'het_contract_format': 'het-oven-classifier/v1'}):
            with self.assertRaises(ModelContractError): self.engine()

    def test_v2_preprocessing_matches_full_frame_training_recipe(self):
        for h, w in [(53,191), (191,53), (223,224)]:
            with self.subTest(size=(h,w)):
                rgb = np.random.default_rng(24).integers(0, 256, (h,w,3), dtype=np.uint8)
                contained = ImageOps.contain(Image.fromarray(rgb), (224,224), Image.Resampling.BILINEAR)
                canvas = Image.new('RGB', (224,224), (114,114,114))
                canvas.paste(contained, ((224-contained.width)//2, (224-contained.height)//2))
                expected = np.asarray(canvas, dtype=np.float32).transpose(2,0,1)[None]/255
                actual, context = prepare_input(rgb[:,:,::-1], 'oven_round2', 224, 224)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.float32)
                self.assertTrue(actual.flags.c_contiguous)
                self.assertIsNone(context)
        with self.assertRaises(ModelContractError):
            prepare_input(np.zeros((10,20,3), np.uint8), 'oven_round2', 320, 320)

    def test_v2_file_decoder_applies_exif_orientation(self):
        path = self.folder/'旋转照片.jpg'
        image = Image.fromarray(np.arange(60, dtype=np.uint8).reshape(4,5,3))
        exif = image.getexif()
        exif[274] = 6
        image.save(path, exif=exif)
        with Image.open(path) as source:
            expected = np.asarray(ImageOps.exif_transpose(source).convert('RGB'))[:,:,::-1]
        np.testing.assert_array_equal(read_image_unicode(path, mode='oven_round2'), expected)

    def test_selfcheck_round2_reports_scope_and_preserves_inputs(self):
        image = self.folder/'SYNTHETIC.png'
        Image.new('RGB', (30,80)).save(image)
        report = self.folder/'report.json'
        with patch.object(sys, 'stdout', io.StringIO()):
            self.assertEqual(gui.model_selfcheck(self.model, self.labels, image, report, mode='oven_round2'), 0)
        obj = json.loads(report.read_text('utf-8'))
        self.assertEqual(obj['mode'], 'oven_round2')
        self.assertEqual(obj['result']['top3'][0]['label'], 'steak')
        self.assertFalse(obj['independent_validation'])
        self.assertNotIn('accuracy', json.dumps(obj).lower())
        before = image.read_bytes()
        with patch.object(sys, 'stderr', io.StringIO()):
            self.assertEqual(gui.model_selfcheck(self.model, self.labels, image, image, mode='oven_round2'), 1)
        self.assertEqual(image.read_bytes(), before)

    def test_gui_and_cli_route_round2_explicitly(self):
        app = gui.DesktopApp.__new__(gui.DesktopApp)
        app.model_task = types.SimpleNamespace(get=lambda: '二轮烤箱分类（实验·12类）')
        self.assertEqual(app._model_mode(), 'oven_round2')
        image = self.folder/'SYNTHETIC.png'
        Image.new('RGB', (30,80)).save(image)
        report = self.folder/'cli.json'
        args = ['oven_capture_gui.py', '--model-selfcheck', '--mode', 'oven_round2',
                '--model', str(self.model), '--classes', str(self.labels),
                '--image', str(image), '--out', str(report)]
        with patch.object(sys, 'argv', args), patch.object(sys, 'stdout', io.StringIO()):
            with self.assertRaises(SystemExit) as exit_result: gui.main()
        self.assertEqual(exit_result.exception.code, 0)
        self.assertEqual(json.loads(report.read_text('utf-8'))['mode'], 'oven_round2')


if __name__ == '__main__': unittest.main()
