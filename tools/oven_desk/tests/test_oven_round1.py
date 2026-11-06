"""Synthetic contract/GUI-state tests only; these are not food accuracy evidence."""
import csv
import hashlib
import io
import json
from pathlib import Path
import queue
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
        'french_fries', 'potato', 'salmon', 'toast']


def synthetic_contract(model_bytes):
    return dict(format='het-oven-classifier/v1', class_keys=KEYS,
        class_list_sha256='42bd80f04446cd87f8c719c78034bf274bd447fe7a60205713f118cd0b584b52',
        classes=[dict(id=i, key=k) for i, k in enumerate(KEYS)],
        preprocess=dict(size=224, color='RGB', resize='bilinear-contain-center-pad',
                        pad=114, scale='1/255', layout='NCHW'),
        source_checkpoint_sha256='a'*64,
        models={'onnx': {'sha256': hashlib.sha256(model_bytes).hexdigest()}},
        input_shape=[1, 3, 224, 224], output_shape=[1, 9],
        output_kind='classification-probabilities', independent_validation=False)


class SyntheticSession:
    """Isolate runtime boundary validation; never load this as a demo model."""
    shape = [1, 3, 224, 224]
    output_shape = [1, 9]
    output = np.array([[.01, .02, .03, .04, .05, .06, .07, .08, .64]], dtype=np.float32)
    metadata = {}

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


class OvenRound1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='SYNTHETIC-oven-contract-')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.model = self.folder/'synthetic.onnx'
        self.model.write_bytes(b'SYNTHETIC runtime boundary fixture, not an ONNX model')
        self.labels = self.folder/'classes.json'
        self.contract = synthetic_contract(self.model.read_bytes())
        self.write_contract()
        self.runtime = patch.dict(sys.modules, {'onnxruntime': types.SimpleNamespace(InferenceSession=SyntheticSession)})
        self.runtime.start()
        self.addCleanup(self.runtime.stop)

    def write_contract(self):
        self.labels.write_text(json.dumps(self.contract), encoding='utf-8')

    def engine(self, **kwargs):
        return ModelEngine(self.model, self.labels, 'oven_round1', **kwargs)

    def test_contain_keeps_full_landscape_and_portrait_without_crop(self):
        for h, w in [(53, 191), (191, 53), (223, 224), (3, 1000)]:
            with self.subTest(size=(h,w)):
                rgb = np.random.default_rng(42).integers(0, 256, (h,w,3), dtype=np.uint8)
                contained = ImageOps.contain(Image.fromarray(rgb), (224,224), Image.Resampling.BILINEAR)
                canvas = Image.new('RGB', (224,224), (114,114,114))
                canvas.paste(contained, ((224-contained.width)//2, (224-contained.height)//2))
                expected = np.asarray(canvas, dtype=np.float32).transpose(2,0,1)[None]/255
                actual, context = prepare_input(rgb[:,:,::-1], 'oven_round1', 224, 224)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.float32)
                self.assertTrue(actual.flags.c_contiguous)
                self.assertIsNone(context)
        with self.assertRaises(ModelContractError):
            prepare_input(np.zeros((2,3,3), np.uint8), 'oven_round1', 320, 320)
        with self.assertRaises(ModelContractError):
            prepare_input(np.zeros((1,1000,3), np.uint8), 'oven_round1', 224, 224)

    def test_file_decoder_applies_exif_orientation_for_round1(self):
        path = self.folder/'旋转照片.jpg'
        im = Image.fromarray(np.arange(60, dtype=np.uint8).reshape(4,5,3))
        exif = im.getexif()
        exif[274] = 6
        im.save(path, exif=exif)
        with Image.open(path) as source:
            expected = np.asarray(ImageOps.exif_transpose(source).convert('RGB'))[:,:,::-1]
        actual = read_image_unicode(path, mode='oven_round1')
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (5,4,3))

    def test_ready_requires_real_forward_and_strict_probabilities(self):
        engine = self.engine()
        self.assertEqual(len(engine.session.inputs_seen), 1)
        np.testing.assert_array_equal(engine.session.inputs_seen[0], np.zeros((1,3,224,224), np.float32))
        result = engine.predict(np.zeros((30,90,3), np.uint8))
        self.assertEqual(result['top3'][0]['label'], 'toast')
        self.assertFalse(result['independent_validation'])
        self.assertEqual(len(result['probabilities']), 9)
        for value in [np.ones((1,9), np.float32), np.zeros(9, np.float32),
                      np.full((1,9), np.nan, np.float32), np.ones((1,9), np.float64)/9,
                      np.array([[-.01, .01, 0, 0, 0, 0, 0, 0, 1]], np.float32)]:
            with self.subTest(output=value), patch.object(SyntheticSession, 'output', value):
                with self.assertRaises(ModelContractError): self.engine()

    def test_bad_contract_rejected_before_runtime(self):
        changes = [dict(format='other'), dict(class_keys=KEYS[::-1]), dict(class_list_sha256='0'*64),
                   dict(classes=[dict(id=i, key=k) for i,k in enumerate(KEYS[::-1])]),
                   dict(preprocess={}), dict(source_checkpoint_sha256='missing'),
                   dict(models={}), dict(input_shape=[2,3,224,224]), dict(output_shape=[9]),
                   dict(output_kind='logits'), dict(independent_validation=True)]
        for change in changes:
            with self.subTest(change=change):
                self.labels.write_text(json.dumps({**self.contract, **change}), encoding='utf-8')
                with self.assertRaises(ModelContractError): self.engine()
        for text in ['[]', '{}', '{broken', 'null']:
            self.labels.write_text(text, encoding='utf-8')
            with self.assertRaises(ModelContractError): self.engine()

    def test_mismatched_checkpoint_and_input_output_rejected(self):
        self.model.write_bytes(b'changed model')
        with self.assertRaises(ModelContractError): self.engine()
        self.contract = synthetic_contract(self.model.read_bytes())
        self.write_contract()
        with self.assertRaises(ModelContractError): self.engine(input_size=320)
        for shape in [[2,3,224,224], [1,3,'h','w'], [1,3,320,320]]:
            with patch.object(SyntheticSession, 'shape', shape):
                with self.assertRaises(ModelContractError): self.engine()
        with patch.object(SyntheticSession, 'output_shape', [1,8]):
            with self.assertRaises(ModelContractError): self.engine()
        mnn_path = self.folder/'unapproved.mnn'
        mnn_path.write_bytes(b'SYNTHETIC unapproved runtime format')
        with self.assertRaises(ModelContractError):
            ModelEngine(mnn_path, self.labels, 'oven_round1')

    def test_generic_mode_refuses_round1_contract_and_marked_model(self):
        for mode in ['yolo_classify', 'efficientnet_imagenet', 'yolo_detect']:
            with self.subTest(mode=mode), self.assertRaises(ModelContractError):
                ModelEngine(self.model, self.labels, mode)
        plain = self.folder/'plain.txt'
        plain.write_text('\n'.join(KEYS), encoding='utf-8')
        with self.assertRaises(ModelContractError):
            ModelEngine(self.model, plain, 'yolo_classify')
        self.labels.unlink()
        plain.write_text('other\nlabels', encoding='utf-8')
        with patch.object(SyntheticSession, 'metadata', {'het_contract_format':'het-oven-classifier/v1'}):
            with self.assertRaises(ModelContractError): ModelEngine(self.model, plain, 'yolo_classify')

    def test_safe_selfcheck_writes_predictions_without_accuracy(self):
        image = self.folder/'SYNTHETIC.png'
        Image.new('RGB', (30,80)).save(image)
        report = self.folder/'selfcheck.json'
        with patch.object(sys, 'stdout', io.StringIO()):
            self.assertEqual(gui.model_selfcheck(self.model, self.labels, image, report), 0)
        obj = json.loads(report.read_text('utf-8'))
        self.assertEqual(obj['status'], 'PASS')
        self.assertEqual(obj['result']['top3'][0]['label'], 'toast')
        self.assertNotIn('accuracy', json.dumps(obj).lower())
        self.model.write_bytes(b'corrupt')
        with patch.object(sys, 'stdout', io.StringIO()):
            self.assertEqual(gui.model_selfcheck(self.model, self.labels, image, report), 1)
        self.assertEqual(json.loads(report.read_text('utf-8'))['status'], 'FAIL')

    def test_selfcheck_never_overwrites_selected_input(self):
        image = self.folder/'SYNTHETIC.png'
        Image.new('RGB', (10,20)).save(image)
        before = image.read_bytes()
        with patch.object(sys, 'stderr', io.StringIO()):
            self.assertEqual(gui.model_selfcheck(self.model, self.labels, image, image), 1)
        self.assertEqual(image.read_bytes(), before)


class Value:
    def __init__(self, value=''): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value


class DeferredThread:
    tasks = []
    def __init__(self, target, **kwargs): self.target = target
    def start(self): self.tasks.append(self.target)


class GuiModelStateTests(unittest.TestCase):
    def setUp(self):
        self.app = gui.DesktopApp.__new__(gui.DesktopApp)
        self.app.model = types.SimpleNamespace(predict=lambda frame: {'task':'classification', 'top3':[{'label':'toast','score':.8}]})
        self.app.model_loading = False
        self.app.inference_running = False
        self.app.model_file = Value('model.onnx')
        self.app.label_file = Value('classes.json')
        self.app.model_task = Value('首轮烤箱分类（实验·9类）')
        self.app.network_size = Value('224')
        self.app.model_state = Value('已加载')
        self.app.events = queue.Queue()
        self.app._log = lambda text: None
        DeferredThread.tasks = []
        thread = patch.object(gui.threading, 'Thread', DeferredThread)
        thread.start(); self.addCleanup(thread.stop)
        for name in ['showwarning', 'showinfo']:
            dialog = patch.object(gui.messagebox, name)
            dialog.start(); self.addCleanup(dialog.stop)

    def test_reload_clears_ready_model_and_busy_reload_is_refused(self):
        old = self.app.model
        self.app.inference_running = True
        self.app.load_model()
        self.assertIs(self.app.model, old)
        self.assertFalse(DeferredThread.tasks)
        self.app.inference_running = False
        self.app.load_model()
        self.assertIsNone(self.app.model)
        self.assertTrue(self.app.model_loading)
        self.assertEqual(len(DeferredThread.tasks), 1)

    def test_invalid_reload_cannot_keep_previous_model_ready(self):
        self.app.network_size.set('invalid')
        with patch.object(gui.messagebox, 'showwarning'):
            self.app.load_model()
        self.assertIsNone(self.app.model)

    def test_failed_model_load_reports_not_ready_and_blocks_prediction(self):
        self.app.load_model()
        with patch.object(gui, 'ModelEngine', side_effect=ModelContractError('SYNTHETIC invalid model')):
            DeferredThread.tasks.pop()()
        self.assertEqual(self.app.events.get_nowait()[0], 'model_error')
        self.assertIsNone(self.app.model)
        self.app._start_predict(np.zeros((2,3,3), np.uint8))
        self.assertFalse(DeferredThread.tasks)

    def test_current_frame_worker_snapshots_model(self):
        self.app._start_predict(np.zeros((2,3,3), np.uint8))
        self.app.model = None
        DeferredThread.tasks.pop()()
        kind, payload = self.app.events.get_nowait()
        self.assertEqual(kind, 'predict')
        self.assertEqual(payload[1]['top3'][0]['label'], 'toast')

    def test_gallery_snapshots_model_and_never_reports_unlabelled_accuracy(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-oven-gallery-') as td:
            folder = Path(td)
            Image.new('RGB', (10,20)).save(folder/'SYNTHETIC.png')
            self.app.output = Value(str(folder/'output'))
            self.app.model.predict_file = lambda path: self.app_result()
            with patch.object(gui.filedialog, 'askdirectory', return_value=str(folder)):
                self.app.test_gallery()
            self.app.model = None
            DeferredThread.tasks.pop()()
            kind, payload = self.app.events.get_nowait()
            self.assertEqual(kind, 'gallery_done')
            self.assertEqual(payload[:2], (1,1))
            with Path(payload[2]).open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['predicted_label'], 'toast')
            self.assertNotIn('accuracy', json.dumps(rows).lower())

    @staticmethod
    def app_result():
        return {'task':'classification', 'top3':[{'label':'toast','score':.8}]}


if __name__ == '__main__': unittest.main()
