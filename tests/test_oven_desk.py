"""Run the standalone Desk tests and compare its transform to training."""
import importlib.util
from pathlib import Path
import sys
import unittest

DESK = Path(__file__).resolve().parents[1]/'tools'/'oven_desk'
sys.path.insert(0, str(DESK))


def load_tests(loader, tests, pattern):
    for path in sorted((DESK/'tests').glob('test_*.py')):
        spec = importlib.util.spec_from_file_location('oven_desk_'+path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tests.addTests(loader.loadTestsFromModule(module))
    return tests


class TrainingTransformParityTests(unittest.TestCase):
    def test_desk_tensor_matches_training_transform_exactly(self):
        import numpy as np
        from PIL import Image
        from inference_engine import prepare_input
        from tools.oven_training.model import transform
        for h,w in [(67,192), (193,68), (224,224)]:
            rgb = np.random.default_rng(7).integers(0,256,(h,w,3),dtype=np.uint8)
            expected = transform(False)(Image.fromarray(rgb)).numpy()[None]
            actual, _ = prepare_input(rgb[:,:,::-1], 'oven_round1', 224, 224)
            np.testing.assert_array_equal(actual, expected)
