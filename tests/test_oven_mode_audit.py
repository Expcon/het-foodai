"""Synthetic-only controls for frozen-state mode diagnosis."""
from pathlib import Path
import tempfile
import unittest
import torch
from PIL import Image

from tools.oven_training.mode_audit import configure_mode, state_digest, probabilities, original_dataset, epoch_batches
from tools.oven_training.model import transform
from tools.oven_training.round2 import V2_KEYS


class OvenModeAuditTests(unittest.TestCase):
    def test_batch_normalization_uses_batch_statistics_without_state_writes(self):
        model = torch.nn.BatchNorm2d(3)
        before = state_digest(model)
        x = torch.arange(4*3*8*8, dtype=torch.float32).reshape(4,3,8,8)/100
        configure_mode(model, 'train_batch_bn')
        with torch.inference_mode():
            actual = model(x)
            expected = torch.nn.functional.batch_norm(x, None, None, model.weight, model.bias,
                                                       training=True, eps=model.eps)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(before, state_digest(model))
        self.assertEqual(int(model.num_batches_tracked), 0)

    def test_training_with_running_bn_matches_eval(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.AdaptiveAvgPool2d(1),
                                    torch.nn.Flatten(), torch.nn.Linear(3,12))
        x = torch.ones(4,3,8,8)
        before = state_digest(model)
        with torch.inference_mode():
            expected = configure_mode(model, 'eval')(x)
            actual = configure_mode(model, 'train_running_bn')(x)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(before, state_digest(model))

    def test_logits_and_eval_tuple_have_same_probability_semantics(self):
        logits = torch.arange(24, dtype=torch.float32).reshape(2,12)
        p, _ = probabilities(logits)
        actual, raw = probabilities((p,logits))
        self.assertTrue(torch.equal(actual, p))
        self.assertTrue(torch.equal(raw, logits))
        with self.assertRaises(ValueError): probabilities((p.roll(1,1),logits))

    def test_real_dataset_class_targets_and_reconstructed_sampler(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'SYNTHETIC.png'
            Image.new('RGB',(320,180),(100,120,130)).save(path)
            before = path.read_bytes()
            dataset, _ = original_dataset([(path,'sausage'),(path,'potato'),(path,'asparagus')], V2_KEYS)
            dataset.augment = transform(False)
            self.assertEqual([dataset[i][1] for i in range(3)],[10,6,0])
            self.assertEqual(tuple(dataset[0][0].shape),(3,224,224))
            self.assertEqual(before,path.read_bytes())
        batches = epoch_batches(1996)
        self.assertEqual(len(batches),63)
        self.assertEqual(len(batches[-1]),12)
        self.assertEqual(sorted(i for batch in batches for i in batch),list(range(1996)))
        self.assertEqual(batches,epoch_batches(1996))
