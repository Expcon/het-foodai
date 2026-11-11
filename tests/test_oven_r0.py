"""SYNTHETIC recovery and counterfactual software tests, never food accuracy."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from tools.oven_training.model import ROOT, digest
from tools.oven_training.monitor import evaluate_fixed, model_state_sha256, tensor_sha256


class SyntheticImages(Dataset):
    def __len__(self): return 11
    def __getitem__(self, i):
        value = torch.rand(3, 4, 4) + random.random() + float(np.random.random()) + i / 10
        return value, i % 3, i


class SyntheticClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)
        self.drop = torch.nn.Dropout(.25)
        self.linear = torch.nn.Linear(3, 3)
    def forward(self, x):
        logits = self.linear(self.drop(self.bn(x).mean((2, 3))))
        return logits if self.training else (logits.softmax(1), logits)


class R0Tests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('tools.oven_training.observation'), 'R0 observation module missing')
        self.r0 = importlib.import_module('tools.oven_training.observation')
        self.temp = tempfile.TemporaryDirectory(prefix='SYNTHETIC-r0-', dir=ROOT / 'runs')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binding = {'scope': 'SYNTHETIC SOFTWARE ONLY', 'classes': ['a', 'b', 'c']}
        torch.manual_seed(42); random.seed(42); np.random.seed(42)
        torch.set_num_threads(4)

    def loader(self):
        return DataLoader(SyntheticImages(), batch_size=4, shuffle=True, num_workers=0,
                          generator=torch.Generator().manual_seed(42))

    def step(self, model, optimizer, batch, device='cpu'):
        x, y, _ = batch
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device))
        loss.backward(); optimizer.step()
        return float(loss.detach()), model_state_sha256(model)

    def save(self, model, optimizer, loader, cursor, pending=None, phase='post_step', step=1):
        path = self.root / ('pending.pt' if pending is not None else 'state.pt')
        sha = self.r0.save_snapshot(path, model, optimizer, epoch=1, global_step=step,
                                   phase=phase, binding=self.binding, loaders={'train': loader},
                                   cursor=cursor.state(loader), pending=pending)
        return path, sha

    def test_rng_roundtrip_preserves_python_numpy_torch_streams(self):
        state = self.r0.capture_rng()
        want = (random.random(), np.random.random(), torch.rand(9))
        self.r0.restore_rng(state)
        got = (random.random(), np.random.random(), torch.rand(9))
        self.assertEqual(want[:2], got[:2]); self.assertTrue(torch.equal(want[2], got[2]))

    def test_safe_snapshot_is_readable_without_pickle_and_refuses_overwrite_or_bad_sha(self):
        model = SyntheticClassifier(); optimizer = torch.optim.AdamW(model.parameters())
        loader = self.loader(); cursor = self.r0.LoaderCursor(loader)
        rng = torch.get_rng_state().clone()
        path, sha = self.save(model, optimizer, loader, cursor)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        payload = torch.load(path, weights_only=True)
        self.assertEqual(payload['global_step'], 1)
        self.assertEqual(self.r0.load_snapshot(path, sha)['model_state_sha256'], model_state_sha256(model))
        with self.assertRaises(FileExistsError): self.save(model, optimizer, loader, cursor)
        self.assertEqual(digest(path), sha)
        with self.assertRaises(ValueError): self.r0.load_snapshot(path, '0' * 64)

    def assert_recovery(self, device='cpu', pending_tail=False):
        model = SyntheticClassifier().to(device).train(); opt = torch.optim.AdamW(model.parameters(), lr=.01)
        loader = self.loader(); cursor = self.r0.LoaderCursor(loader); iterator = iter(loader)
        first = next(iterator); cursor.record(first[2]); self.step(model, opt, first, device)
        if pending_tail:
            middle = next(iterator); cursor.record(middle[2]); self.step(model, opt, middle, device)
            batch = next(iterator); cursor.record(batch[2])
            opt.zero_grad(set_to_none=True)
            path, sha = self.save(model, opt, loader, cursor, pending=batch, phase='pre_forward', step=2)
            expected = batch
        else:
            path, sha = self.save(model, opt, loader, cursor)
            expected = next(iterator)
        expected_rng = self.r0.capture_rng()
        wanted = self.step(model, opt, expected, device)
        expected_state = {k: v.clone() for k, v in model.state_dict().items()}
        expected_opt = copy.deepcopy(opt.state_dict())
        payload = self.r0.load_snapshot(path, sha)
        restored = SyntheticClassifier().to(device); other_opt = torch.optim.AdamW(restored.parameters(), lr=.01)
        other_loader = self.loader()
        self.r0.restore_training(payload, restored, other_opt, {'train': other_loader}, self.binding)
        actual = next(self.r0.resume_batches(other_loader, payload))
        self.assertTrue(torch.equal(expected[2], actual[2]))
        self.assertEqual(tensor_sha256(expected[0]), tensor_sha256(actual[0]))
        actual_rng = self.r0.capture_rng()
        self.assertEqual(expected_rng['python'], actual_rng['python'])
        self.assertTrue(torch.equal(expected_rng['torch'], actual_rng['torch']))
        got = self.step(restored, other_opt, actual, device)
        self.assertEqual(wanted, got)
        for k, v in expected_state.items(): self.assertTrue(torch.equal(v, restored.state_dict()[k]), k)
        for i, states in expected_opt['state'].items():
            for k, v in states.items():
                self.assertTrue(torch.equal(v, other_opt.state_dict()['state'][i][k]), (i, k))
        # Loaded payload must not alias optimizer state mutated by the resumed step.
        self.assertEqual(self.r0.load_snapshot(path, sha)['global_step'], payload['global_step'])

    def test_mid_epoch_next_batch_loss_and_adamw_state_are_exact(self): self.assert_recovery()

    def test_pre_forward_small_tail_replays_augmented_tensor_without_resampling(self):
        self.assert_recovery(pending_tail=True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable for synthetic recovery')
    def test_cuda_pending_tail_loss_and_updated_parameters_are_exact(self):
        self.assert_recovery('cuda:0', pending_tail=True)

    def test_epoch_boundary_restores_next_shuffle_and_all_rng(self):
        model = SyntheticClassifier(); opt = torch.optim.AdamW(model.parameters())
        loader = self.loader(); cursor = self.r0.LoaderCursor(loader)
        for batch in loader: cursor.record(batch[2]); self.step(model, opt, batch)
        path = self.root / 'epoch.pt'
        sha = self.r0.save_snapshot(path, model, opt, epoch=1, global_step=3, phase='epoch_end',
            binding=self.binding, loaders={'train': loader}, cursor=None)
        expected = next(iter(loader)); wanted = self.step(model, opt, expected)
        payload = self.r0.load_snapshot(path, sha)
        m = SyntheticClassifier(); o = torch.optim.AdamW(m.parameters()); dl = self.loader()
        self.r0.restore_training(payload, m, o, {'train': dl}, self.binding)
        batch = next(self.r0.resume_batches(dl, payload))
        self.assertTrue(torch.equal(batch[2], expected[2])); self.assertTrue(torch.equal(batch[0], expected[0]))
        self.assertEqual(self.step(m, o, batch), wanted)

    def test_binding_mismatch_is_rejected_before_mutating_model(self):
        m = SyntheticClassifier(); opt = torch.optim.AdamW(m.parameters()); dl = self.loader()
        path, sha = self.save(m, opt, dl, self.r0.LoaderCursor(dl))
        payload = self.r0.load_snapshot(path, sha); before = model_state_sha256(m)
        with self.assertRaises(ValueError): self.r0.restore_training(payload, m, opt, {'train': dl}, {'classes': ['c','b','a']})
        self.assertEqual(model_state_sha256(m), before)

    def test_fixed_per_image_evidence_preserves_metrics_weights_and_rng(self):
        model = SyntheticClassifier().train()
        x = torch.arange(7 * 3 * 4 * 4, dtype=torch.float32).reshape(7,3,4,4) / 400
        labels = torch.tensor([0,1,2,0,1,2,0])
        loader = DataLoader(TensorDataset(x, labels), batch_size=4, generator=torch.Generator().manual_seed(6))
        before = model_state_sha256(model); rng = torch.get_rng_state().clone()
        baseline = evaluate_fixed(model, loader, ['a','b','c'], 'cpu')
        evidence = []
        actual = evaluate_fixed(model, loader, ['a','b','c'], 'cpu', per_image=evidence)
        self.assertEqual(baseline, actual); self.assertEqual(len(evidence), 7)
        self.assertEqual(model_state_sha256(model), before); self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual([r['target_id'] for r in evidence], labels.tolist())
        for i, row in enumerate(evidence):
            self.assertEqual(row['input_sha256'], tensor_sha256(x[i]))
            self.assertEqual(len(row['logits']), 3)
            self.assertEqual(row['top1_id'], int(torch.tensor(row['logits']).argmax()))
        self.assertAlmostEqual(sum(r['loss'] for r in evidence) / 7, actual['loss'], places=12)

    def test_mixing_changes_only_bn_buffers_or_only_learnable_parameters(self):
        a = SyntheticClassifier(); b = copy.deepcopy(a)
        with torch.no_grad():
            b.linear.weight.add_(1); b.bn.weight.add_(2); b.bn.running_mean.add_(3); b.bn.num_batches_tracked.add_(4)
        mixed = self.r0.mix_states(a, a.state_dict(), b.state_dict())
        self.assertTrue(torch.equal(mixed['linear.weight'], a.linear.weight))
        self.assertTrue(torch.equal(mixed['bn.weight'], a.bn.weight))
        self.assertTrue(torch.equal(mixed['bn.running_mean'], b.bn.running_mean))
        self.assertEqual(int(mixed['bn.num_batches_tracked']), 4)
        bad = dict(b.state_dict()); bad['unknown'] = torch.ones(1)
        with self.assertRaises(ValueError): self.r0.mix_states(a, a.state_dict(), bad)

    def test_observed_fit_is_transparent_and_writes_epoch_and_tail_checkpoints(self):
        from unittest.mock import patch
        from PIL import Image
        from tools.oven_training.train import fit_exploratory
        photo=self.root/'SYNTHETIC.png';Image.new('RGB',(300,180),(130,80,40)).save(photo)
        pretrained=self.root/'SYNTHETIC-init';pretrained.write_bytes(b'SYNTHETIC')
        keys=['a','b','c'];entries=[(photo,keys[i%3]) for i in range(33)]
        with patch('tools.oven_training.train.new_model',side_effect=lambda *args:SyntheticClassifier()), patch('tools.oven_training.train.configure_sdk'):
            fit_exploratory(entries,keys,self.root/'plain',pretrained,'cpu','a'*64,recipe={'epochs':2})
            fit_exploratory(entries,keys,self.root/'observed',pretrained,'cpu','a'*64,recipe={'epochs':2},
                            observation={'reference':self.root/'plain','tail_epochs':(1,2)})
        records=[json.loads(s) for s in (self.root/'observed/observations/checkpoints.jsonl').read_text().splitlines()]
        self.assertEqual(len(records),7)
        for r in records:
            payload=self.r0.load_snapshot(self.root/'observed/observations'/r['file'],r['sha256'])
            self.assertEqual(payload['phase'],r['phase'])
        gate=json.loads((self.root/'observed/observations/gate.json').read_text())
        self.assertEqual(gate['status'],'PASS');self.assertEqual(gate['global_steps'],4)
        for epoch in range(3):
            rows=[json.loads(s) for s in (self.root/f'observed/observations/eval/epoch_{epoch:03d}.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),33)
            self.assertTrue(all(r['source_sha256']==digest(photo) for r in rows))


if __name__ == '__main__': unittest.main()
