"""Synthetic software checks; no food accuracy evidence."""
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

from tools.oven_training.monitor import evaluate_fixed, set_training_mode, bn_snapshot
from tools.oven_training.controlled import recipe_config, validation_readiness
from tools.oven_training.model import ROOT


class TinyClassifier(torch.nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)
        self.linear = torch.nn.Linear(3, classes)

    def forward(self, x):
        logits = self.linear(self.bn(x).mean((2, 3)))
        return logits if self.training else (logits.softmax(1), logits)


class TrainingMonitorTests(unittest.TestCase):
    def test_eval_reports_sample_weighted_loss_confusion_and_absent_class(self):
        class Logits(torch.nn.Module):
            def forward(self, x):
                return x.softmax(1), x
        x = torch.tensor([[4., 0., 0.], [0., 3., 0.], [2., 1., 0.]])
        loader = DataLoader(TensorDataset(x, torch.tensor([0, 1, 1])), batch_size=2,
                            generator=torch.Generator().manual_seed(7))
        result = evaluate_fixed(Logits(), loader, ['a', 'b', 'absent'], 'cpu')
        losses = [math.log(math.exp(4)+2)-4, math.log(math.exp(3)+2)-3,
                  math.log(math.exp(2)+math.exp(1)+1)-1]
        self.assertEqual(result['confusion_matrix'], [[1,0,0], [1,1,0], [0,0,0]])
        self.assertEqual((result['samples'], result['correct']), (3, 2))
        self.assertAlmostEqual(result['loss'], sum(losses)/3, places=6)
        self.assertAlmostEqual(result['per_class']['b']['loss'], sum(losses[1:])/2, places=6)
        self.assertEqual(result['per_class']['b']['top1'], .5)
        self.assertIsNone(result['per_class']['absent']['top1'])
        self.assertIsNone(result['per_class']['absent']['loss'])

    def test_eval_preserves_mixed_modes_buffers_and_rng_even_on_failure(self):
        model = TinyClassifier();model.train();model.bn.eval()
        loader = DataLoader(TensorDataset(torch.ones(3,3,8,8), torch.tensor([0,1,0])), batch_size=2)
        original_bn = bn_snapshot(model)['running_sha256']
        for bad in [False, True]:
            torch.manual_seed(32);random.seed(9);np.random.seed(8)
            rng = torch.get_rng_state().clone();py = random.getstate();npy = np.random.get_state()
            if bad:
                invalid = DataLoader(TensorDataset(torch.ones(1,3,8,8),torch.tensor([9])), batch_size=1)
                with self.assertRaises(ValueError):evaluate_fixed(model, invalid, ['a','b'], 'cpu')
            else:evaluate_fixed(model, loader, ['a','b'], 'cpu')
            self.assertTrue(model.training);self.assertFalse(model.bn.training)
            self.assertEqual(bn_snapshot(model)['running_sha256'], original_bn)
            self.assertTrue(torch.equal(torch.get_rng_state(),rng));self.assertEqual(random.getstate(),py)
            self.assertTrue(np.array_equal(np.random.get_state()[1],npy[1]))

    def test_frozen_stats_survive_repeated_epochs_while_affine_parameters_train(self):
        torch.manual_seed(6);model = TinyClassifier();optimizer = torch.optim.AdamW(model.parameters(),lr=.01)
        before = bn_snapshot(model)['running_sha256'];affine = model.bn.weight.detach().clone()
        for _ in range(2):
            set_training_mode(model, 'frozen_stats')
            for _ in range(2):
                optimizer.zero_grad();loss = torch.nn.functional.cross_entropy(model(torch.ones(4,3,8,8)),torch.tensor([0,1,0,0]))
                loss.backward();optimizer.step()
            self.assertEqual(bn_snapshot(model)['running_sha256'], before)
        self.assertTrue(model.bn.weight.requires_grad)
        self.assertFalse(torch.equal(model.bn.weight,affine))
        set_training_mode(model,'update');model(torch.ones(4,3,8,8))
        self.assertEqual(int(model.bn.num_batches_tracked),1)

    def test_trainer_writes_every_epoch_metrics_tail_and_never_overwrites(self):
        from tools.oven_training.train import fit_exploratory
        import json
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-monitor-',dir=ROOT/'runs') as directory:
            root=Path(directory);photo=root/'SYNTHETIC.png';pretrained=root/'SYNTHETIC-pretrained.pt'
            Image.new('RGB',(320,180),(180,100,80)).save(photo);original=photo.read_bytes();pretrained.write_bytes(b'SYNTHETIC')
            entries=[(photo,'a' if i%2 else 'b') for i in range(33)]
            with patch('tools.oven_training.train.new_model',side_effect=lambda *args:TinyClassifier()), patch('tools.oven_training.train.configure_sdk'):
                fit_exploratory(entries,['a','b'],root/'fit',pretrained,'cpu','a'*64,
                                recipe={'epochs':2,'bn_policy':'frozen_stats'})
                with self.assertRaises(ValueError):fit_exploratory(entries,['a','b'],root/'fit',pretrained,'cpu','a'*64)
            for epoch in [0,1,2]:
                record=json.loads((root/'fit/monitor'/f'epoch_{epoch:03d}.json').read_text())
                self.assertEqual(record['train_eval']['samples'],33)
                self.assertEqual(sum(sum(r) for r in record['train_eval']['confusion_matrix']),33)
                self.assertEqual(record['validation']['status'],'BLOCKED')
            last=json.loads((root/'fit/monitor/epoch_002.json').read_text())
            self.assertEqual(last['batch_trace']['sizes'],[32,1])
            self.assertEqual(last['bn_update']['layers']['bn']['counter_delta'],0)
            self.assertTrue(last['train_eval']['bn_unchanged'])
            self.assertEqual(original,photo.read_bytes())


class HoldoutReadinessTests(unittest.TestCase):
    def row(self, split, key='a', marker='1'):
        return dict(new_split=split,class_key=key,class_id=0,sha256=marker*64,pixel_sha256=(marker.upper() if marker.isalpha() else chr(97+int(marker)))*64,
                    group_id='group-'+marker,physical_sample_id='physical-'+marker,session_id='session-'+marker,
                    physical_identity_verified=True,session_identity_verified=True,identity_reviewed_by='SYNTHETIC-reviewer',
                    identity_reviewed_at='2026-11-11',trusted_label=True,label_reviewed_by='SYNTHETIC-reviewer',label_reviewed_at='2026-11-11')

    def test_holdout_rejects_missing_verified_identity_hash_group_and_session_leaks(self):
        train=self.row('train',marker='1');val=self.row('val',marker='2')
        self.assertEqual(validation_readiness([train],[val],['a'])['status'],'READY_FOR_EXPLORATORY_MONITORING')
        for change in [dict(physical_identity_verified=False),dict(session_identity_verified=False),dict(trusted_label=False),
                       dict(physical_sample_id=train['physical_sample_id']),dict(session_id=train['session_id']),
                       dict(sha256=train['sha256']),dict(pixel_sha256=train['pixel_sha256']),dict(group_id=train['group_id']),
                       dict(new_split='test'),dict(class_id=9),dict(identity_reviewed_by='')]:
            self.assertEqual(validation_readiness([train],[{**val,**change}],['a'])['status'],'BLOCKED',change)
        self.assertEqual(validation_readiness([{**train,'physical_identity_verified':False}],[val],['a'])['status'],'BLOCKED')
        self.assertEqual(validation_readiness([train],[],['a'])['status'],'BLOCKED')

    def test_paired_recipes_differ_only_in_bn_policy_and_invalid_options_fail(self):
        import json
        root=ROOT/'tools/oven_training/recipes'
        a=recipe_config(json.loads((root/'bn-update.json').read_text()))
        b=recipe_config(json.loads((root/'bn-frozen-stats.json').read_text()))
        self.assertEqual({k:v for k,v in a.items() if k!='bn_policy'}, {k:v for k,v in b.items() if k!='bn_policy'})
        self.assertEqual((a['bn_policy'],b['bn_policy']),('update','frozen_stats'))
        for change in [dict(epochs=0),dict(epochs=True),dict(bn_policy='train_at_inference'),dict(seed=99),dict(batch_size=1),dict(extra=True)]:
            with self.assertRaises(ValueError):recipe_config(change)

    def test_validation_duplicate_images_cannot_fake_class_coverage(self):
        train=self.row('train',marker='1');val=self.row('val',marker='2')
        self.assertEqual(validation_readiness([train],[val,val],['a'])['status'],'BLOCKED')
        conflicting={**val,'class_key':'b','class_id':1}
        self.assertEqual(validation_readiness([train],[val,conflicting],['a','b'])['status'],'BLOCKED')
        reencoded={**conflicting,'sha256':'3'*64}
        self.assertEqual(validation_readiness([train],[val,reencoded],['a','b'])['status'],'BLOCKED')
        independent={**reencoded,'pixel_sha256':'d'*64,'physical_sample_id':'physical-3','session_id':'session-3','group_id':'group-3'}
        self.assertEqual(validation_readiness([train],[val,independent],['a','b'])['status'],'READY_FOR_EXPLORATORY_MONITORING')

    def test_intake_csv_parses_explicit_signoffs_and_empty_template_is_not_ready(self):
        from tools.oven_training.controlled import read_rows
        import csv
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'SYNTHETIC.csv';row=self.row('val',marker='2')
            with path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            parsed=read_rows(path)
            self.assertIs(parsed[0]['trusted_label'],True)
            self.assertIsInstance(parsed[0]['class_id'],int)
            self.assertEqual(validation_readiness([self.row('train',marker='1')],parsed,['a'])['status'],'READY_FOR_EXPLORATORY_MONITORING')
            path.write_text('class_id,trusted_label\n0,yes\n')
            with self.assertRaises(ValueError):read_rows(path)
        empty=read_rows(ROOT/'tools/oven_training/recipes/physical-validation-intake.csv')
        self.assertEqual(validation_readiness([self.row('train')],empty,['a'])['status'],'BLOCKED')
