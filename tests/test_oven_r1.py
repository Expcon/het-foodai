"""SYNTHETIC tests of one-factor LR control, not food performance."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from tools.oven_training.controlled import recipe_config
from tools.oven_training.model import ROOT, digest
from tools.oven_training.observation import load_snapshot


class SyntheticClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__();self.bn=torch.nn.BatchNorm2d(3);self.drop=torch.nn.Dropout(.25);self.linear=torch.nn.Linear(3,3)
    def forward(self,x):
        logits=self.linear(self.drop(self.bn(x).mean((2,3))))
        return logits if self.training else (logits.softmax(1),logits)


class R1Tests(unittest.TestCase):
    def test_lr_requires_explicit_control_and_does_not_unlock_old_recipes(self):
        with self.assertRaises(ValueError):recipe_config({'lr':.0003})
        self.assertEqual(recipe_config({'lr':.0003},lr_control=True)['lr'],.0003)
        self.assertEqual(recipe_config()['lr'],.001)

    def test_lr_control_rejects_unregistered_rates_types_and_other_changes(self):
        for value in [True,0,-.001,.001,.0002,float('nan'),float('inf'),'0.0003']:
            with self.assertRaises(ValueError):recipe_config({'lr':value},lr_control=True)
        for change in [dict(seed=7),dict(batch_size=16),dict(weight_decay=.0001),dict(drop_last=True),
                       dict(bn_policy='frozen_stats'),dict(amp=True),dict(workers=1),dict(epochs=0),dict(extra=True)]:
            with self.assertRaises(ValueError):recipe_config({'lr':.0003,**change},lr_control=True)

    def test_new_r1_recipe_changes_only_lr_and_old_recipe_bytes_are_pinned(self):
        recipes=ROOT/'tools/oven_training/recipes'
        self.assertEqual(digest(recipes/'bn-update.json'),'f6d98f01522e5338f39d9178852156bcdaee36a3945e1e14723372ecf2ae4195')
        self.assertEqual(digest(recipes/'bn-frozen-stats.json'),'d31a946f03d7a61026d11b6b7e110cdb608644ac6e224cdec8a01efb54f229aa')
        a=json.loads((recipes/'bn-update.json').read_text())
        b=json.loads((recipes/'r1-lr-0003.json').read_text())
        self.assertEqual({k for k in a if a[k]!=b[k]},{'lr'})
        self.assertEqual(recipe_config(b,lr_control=True)['lr'],.0003)

    def test_fit_applies_real_lr_preserves_inputs_and_rejects_unqualified_reference_or_new_sources(self):
        from tools.oven_training.train import fit_exploratory
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-r1-',dir=ROOT/'runs') as directory:
            root=Path(directory);photo=root/'SYNTHETIC.png';Image.new('RGB',(300,180),(130,80,40)).save(photo)
            pretrained=root/'SYNTHETIC-init';pretrained.write_bytes(b'SYNTHETIC')
            keys=['a','b','c'];entries=[(photo,keys[i%3]) for i in range(33)]
            common=dict(entries=entries,keys=keys,pretrained=pretrained,device='cpu',rows_sha256='a'*64,
                        provenance={'scope':'SYNTHETIC SOFTWARE ONLY'})
            with patch('tools.oven_training.train.new_model',side_effect=lambda *args:SyntheticClassifier()), patch('tools.oven_training.train.configure_sdk'):
                fit_exploratory(**common,out=root/'A',recipe={'epochs':2})
                fit_exploratory(**common,out=root/'R0',recipe={'epochs':2},observation={'reference':root/'A','tail_epochs':(1,2)})
                fit_exploratory(**common,out=root/'R1',recipe={'epochs':2,'lr':.0003},
                    observation={'reference':root/'R0','comparison':'lr_only','tail_epochs':(1,2)})
                with self.assertRaises(ValueError):
                    fit_exploratory(**common,out=root/'bad-reference',recipe={'epochs':2,'lr':.0003},
                        observation={'reference':root/'A','comparison':'lr_only'})
                alternative=root/'SYNTHETIC-other.png';Image.new('RGB',(300,180),(90,110,20)).save(alternative)
                with self.assertRaises(ValueError):
                    fit_exploratory(**{**common,'entries':[(alternative,key) for _,key in entries]},out=root/'bad-source',
                        recipe={'epochs':2,'lr':.0003},observation={'reference':root/'R0','comparison':'lr_only'})
            obs=root/'R1/observations';gate=json.loads((obs/'gate.json').read_text())
            self.assertEqual(gate['comparison_mode'],'lr_only')
            self.assertTrue(gate['all_reference_batch_traces_equal'])
            self.assertFalse(gate['all_reference_metrics_and_batches_equal'])
            self.assertEqual(gate['only_config_difference'],'lr')
            records=[json.loads(s) for s in (obs/'checkpoints.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),7)
            initial=next(r for r in records if r['file'].endswith('epoch_000.pt'))
            payload=load_snapshot(obs/initial['file'],initial['sha256'])
            self.assertEqual(payload['optimizer_state']['param_groups'][0]['lr'],.0003)
            for epoch in [1,2]:
                a=json.loads((root/f'R0/monitor/epoch_{epoch:03d}.json').read_text())
                b=json.loads((root/f'R1/monitor/epoch_{epoch:03d}.json').read_text())
                self.assertEqual(a['batch_trace'],b['batch_trace'])
            a=torch.load(root/'R0/last.pt',weights_only=True)['model_state']
            b=torch.load(root/'R1/last.pt',weights_only=True)['model_state']
            self.assertTrue(any(not torch.equal(a[k],b[k]) for k in a))
            steps=[json.loads(s) for s in (obs/'batch_diagnostics.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['optimizer_lr']==[.0003] for r in steps))


if __name__=='__main__':unittest.main()
