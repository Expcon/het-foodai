"""SYNTHETIC paired prediction comparisons; no real model performance claim."""
import importlib
import importlib.util
import unittest
import json
from pathlib import Path
import tempfile
from tools.oven_training.model import ROOT


class CrossTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('tools.oven_training.cross_states'), 'Cross-state diagnostics missing')
        self.cross=importlib.import_module('tools.oven_training.cross_states')

    def rows(self, predictions):
        targets=[0,0,1,1]
        return [dict(sample_index=i,target_id=targets[i],top1_id=p,input_sha256=str(i)*64,loss=.1+i)
                for i,p in enumerate(predictions)]

    def test_paired_transitions_distinguish_rescues_and_new_errors(self):
        result=self.cross.transition_summary(self.rows([0,1,0,1]),self.rows([1,0,1,1]),['a','b'])
        self.assertEqual(result['before_correct'],2);self.assertEqual(result['after_correct'],3)
        self.assertEqual(result['rescued'],2);self.assertEqual(result['introduced'],1)
        self.assertEqual(result['changed_predictions'],3)
        self.assertEqual(result['per_class']['a']['rescued'],1)
        self.assertEqual(result['per_class']['a']['introduced'],1)
        self.assertEqual(result['per_class']['b']['rescued'],1)

    def test_pairing_rejects_wrong_image_order_inputs_targets_or_class_index(self):
        left=self.rows([0,1,0,1]);right=self.rows([0,0,1,1])
        for bad in [right[::-1],[{**right[0],'input_sha256':'bad'},*right[1:]],
                    [{**right[0],'target_id':1},*right[1:]],right[:-1],
                    [{**right[0],'top1_id':8},*right[1:]]]:
            with self.assertRaises(ValueError):self.cross.transition_summary(left,bad,['a','b'])

    def test_cross_refuses_incomplete_gate_before_reading_data_or_creating_outputs(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-cross-',dir=ROOT/'runs') as directory:
            root=Path(directory);(root/'gate.json').write_text(json.dumps({'status':'RUNNING'}))
            with self.assertRaises(ValueError):self.cross.run_cross(root,root/'missing_rows',root,root/'output','cpu')
            self.assertFalse((root/'output').exists())


if __name__=='__main__':unittest.main()
