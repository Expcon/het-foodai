import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
from tools.oven_training.export_desk import compare_probabilities, make_contract, revoke_mnn
from tools.oven_training.model import PREPROCESS, class_digest


class OvenExportTests(unittest.TestCase):
    def test_alignment_requires_all_probabilities_and_ordered_top3(self):
        p = np.array([.5, .3, .2], dtype=np.float32)
        self.assertTrue(compare_probabilities(p, p)['pass'])
        self.assertFalse(compare_probabilities(p, np.array([.5,.2,.3]))['pass'])
        self.assertFalse(compare_probabilities(p, p + np.array([.001,-.001,0]))['pass'])
        for bad in (np.array([np.nan,.3,.2]), np.array([.5,.5]), p[None]):
            with self.assertRaises(ValueError): compare_probabilities(p,bad)

    def test_contract_preserves_experimental_namespace(self):
        keys=['asparagus','bacon','broccoli','chicken_wing','croissant','french_fries','potato','salmon','toast']
        meta={'class_keys':keys,'class_list_sha256':class_digest(keys),'preprocess':PREPROCESS}
        c=make_contract(meta,'a'*64,'b'*64)
        self.assertEqual([r['key'] for r in c['classes']], keys)
        self.assertEqual(c['preprocess'],PREPROCESS)
        self.assertFalse(c['independent_validation'])
        self.assertEqual(c['models']['onnx']['sha256'],'b'*64)
        self.assertNotIn('accuracy',c)

    def test_recheck_revokes_prior_mnn_approval_before_runtime_can_fail(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'classes.json'
            p.write_text(json.dumps({'models':{'onnx':{'sha256':'a'*64},'mnn':{'sha256':'b'*64}}}))
            result=revoke_mnn(p)
            self.assertNotIn('mnn',result['models'])
            self.assertEqual(json.loads(p.read_text()),result)
            self.assertEqual(result['models']['onnx']['sha256'],'a'*64)

    def test_mnn_out_of_range_scores_are_failure_not_clamped(self):
        p=np.array([.9999997,.0000002,.0000001])
        q=np.array([1.000244,.0000002,.0000001])
        result=compare_probabilities(p,q)
        self.assertTrue(result['top3_agree'])
        self.assertFalse(result['valid_probabilities'])
        self.assertFalse(result['pass'])
