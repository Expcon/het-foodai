import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
from tools.oven_training.export_desk import compare_probabilities, make_contract, revoke_mnn, pending_contract
from tools.oven_training.model import PREPROCESS, class_digest
from tools.oven_training import export_desk


class OvenExportTests(unittest.TestCase):
    def test_r2_and_diagnostic_roots_are_explicit_readonly_and_confined(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-r2-roots-') as td:
            root=Path(td);old=root/'old';new=root/'new';diag=root/'diag'
            for path in [old,new,diag]:path.mkdir()
            resolve=export_desk.resolve_alignment_image
            self.assertEqual(resolve({'source':'new','relative_path':'old.png'},old,new,diag),old/'old.png')
            self.assertEqual(resolve({'source':'r2_whole_corn','relative_path':'new.png'},old,new,diag),new/'new.png')
            self.assertEqual(resolve({'source':'supplement_diagnostic','relative_path':'diag.png'},old,new,diag),diag/'diag.png')
            for row,r2root,droot in [({'source':'r2_whole_corn','relative_path':'new.png'},None,diag),
                                   ({'source':'supplement_diagnostic','relative_path':'diag.png'},new,None),
                                   ({'source':'r2_whole_corn','relative_path':'../escape.png'},new,diag)]:
                with self.assertRaises(ValueError):resolve(row,old,r2root,droot)

    def test_extra_rows_must_be_exposed_diagnostics_never_sealed_test(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-extra-rows-') as td:
            original=Path(td)/'rows.json';extra=Path(td)/'extra.json'
            original.write_text(json.dumps([{'image_id':'old','source':'new'}]))
            record={'image_id':'diag','source':'supplement_diagnostic','route':'diagnostic_only'}
            extra.write_text(json.dumps([record]))
            self.assertEqual(len(export_desk.read_alignment_rows(original,extra)),2)
            for change in [dict(route='sealed_excluded'),dict(source='unknown'),dict(route='qualified_dev'),
                           dict(new_split='test'),dict(intended_use='final_test'),dict(reserved_final_test=True)]:
                extra.write_text(json.dumps([{**record,**change}]))
                with self.assertRaises(ValueError):export_desk.read_alignment_rows(original,extra)

    def test_pending_alignment_has_no_loadable_model_authorization(self):
        contract={'format':'het-oven-classifier/v2','models':{'onnx':{'sha256':'a'*64}}}
        pending=pending_contract(contract)
        self.assertEqual(pending['models'],{})
        self.assertEqual(pending['alignment_status'],'PENDING')
        self.assertIn('onnx',contract['models'])

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
