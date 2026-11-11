"""SYNTHETIC data-qualification tests; no food labels or accuracy evidence."""
import importlib
import importlib.util
from pathlib import Path
import tempfile
import json
import unittest

from PIL import Image
from tools.oven_training.model import ROOT


class SupplementTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('tools.oven_training.supplement'), 'Supplement audit missing')
        self.module=importlib.import_module('tools.oven_training.supplement')

    def row(self, marker='1', **changes):
        return dict(image_id='SYNTHETIC-'+marker,sha256=marker*64,pixel_sha256=marker*64,phash='123456789abcdef0',
                    candidate_class_key='potato',class_id=6,decode_status='OK',declared_group='folder-'+marker,
                    physical_sample_id='physical-'+marker,session_id='session-'+marker,parent_source_group_id='parent-'+marker,
                    source_evidence='SYNTHETIC source record',physical_identity_verified=True,session_identity_verified=True,
                    identity_reviewed_by='SYNTHETIC human',identity_reviewed_at='2026-11-11',trusted_label=True,
                    label_reviewed_by='SYNTHETIC human',label_reviewed_at='2026-11-11',label_evidence='SYNTHETIC manual material check',
                    training_independence_verified=True,training_independence_evidence='SYNTHETIC checked disjoint source register',
                    intended_use='dev',training_byte_matches=[],training_pixel_matches=[],near_training_matches=0,
                    near_relation_reviewed=True,**changes)

    def train(self, verified=True):
        return [dict(sha256='a'*64,pixel_sha256='b'*64,physical_sample_id='old-object',session_id='old-session',
                     parent_source_group_id='old-parent',physical_identity_verified=verified,session_identity_verified=verified)]

    def test_folder_numbers_and_novel_hashes_never_prove_independence(self):
        row=self.row();row.update(physical_identity_verified=False,session_identity_verified=False,trusted_label=False)
        result=self.module.qualify([row],self.train(False))[0]
        self.assertEqual(result['route'],'diagnostic_only')
        self.assertIn('training_identity_incomplete',result['qualification_reasons'])
        self.assertIsNone(result['independent_physical_count'])

    def test_complete_explicit_dev_evidence_can_qualify_but_missing_parent_or_label_cannot(self):
        good=self.module.qualify([self.row()],self.train())[0]
        self.assertEqual(good['route'],'qualified_dev')
        for change in [dict(parent_source_group_id='unknown'),dict(trusted_label=False),dict(intended_use='unassigned'),
                       dict(label_reviewed_by=''),dict(training_independence_verified=False),dict(near_training_matches=1,near_relation_reviewed=False)]:
            row=self.row();row.update(change)
            self.assertEqual(self.module.qualify([row],self.train())[0]['route'],'diagnostic_only')

    def test_sealed_role_propagates_through_session_parent_and_pixel_relations(self):
        for field in ['session_id','physical_sample_id','parent_source_group_id','pixel_sha256','sha256','declared_group']:
            a=self.row('1');a['intended_use']='final_test'
            b=self.row('2');b[field]=a[field]
            result=self.module.qualify([a,b],self.train())
            self.assertEqual([r['route'] for r in result],['sealed_excluded','sealed_excluded'],field)

    def test_training_identity_or_duplicate_overlap_blocks_entire_linked_group(self):
        a=self.row('1');a['training_pixel_matches']=['SYNTHETIC-old']
        b=self.row('2');b['declared_group']=a['declared_group']
        self.assertTrue(all(r['route']=='diagnostic_only' for r in self.module.qualify([a,b],self.train())))
        for field in ['physical_sample_id','session_id','parent_source_group_id']:
            row=self.row();row[field]=self.train()[0][field]
            self.assertEqual(self.module.qualify([row],self.train())[0]['route'],'diagnostic_only')

    def test_weak_labels_never_populate_dev_accuracy_and_all12_classes_remain(self):
        row=self.row();row.update(route='diagnostic_only',independent_physical_count=None)
        prediction={**row,'v2':{'top1_id':7,'top3_ids':[7,4,6]},'R1':{'top1_id':6,'top3_ids':[6,7,4]}}
        result=self.module.score([prediction])
        self.assertIsNone(result['v2']['development']['top1'])
        self.assertEqual(result['R1']['diagnostic']['candidate_top1_matches'],1)
        self.assertEqual(len(result['R1']['development']['confusion']),12)
        self.assertEqual(result['R1']['development']['per_class']['croissant']['images'],0)
        self.assertIsNone(result['R1']['diagnostic']['physical_objects'])
        self.assertIsNone(result['R1']['diagnostic']['independent_source_groups'])

    def test_only_explicit_qualified_dev_rows_receive_dev_metrics(self):
        row=self.module.qualify([self.row()],self.train())[0]
        prediction={**row,'v2':{'top1_id':7,'top3_ids':[7,4,6]},'R1':{'top1_id':6,'top3_ids':[6,7,4]}}
        result=self.module.score([prediction])
        self.assertEqual(result['v2']['development']['top1'],0)
        self.assertEqual(result['v2']['development']['top3'],1)
        self.assertEqual(result['R1']['development']['top1'],1)

    def test_physical_specimens_and_correlated_parent_groups_are_counted_separately(self):
        a=self.row('1');b=self.row('2');b['parent_source_group_id']=a['parent_source_group_id']
        rows=self.module.qualify([a,b],self.train())
        prediction={'top1_id':6,'top3_ids':[6,7,4]}
        result=self.module.score([{**row,'v2':prediction,'R1':prediction} for row in rows])
        metrics=result['R1']['development']
        self.assertEqual(metrics['physical_objects'],2)
        self.assertEqual(metrics['independent_source_groups'],1)
        self.assertEqual(metrics['per_class']['potato']['physical_objects'],2)
        self.assertEqual(metrics['per_class']['potato']['independent_source_groups'],1)

    def test_real_audit_white_corn_mapping_and_original_bytes_preserved(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-supplement-',dir=ROOT/'runs') as directory:
            root=Path(directory);source=root/'source';source.mkdir()
            photo=source/'玉米_白/玉米1/SYNTHETIC.png';photo.parent.mkdir(parents=True);Image.new('RGB',(48,32),(130,80,40)).save(photo)
            original=photo.read_bytes()
            rows=self.module.audit(source,[],root/'audit')
            self.assertEqual(rows[0]['candidate_class_key'],'corn');self.assertEqual(rows[0]['class_id'],9)
            self.assertEqual(rows[0]['route'],'diagnostic_only')
            self.assertEqual(photo.read_bytes(),original)
            with self.assertRaises(FileExistsError):self.module.audit(source,[],root/'audit')

    def test_plan_tampering_is_rejected_before_loading_any_checkpoint(self):
        from tools.oven_training.model import digest
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-plan-',dir=ROOT/'runs') as directory:
            root=Path(directory);source=root/'source';source.mkdir()
            self.module.audit(source,[],root/'audit')
            plan=root/'audit/pre_prediction_plan.json';sha=digest(plan)
            plan.write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError):self.module.predict(root/'audit',root/'predict',sha,root/'missing-v2',root/'missing-r1','cpu')

    def test_corrupt_picture_is_reported_without_creating_fake_prediction(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-corrupt-',dir=ROOT/'runs') as directory:
            root=Path(directory);source=root/'source';(source/'土豆').mkdir(parents=True)
            (source/'土豆/bad.png').write_bytes(b'SYNTHETIC not an image')
            rows=self.module.audit(source,[],root/'audit')
            self.assertEqual(rows[0]['decode_status'],'FAIL');self.assertEqual(rows[0]['route'],'diagnostic_only')


if __name__=='__main__':unittest.main()
