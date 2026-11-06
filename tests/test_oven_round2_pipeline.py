"""Synthetic records only: source isolation and metric eligibility."""
from pathlib import Path
import tempfile
import unittest
from tools.oven_training.round2 import resolve_image, diagnostic_metrics, summarize_diagnostic, require_complete_v1_diagnostic, V1_KEYS, V2_KEYS, V1_SHA
from tools.oven_training.export_desk import make_contract
from tools.oven_training.model import class_digest


class Round2PipelineTests(unittest.TestCase):
    def test_training_requires_complete_matching_v1_diagnosis(self):
        rows=[{'source':'new'},{'source':'round1'}]
        report={'state':'DIAGNOSTIC_COMPLETED','checkpoint_sha256':V1_SHA,'rows_sha256':'a'*64,
                'failures':[],'class_keys':V1_KEYS,'images':1}
        require_complete_v1_diagnostic(report,rows,'a'*64)
        for key,value in [('images',0),('failures',None),('class_keys',V2_KEYS),('rows_sha256','b'*64),('checkpoint_sha256','c'*64)]:
            with self.assertRaises(ValueError):require_complete_v1_diagnostic({**report,key:value},rows,'a'*64)

    def test_append_only_classes(self):
        self.assertEqual(V2_KEYS[:9],V1_KEYS)
        self.assertEqual(V2_KEYS[9:],['corn','sausage','steak'])

    def test_round2_contract_requires_fixed_twelve_classes(self):
        meta={'class_keys':V2_KEYS,'class_list_sha256':class_digest(V2_KEYS)}
        result=make_contract(meta,'a'*64,'b'*64,'v2')
        self.assertEqual(result['format'],'het-oven-classifier/v2')
        self.assertEqual(result['output_shape'],[1,12])
        with self.assertRaises(ValueError):make_contract({'class_keys':V1_KEYS},'a'*64,'b'*64,'v2')

    def test_scene_evidence_does_not_fragment_strata(self):
        rows=[{'class_key':'toast','trusted_label':False,'top3':[{'class_key':'toast','score':.8}],
               'foil':{'value':'present','raw':raw}} for raw in ['with_foil','withfoil']]
        result=summarize_diagnostic(rows,V1_KEYS)
        self.assertEqual(list(result['strata']['foil']),['present'])
        self.assertEqual(result['strata']['foil']['present']['candidate_directory_agreement_not_accuracy']['count'],2)

    def test_source_paths_cannot_escape_explicit_root(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            self.assertEqual(resolve_image({'source':'new','relative_path':'x/a.png'},root),root/'x/a.png')
            for rel in ['../a.png','D:/other/a.png','/a.png','x\\a.png']:
                with self.assertRaises(ValueError):resolve_image({'source':'new','relative_path':rel},root)
            with self.assertRaises(ValueError):resolve_image({'source':'unapproved','relative_path':'a.png'},root)

    def test_unreviewed_and_unknown_classes_never_enter_accuracy(self):
        records=[{'class_key':'toast','trusted_label':True,'top3':[{'class_key':'toast'}]},
                 {'class_key':'croissant','trusted_label':False,'top3':[{'class_key':'salmon'}]},
                 {'class_key':'steak','trusted_label':True,'top3':[{'class_key':'bacon'}]}]
        m=diagnostic_metrics(records,V1_KEYS)
        self.assertEqual(m['trusted_supported']['count'],1)
        self.assertEqual(m['trusted_supported']['correct'],1)
        self.assertEqual(m['unsupported_images'],1)
        self.assertEqual(m['candidate_directory_agreement_not_accuracy']['count'],2)
        self.assertIsNone(diagnostic_metrics(records[1:],V1_KEYS)['trusted_supported']['accuracy'])
        self.assertIsNone(m['independent_test_accuracy'])
