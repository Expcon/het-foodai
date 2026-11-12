"""SYNTHETIC UI ownership/persistence tests; no food accuracy or camera evidence."""
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np


class RecognitionSessionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('recognition_session'), 'Recognition session is required')
        self.module=importlib.import_module('recognition_session')
        self.session=self.module.RecognitionSession()
        self.frame=np.full((45,80,3),114,dtype=np.uint8)
        self.identity={'display_name':'SYNTHETIC model','model_sha256':'a'*64,'classes_sha256':'b'*64,'checkpoint_sha256':'c'*64}
        self.prediction={'task':'classification','top3':[{'id':9,'label':'corn','score':.6}], 'probabilities':[.4,.6]}
        self.session.set_source('file','SYNTHETIC A.png',self.frame,original_sha256='d'*64)

    def test_request_owns_copied_frame_and_cannot_submit_twice(self):
        request=self.session.begin(self.frame,self.identity)
        self.frame[:]=0;self.identity['display_name']='changed'
        self.assertTrue(np.all(request.frame==114));self.assertEqual(request.model_identity['display_name'],'SYNTHETIC model')
        with self.assertRaises(RuntimeError):self.session.begin(self.frame,self.identity)
        self.assertTrue(self.session.finish(request,self.prediction,12.5))
        self.assertEqual(self.session.record['prediction'],self.prediction)
        self.assertEqual(self.session.record['latency_ms'],12.5)

    def test_a_to_b_to_a_rejects_stale_completion_by_generation_not_hash(self):
        request=self.session.begin(self.frame,self.identity)
        self.session.set_source('file','B.png',self.frame)
        self.session.set_source('file','SYNTHETIC A.png',self.frame,original_sha256='d'*64)
        self.assertFalse(self.session.finish(request,self.prediction,1))
        self.assertIsNone(self.session.record);self.assertIsNone(self.session.active)

    def test_model_change_rejects_pending_result_and_clears_saved_success(self):
        request=self.session.begin(self.frame,self.identity)
        self.session.invalidate_model()
        self.assertFalse(self.session.finish(request,self.prediction,1))
        request=self.session.begin(self.frame,self.identity);self.session.finish(request,self.prediction,1)
        self.session.invalidate_model();self.assertIsNone(self.session.record)

    def test_old_error_cannot_clear_a_new_request_or_replace_new_source_message(self):
        first=self.session.begin(self.frame,self.identity)
        self.session.finish(first,self.prediction,1)
        second=self.session.begin(self.frame,self.identity)
        self.assertFalse(self.session.fail(first,'stale error'));self.assertIs(self.session.active,second)
        self.session.set_source('file','B.png',self.frame)
        self.assertFalse(self.session.fail(second,'stale error'));self.assertEqual(self.session.error,'')

    def test_failure_and_source_disconnect_do_not_keep_previous_success(self):
        first=self.session.begin(self.frame,self.identity);self.session.finish(first,self.prediction,1)
        second=self.session.begin(self.frame,self.identity);self.assertIsNone(self.session.record)
        self.assertTrue(self.session.fail(second,'decode failed'));self.assertEqual(self.session.error,'decode failed')
        self.session.set_source('camera','Disconnected');self.assertIsNone(self.session.record)
        with self.assertRaises(ValueError):self.session.begin(None,self.identity)

    def test_unmeasured_or_invalid_latency_is_rejected(self):
        for value in [None,-1,float('nan'),float('inf')]:
            request=self.session.begin(self.frame,self.identity)
            with self.assertRaises(ValueError):self.session.finish(request,self.prediction,value)
            self.session.fail(request,'invalid timing')

    def test_save_binds_inference_frame_identity_and_uses_unique_new_bundle(self):
        request=self.session.begin(self.frame,self.identity);self.session.finish(request,self.prediction,9.3)
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-result-') as td:
            first=self.session.save_result(td);second=self.session.save_result(td)
            self.assertNotEqual(first,second)
            metadata=json.loads((first/'result.json').read_text('utf-8'))
            image=first/'frame.png';decoded=cv2.imdecode(np.frombuffer(image.read_bytes(),dtype=np.uint8),1)
            self.assertTrue(np.array_equal(decoded,request.frame))
            self.assertEqual(metadata['frame_sha256'],self.module.frame_sha256(decoded))
            self.assertEqual(metadata['saved_png_sha256'],self.module.file_sha256(image))
            self.assertEqual(metadata['source']['original_sha256'],'d'*64)
            self.assertEqual(metadata['model']['model_sha256'],'a'*64)
            self.assertEqual(metadata['prediction']['top3'][0]['label'],'corn')
            self.assertNotIn('accuracy',metadata)

    def test_incomplete_save_cleans_only_its_new_bundle(self):
        request=self.session.begin(self.frame,self.identity);self.session.finish(request,self.prediction,1)
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-write-fail-') as td:
            old=Path(td)/'existing.txt';old.write_text('keep')
            original=self.module.write_exclusive
            def fail_json(path,chunks):
                if Path(path).suffix=='.json':raise OSError('SYNTHETIC no space')
                original(path,chunks)
            with patch.object(self.module,'write_exclusive',side_effect=fail_json):
                with self.assertRaises(OSError):self.session.save_result(td)
            self.assertEqual(list(Path(td).iterdir()),[old]);self.assertEqual(old.read_text(),'keep')


if __name__=='__main__':unittest.main()
