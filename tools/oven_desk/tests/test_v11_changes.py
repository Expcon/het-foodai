"""Regression coverage for group feedback and camera index/identity safety."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from capture_core import PhotoStore, VideoRecorder, safe_sample_prefix, file_sha256
from camera_devices import CameraChoice, compatible_alternate, manual_choice, list_cameras


class PhotoNamingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.dir=Path(self.temp.name)/"中文照片 组员"
        self.frame=np.zeros((180,320,3),dtype=np.uint8)
        self.frame[:,:,1]=160

    def tearDown(self):
        self.temp.cleanup()

    def test_numbered_names_survive_restart_and_gaps(self):
        store=PhotoStore(self.dir)
        a,meta=store.save(self.frame,{"physical_sample_id":"carrot_001"})
        b,_=PhotoStore(self.dir).save(self.frame,{"physical_sample_id":"carrot_001"})
        self.assertEqual((a.name,b.name),("carrot_001_0001.png","carrot_001_0002.png"))
        self.assertTrue(meta.exists())
        (self.dir/"carrot_001_0009.png").write_bytes(b"existing")
        c,_=PhotoStore(self.dir).save(self.frame,{"physical_sample_id":"carrot_001"})
        self.assertEqual(c.name,"carrot_001_0010.png")
        self.assertEqual((self.dir/"carrot_001_0009.png").read_bytes(),b"existing")

    def test_quick_mode_does_not_create_json(self):
        path,meta=PhotoStore(self.dir).save(self.frame,{"physical_sample_id":"corn_003"},include_metadata=False)
        self.assertEqual(path.name,"corn_003_0001.png")
        self.assertIsNone(meta)
        self.assertFalse(list(self.dir.glob("*.json")))

    def test_jpg_and_json_metadata(self):
        p,info=PhotoStore(self.dir).save(self.frame,{"physical_sample_id":"白萝卜 01"},image_format="JPG")
        self.assertEqual(p.name,"白萝卜_01_0001.jpg")
        self.assertEqual(json.loads(info.read_text(encoding="utf-8"))["sha256"],file_sha256(p))
        decoded=cv2.imdecode(np.frombuffer(p.read_bytes(),dtype=np.uint8),1)
        self.assertEqual(decoded.shape,self.frame.shape)

    def test_concurrent_saves_never_overwrite(self):
        paths=[]
        lock=threading.Lock()
        def task():
            p,_=PhotoStore(self.dir).save(self.frame,{"physical_sample_id":"corn_1"},include_metadata=False)
            with lock:paths.append(p)
        workers=[threading.Thread(target=task) for _ in range(12)]
        for t in workers:t.start()
        for t in workers:t.join()
        self.assertEqual(len(paths),12)
        self.assertEqual(len(set(paths)),12)
        self.assertEqual(len(list(self.dir.glob("corn_1_*.png"))),12)

    def test_video_without_sidecar(self):
        recorder=VideoRecorder(self.dir,self.frame,12,include_metadata=False)
        for _ in range(3):recorder.write(self.frame)
        p,sidecar,summary=recorder.stop()
        self.assertTrue(p.exists())
        self.assertIsNone(sidecar)
        self.assertEqual(summary["frame_count"],3)
        self.assertEqual(len(list(self.dir.glob("*.json"))),0)

    def test_sample_names_are_safe(self):
        self.assertEqual(safe_sample_prefix(" carrot_001 "),"carrot_001")
        self.assertEqual(safe_sample_prefix("CON"),"sample_CON")
        self.assertEqual(safe_sample_prefix("../abc"),"abc")


class CameraIdentityTests(unittest.TestCase):
    def test_backend_specific_indices(self):
        first=CameraChoice(1,"dshow","icspring camera",vid=0x32F6,pid=0x9251)
        second=CameraChoice(0,"msmf","icspring camera",vid=0x32F6,pid=0x9251)
        self.assertEqual(compatible_alternate(first,[first,second]),second)
        self.assertNotEqual(first.index,second.index)

    def test_ambiguous_name_does_not_fallback(self):
        first=CameraChoice(1,"dshow","USB Camera")
        second=CameraChoice(2,"dshow","USB Camera")
        third=CameraChoice(0,"msmf","USB Camera")
        self.assertIsNone(compatible_alternate(first,[first,second,third]))

    def test_manual_requires_explicit_backend(self):
        with self.assertRaises(ValueError):manual_choice(1,"auto")
        self.assertEqual(manual_choice(1,"dshow").index,1)

    def test_enumeration_uses_backend_specific_indices_without_opening_camera(self):
        import sys
        import types
        def listing(api):
            if api==cv2.CAP_DSHOW:
                return [types.SimpleNamespace(index=1,name="icspring camera",path="device-A",vid=1,pid=2,backend=api)]
            if api==cv2.CAP_MSMF:
                return [types.SimpleNamespace(index=0,name="icspring camera",path="device-A",vid=1,pid=2,backend=api)]
            return []
        with patch.dict(sys.modules,{"cv2_enumerate_cameras":types.SimpleNamespace(enumerate_cameras=listing)}):
            found=list_cameras(cv2)
        self.assertEqual({(x.backend,x.index) for x in found},{("dshow",1),("msmf",0)})
        self.assertIsNotNone(compatible_alternate(found[0],found))


if __name__=="__main__":unittest.main()
