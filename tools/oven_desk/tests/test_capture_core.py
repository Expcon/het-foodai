import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import cv2

from capture_core import PhotoStore, VideoRecorder, write_exclusive, file_sha256


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name) / "含中文和空格" / "蔬菜 2026"
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.frame[:, :, 1] = 128
        self.frame[100:500, 300:1000, 2] = 230

    def tearDown(self):
        self.temp.cleanup()

    def test_three_frames_unique_and_unicode_paths(self):
        store = PhotoStore(self.folder)
        saved = [store.save(self.frame, {"exposure": -7}) for _ in range(3)]
        self.assertEqual(len(set(p for p, _ in saved)), 3)
        self.assertEqual(len(list(self.folder.glob("*.png"))), 3)
        for img, sidecar in saved:
            original = cv2.imdecode(np.frombuffer(img.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(original, self.frame)
            info = json.loads(sidecar.read_text("utf-8"))
            self.assertEqual(info["sha256"], file_sha256(img))
            self.assertEqual(info["camera_context"]["exposure"], -7)
            with self.assertRaises(FileExistsError):
                write_exclusive(img, [b"data"])
            np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(img.read_bytes(), np.uint8), 1), self.frame)

    def test_restart_never_overwrites(self):
        PhotoStore(self.folder).save(self.frame)
        PhotoStore(self.folder).save(self.frame)
        self.assertEqual(len(list(self.folder.glob("*.png"))), 2)

    def test_video_unicode_path_and_no_overwrite(self):
        recorder = VideoRecorder(self.folder, self.frame, fps=12)
        for _ in range(3):
            recorder.write(self.frame)
        out = recorder.stop()
        self.assertIsNotNone(out)
        video, meta, details = out
        self.assertEqual(details["frame_count"], 3)
        self.assertEqual(file_sha256(video), details["sha256"])
        self.assertTrue(meta.is_file())
        cap = cv2.VideoCapture(str(video))
        try:
            self.assertTrue(cap.isOpened())
            ok, pic = cap.read()
            self.assertTrue(ok)
            self.assertEqual(pic.shape, self.frame.shape)
        finally:
            cap.release()
        with self.assertRaises(FileExistsError):
            write_exclusive(video, [b"bad"])

    def test_invalid_frame(self):
        with self.assertRaises(ValueError):
            PhotoStore(self.folder).save(np.zeros((3,4), dtype=np.uint8))

if __name__ == "__main__":
    unittest.main()
