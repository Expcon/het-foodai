"""Optional GUI integration test: run under Xvfb on Linux.
No physical camera, user files, serial port or network touched.
On Windows use your real USB camera instead of this synthetic fake.
"""
import os
import sys
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import oven_capture_gui as gui

class FakeCamera:
    def __init__(self, index, backend):
        self.index=index
        self.backend=backend
        self.opened = index == 1 and backend == cv2.CAP_DSHOW
        self.frame=np.zeros((180,320,3), dtype=np.uint8)
        self.frame[:,:,:] = [18, 70, 112]
        self.frame[35:125,90:225, :] = [4, 220, 255]
    def isOpened(self): return self.opened
    def set(self, *args): return True
    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH:320, cv2.CAP_PROP_FRAME_HEIGHT:180,
                cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'YUY2'),
                cv2.CAP_PROP_FPS:15, cv2.CAP_PROP_EXPOSURE:-7.0}.get(prop, 0.)
    def read(self):
        time.sleep(.05)
        return True, self.frame.copy()
    def release(self):self.opened=False

def pump(root, duration=.5):
    stop=time.monotonic()+duration
    while time.monotonic()<stop:
        root.update_idletasks(); root.update(); time.sleep(.02)

def main():
    with tempfile.TemporaryDirectory() as td:
        out=Path(td)/'中文路径'/'测试照片'
        root=gui.RootClass()
        app=gui.DesktopApp(root)
        app.output.set(str(out)); app.device.set('1'); app.photo_mode.set('数据集采集（保存JSON）'); app.sample_id.set('pilot_001')
        app.backend.set('DirectShow (dshow)')
        with patch.object(cv2,'VideoCapture', FakeCamera):
            app.toggle_camera();pump(root,.6)
            assert app.camera_ready, 'GUI did not receive READY from fake camera'
            assert app.latest_frame is not None
            app.save_photo(); app.save_photo();pump(root,1.1)
            assert len(list(out.glob('*.png')))==2, list(out.iterdir())
            assert len(list(out.glob('pilot_001_*.json')))==2
            # Real frames are spaced apart; 5-shot burst should save 5 unique shots.
            app.sample_id.set('carrot_001')
            app.burst_count.set('5')
            app.burst_interval.set('0.25')
            app.photo_mode.set('快速拍照（不存JSON）')
            app.toggle_burst();pump(root,2.1)
            assert not app.burst_running, 'Burst did not finish'
            numbered = sorted(out.glob('carrot_001_*.png'))
            assert len(numbered)==5, numbered
            assert [v.name for v in numbered] == [f'carrot_001_{i:04d}.png' for i in range(1,6)]
            assert not list(out.glob('carrot_001_*.json'))
            app.photo_mode.set('数据集采集（保存JSON）')
            app.toggle_recording();pump(root,.4)
            assert app.recording
            pump(root,.45)
            app.toggle_recording();pump(root,.6)
            assert not app.recording
            assert len(list(out.glob('oven_video*.avi')))==1
            assert len(list(out.glob('oven_video*.json')))==1
            app.toggle_camera();pump(root,.4)
            assert not app.camera_ready
        app._on_close()
        print('PASS GUI: named/manual camera -> preview -> 2 metadata PNGs -> burst 5 numbered PNGs -> AVI -> safe stop')

if __name__=='__main__': main()
