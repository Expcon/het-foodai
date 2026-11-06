"""GUI test of named camera selection and safe backend-specific fallback."""
from pathlib import Path
import sys
from unittest.mock import patch
import time
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import oven_capture_gui as gui
from camera_devices import CameraChoice


def pump(root,duration=.25):
    end=time.monotonic()+duration
    while time.monotonic()<end:
        root.update_idletasks();root.update();time.sleep(.02)


def main():
    first=CameraChoice(1,"dshow","icspring camera",vid=0x32f6,pid=0x9251)
    second=CameraChoice(0,"msmf","icspring camera",vid=0x32f6,pid=0x9251)
    root=gui.RootClass()
    app=gui.DesktopApp(root)
    with patch.object(gui,"list_cameras",return_value=[first,second]):
        app.refresh_cameras()
        pump(root,.3)
    assert first.display in app.named_choices, app.named_choices
    assert second.display in app.named_choices
    app.camera_choice.set(first.display)
    app._camera_selected()
    assert app.device.get()=="1"
    assert app.backend.get()=="DirectShow (dshow)"
    app.backend.set("自动兼容（同设备）")
    attempts, *_ = app._get_camera_inputs()
    assert [(x.index,x.backend) for x in attempts]==[(1,"dshow"),(0,"msmf")],attempts
    app.camera_choice.set(second.display)
    app._camera_selected()
    attempts,*_=app._get_camera_inputs()
    assert [(x.index,x.backend) for x in attempts]==[(0,"msmf")],attempts
    app.camera_choice.set("手动编号（下面填写）")
    app.device.set("1")
    app.backend.set("自动兼容（同设备）")
    try:
        app._get_camera_inputs()
    except ValueError as exc:
        assert "手动编号不能自动切换" in str(exc)
    else:
        raise AssertionError("Manual camera index was silently used across backends")
    app._on_close()
    print("PASS GUI: real camera name choices + per-backend index mapping + safe manual fallback")

if __name__=='__main__': main()
