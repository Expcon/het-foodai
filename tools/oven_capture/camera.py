"""Passive inventory plus explicitly selected, optional UVC/OpenCV adapter."""
from __future__ import annotations

import json
import math
from pathlib import Path
import platform
import re
import subprocess
from typing import Protocol

from PIL import Image

from .core import NotReady, now


class FrameSource(Protocol):
    """Vendor adapters must implement this decoded-image interface explicitly."""
    def read(self) -> tuple[Image.Image, dict]: ...
    def close(self) -> None: ...


def inventory():
    """Never opens a device or maps PnP identity to a guessed OpenCV index."""
    devices = []
    try:
        if platform.system() == 'Windows':
            command = ("Get-CimInstance Win32_PnPSignedDriver | "
                       "Where-Object { $_.DeviceClass -in @('CAMERA','IMAGE') } | "
                       "Select-Object DeviceName,DeviceID,DriverVersion,DriverProviderName | ConvertTo-Json -Compress")
            result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                                    capture_output=True, text=True, timeout=20, check=True)
            if result.stdout.strip():
                parsed = json.loads(result.stdout)
                devices = parsed if isinstance(parsed, list) else [parsed]
        elif platform.system() == 'Linux':
            for device in sorted(Path('/sys/class/video4linux').glob('video*')):
                devices.append({'device': '/dev/' + device.name,
                    'name': (device / 'name').read_text().strip(),
                    'driver': (device / 'device/driver').resolve().name if (device / 'device/driver').exists() else 'UNKNOWN'})
        else:
            return {'state': 'BLOCKED', 'reason': 'Passive inventory adapter unavailable for this OS', 'devices': []}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {'state': 'BLOCKED', 'reason': f'Inventory failed: {type(exc).__name__}', 'devices': [], 'captured_at': now()}
    return {'state': 'UNKNOWN' if devices else 'BLOCKED', 'devices': devices, 'captured_at': now(),
            'opened_any_device': False, 'actual_resolution': 'UNKNOWN_UNTIL_SELECTED_PROBE',
            'reason': 'Inventory only; not proof of an oven camera or index mapping' if devices else 'No camera/image device enumerated'}


class OpenCVSource:
    """Decoded uint8 BGR frames only; no sensor RAW/encoded packet acquisition."""
    def __init__(self, device, backend, *, width=None, height=None):
        if backend not in {'dshow', 'msmf', 'v4l2'}:
            raise ValueError('Choose an explicit supported backend; vendor SDK is not configured')
        device = str(device)
        if device.isdecimal():
            selected = int(device)
        elif backend == 'v4l2' and re.fullmatch(r'/dev/video\d+', device):
            selected = device
        else:
            raise ValueError('Explicit nonnegative index or /dev/videoN required; no URLs/unknown interfaces')
        if (width is None) != (height is None) or any(type(n) is not int or n <= 0 for n in [width, height] if n is not None):
            raise ValueError('Requested width and height must both be positive integers')
        try:
            import cv2
        except ImportError as exc:
            raise NotReady('Optional opencv-python is absent; offline import/audit remain usable') from exc
        self.cv2 = cv2
        self.cap = None
        try:
            self.cap = cv2.VideoCapture(selected, {'dshow': cv2.CAP_DSHOW, 'msmf': cv2.CAP_MSMF, 'v4l2': cv2.CAP_V4L2}[backend])
            if not self.cap.isOpened():
                raise NotReady('Selected camera cannot be opened; confirm interface/permissions, do not scan other indices')
            if width is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        except (cv2.error, OSError, NotReady) as exc:
            self.close()
            raise NotReady(f'Selected camera initialization failed: {type(exc).__name__}') from exc
        self.requested = {'width': width, 'height': height}
        self.backend = backend

    def read(self):
        try:
            ok, frame = self.cap.read()
        except (self.cv2.error, OSError) as exc:
            raise NotReady(f'Selected camera read failed: {type(exc).__name__}') from exc
        if not ok or frame is None:
            raise NotReady('Selected device returned no frame')
        if str(frame.dtype) != 'uint8' or frame.ndim != 3 or frame.shape[2] != 3:
            raise NotReady('Unsupported decoded frame layout; vendor/RAW adapter requires explicit implementation')
        cv2 = self.cv2
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        fps = self._reported(cv2.CAP_PROP_FPS)
        fourcc_value = self._reported(cv2.CAP_PROP_FOURCC)
        fourcc = int(fourcc_value) if fourcc_value is not None and fourcc_value <= 0xffffffff else 0
        details = {'backend': self.backend, 'requested_resolution': self.requested,
            'reported_resolution': [self._reported(cv2.CAP_PROP_FRAME_WIDTH), self._reported(cv2.CAP_PROP_FRAME_HEIGHT)],
            'actual_decoded_resolution': [image.width, image.height],
            'reported_fps_not_measured': fps,
            'reported_fourcc': ''.join(chr((fourcc >> (8 * i)) & 0xff) for i in range(4)) if fourcc else 'UNKNOWN',
            'representation': 'OpenCV BGR uint8 decoded to RGB; PNG encoded on save',
            'encoded_camera_packets_preserved': False, 'sensor_raw_claimed': False,
            'driver_version': 'Use passive inventory; no verified index-to-PnP mapping'}
        return image, details

    def _reported(self, prop):
        try:
            value = float(self.cap.get(prop))
            return value if math.isfinite(value) and value > 0 else None
        except (self.cv2.error, TypeError, ValueError, OSError):
            return None

    def close(self):
        if self.cap is not None:
            self.cap.release()
