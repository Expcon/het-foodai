"""Lossless-by-default camera capture storage and defensive video recording.

No camera is ever opened by importing this module. All output filenames are
unique and opened with O_EXCL; Unicode Windows paths work via Python file IO.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
import shutil
import tempfile
import threading
import time
from uuid import uuid4


def now_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def safe_stem(prefix: str = "oven") -> str:
    t = datetime.now().astimezone()
    return f"{prefix}_{t:%Y%m%d_%H%M%S_%f}_{uuid4().hex[:12]}"


def write_exclusive(path: Path, chunks) -> None:
    """OS-level create-if-absent, no overwrite, clean partial on write failure."""
    path = Path(path)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    try:
        with os.fdopen(fd, "wb") as stream:
            for chunk in chunks:
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def frame_properties(cap, cv2):
    """Best-effort observational read; -1/0 can mean unsupported."""
    prop_names = {
        "exposure": "CAP_PROP_EXPOSURE", "auto_exposure": "CAP_PROP_AUTO_EXPOSURE",
        "gain": "CAP_PROP_GAIN", "brightness": "CAP_PROP_BRIGHTNESS",
        "contrast": "CAP_PROP_CONTRAST", "saturation": "CAP_PROP_SATURATION",
        "gamma": "CAP_PROP_GAMMA", "sharpness": "CAP_PROP_SHARPNESS",
        "white_balance": "CAP_PROP_WHITE_BALANCE_BLUE_U", "auto_white_balance": "CAP_PROP_AUTO_WB",
        "fps_reported_not_measured": "CAP_PROP_FPS",
    }
    result = {}
    for name, attr in prop_names.items():
        try:
            v = float(cap.get(getattr(cv2, attr)))
            import math
            result[name] = v if math.isfinite(v) else None
        except Exception:
            result[name] = None
    return result


def fourcc_to_text(raw):
    try:
        n = int(raw)
        if n < 1:
            return "UNKNOWN"
        s = "".join(chr((n >> (8 * j)) & 255) for j in range(4))
        return s if s.isprintable() and s.strip("\x00 ") else "UNKNOWN"
    except (ValueError, OverflowError, TypeError):
        return "UNKNOWN"


def safe_sample_prefix(value):
    """Short, filesystem-safe visible physical sample ID; no platform-reserved names."""
    value = unicodedata.normalize("NFKC", str(value or "").strip())
    out = []
    for ch in value:
        if ch.isalnum() or ch in "_-":
            out.append(ch)
        elif ch.isspace() or ch in ".:/\\|<>'\"*?":
            out.append("_")
        else:
            out.append("_")
    name = re.sub(r"_+", "_", "".join(out)).strip("_.-")[:60]
    if not name:
        name = "photo"
    # Reserved filenames on Windows, including CON.jpg, COM1.jpg etc.
    if name.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                        *(f"LPT{i}" for i in range(1, 10))}:
        name = "sample_" + name
    return name


class PhotoStore:
    """Numbered, never overwritten photos; optional metadata; PNG/JPEG export."""
    def __init__(self, folder):
        self.folder = Path(folder).expanduser()

    def save(self, frame_bgr, context=None, *, include_metadata=True, image_format="PNG", sample_id=None):
        import cv2
        import numpy as np
        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
                or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3):
            raise ValueError("Expected decoded uint8 BGR camera frame")
        context = dict(context or {})
        prefix = safe_sample_prefix(sample_id if sample_id is not None else context.get("physical_sample_id"))
        fmt = str(image_format).upper()
        if fmt not in {"PNG", "JPG"}:
            raise ValueError("Image format must be PNG or JPG")
        ext = ".png" if fmt == "PNG" else ".jpg"
        encode_options = [] if fmt == "PNG" else [cv2.IMWRITE_JPEG_QUALITY, 95]
        ok, encoded = cv2.imencode(ext, frame_bgr, encode_options)
        if not ok:
            raise OSError("Could not encode photo")
        content = encoded.tobytes()
        self.folder.mkdir(parents=True, exist_ok=True)
        # Use the greatest existing number, then atomically reserve the next.
        # The O_EXCL write is also safe if multiple photo jobs arrive concurrently.
        pattern = re.compile(r"^" + re.escape(prefix) + r"_(\d+)\.(?:png|jpg)$", re.I)
        current = 0
        for f in self.folder.iterdir():
            match = pattern.fullmatch(f.name)
            if match:
                current = max(current, int(match.group(1)))
        for number in range(current + 1, current + 10000):
            stem = f"{prefix}_{number:04d}"
            photo = self.folder / (stem + ext)
            sidecar = self.folder / (stem + ".json")
            if (self.folder / (stem + ".png")).exists() or (self.folder / (stem + ".jpg")).exists() or sidecar.exists():
                continue
            try:
                write_exclusive(photo, [content])
            except FileExistsError:
                continue
            break
        else:
            raise OSError("无法分配新的唯一文件名")
        if not include_metadata:
            return photo, None
        details = {
            "schema": "het-oven-capture/2",
            "image_file": photo.name,
            "sample_prefix": prefix,
            "photo_sequence": number,
            "created_host_local": now_local(),
            "width": int(frame_bgr.shape[1]), "height": int(frame_bgr.shape[0]),
            "format": fmt, "source_representation": "full_decoded_camera_frame",
            "sensor_raw": False, "resized": False, "cropped": False,
            "sha256": hashlib.sha256(content).hexdigest(),
            "label_status": "pending_manual_review",
            "camera_context": context,
        }
        try:
            write_exclusive(sidecar, [(json.dumps(details, ensure_ascii=False, indent=2) + "\n").encode("utf-8")])
        except Exception as err:
            raise OSError(f"照片已保存：{photo}，但元数据失败：{err}") from err
        return photo, sidecar


class VideoRecorder:
    """AVI/MJPEG recording via ASCII-only temporary path, copied safely to Unicode output."""

    def __init__(self, folder, first_frame, fps: float, context=None, include_metadata=True):
        import cv2
        import numpy as np
        if fps < 1 or fps > 60:
            raise ValueError("Video FPS must be within 1..60")
        if not isinstance(first_frame, np.ndarray) or first_frame.ndim != 3 or first_frame.shape[2] != 3:
            raise ValueError("Invalid frame shape")
        self.target_folder = Path(folder).expanduser()
        self.target_folder.mkdir(parents=True, exist_ok=True)
        self.height, self.width = first_frame.shape[:2]
        self.fps = float(fps)
        self.context = dict(context or {})
        self.include_metadata = bool(include_metadata)
        self.started = now_local()
        self.started_mono = time.monotonic()
        self.frames = 0
        self.first_frame_at = None
        self.last_frame_at = None
        self.closed = False
        self.stem = safe_stem("oven_video")
        self._temp_dir = tempfile.TemporaryDirectory(prefix="het_oven_video_")
        self._temp = Path(self._temp_dir.name) / (uuid4().hex + ".avi")
        # Some cv2 builds cannot write to non-ASCII paths. We create AVI in temporary
        # directory then move its bytes through Python file IO with O_EXCL.
        self._writer = cv2.VideoWriter(
            str(self._temp), cv2.VideoWriter_fourcc(*"MJPG"), self.fps,
            (self.width, self.height))
        if not self._writer.isOpened():
            self._temp_dir.cleanup()
            raise OSError("OpenCV MJPG/AVI encoder unavailable")
        self._lock = threading.Lock()

    def write(self, frame):
        import numpy as np
        with self._lock:
            if self.closed:
                raise RuntimeError("Recorder already closed")
            if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or tuple(frame.shape) != (self.height, self.width, 3):
                raise ValueError("Camera frame size/dtype changed during recording")
            self._writer.write(frame)
            timestamp = now_local()
            if self.first_frame_at is None:
                self.first_frame_at = timestamp
            self.last_frame_at = timestamp
            self.frames += 1

    def stop(self):
        with self._lock:
            if self.closed:
                return None
            self.closed = True
            self._writer.release()
            try:
                if self.frames == 0 or not self._temp.is_file() or self._temp.stat().st_size < 500:
                    raise OSError("No complete frames were recorded")
                final = self.target_folder / (self.stem + ".avi")
                with self._temp.open("rb") as reader:
                    write_exclusive(final, iter(lambda: reader.read(1 << 20), b""))
                if not self.include_metadata:
                    return final, None, {"frame_count": self.frames, "elapsed_wall_s": round(time.monotonic() - self.started_mono, 3)}
                meta = {
                    "schema": "het-oven-video/1", "video_file": final.name,
                    "codec": "MJPG", "container": "AVI", "no_audio": True,
                    "declared_fps": self.fps, "frame_count": self.frames,
                    "width": self.width, "height": self.height,
                    "started_host_local": self.started,
                    "first_frame_host_local": self.first_frame_at,
                    "last_frame_host_local": self.last_frame_at,
                    "elapsed_wall_s": round(time.monotonic() - self.started_mono, 3),
                    "timing_note": "AVI encodes constant FPS; real capture intervals may differ. This is camera video, not screen recording.",
                    "sha256": file_sha256(final), "camera_context": self.context,
                }
                sidecar = self.target_folder / (self.stem + ".json")
                try:
                    write_exclusive(sidecar, [(json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8")])
                except Exception as err:
                    raise OSError(f"Video saved at {final}, but metadata failed: {err}") from err
                return final, sidecar, meta
            finally:
                self._temp_dir.cleanup()

    def abort(self):
        """Best-effort release on unexpected exit; never overwrite any finished file."""
        with self._lock:
            if not self.closed:
                self._writer.release()
                self.closed = True
            self._temp_dir.cleanup()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
