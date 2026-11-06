"""Passive camera enumeration: names with backend-specific OpenCV indices.

Windows implementation uses cv2-enumerate-cameras. Enumerating never opens a
VideoCapture. Do not assume DSHOW index N and MSMF index N are the same camera.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraChoice:
    index: int
    backend: str
    name: str
    path: str = ""
    vid: int | None = None
    pid: int | None = None

    @property
    def display(self):
        label = "DirectShow" if self.backend == "dshow" else "Media Foundation"
        return f"{self.name}  [{label} #{self.index}]"


def list_cameras(cv2_module=None):
    """Read system camera names; never open frames or guess unavailable indices."""
    if cv2_module is None:
        import cv2 as cv2_module
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except ImportError as exc:
        raise RuntimeError("缺少摄像头枚举组件，请用新版完整安装包；仍可使用手动编号") from exc
    found = []
    errors = []
    for backend, opencv_backend in (("dshow", cv2_module.CAP_DSHOW), ("msmf", cv2_module.CAP_MSMF)):
        try:
            for item in enumerate_cameras(opencv_backend):
                found.append(CameraChoice(int(item.index), backend, str(item.name or "未知设备"),
                                          str(item.path or ""), getattr(item, "vid", None), getattr(item, "pid", None)))
        except Exception as exc:
            errors.append(f"{backend}: {type(exc).__name__}: {exc}")
    if not found and errors:
        raise RuntimeError("摄像头名称枚举失败；" + "; ".join(errors))
    return sorted(found, key=lambda c: (c.name.lower(), c.backend, c.index))


def compatible_alternate(primary, choices):
    """Use a different backend only for a uniquely identified matching camera."""
    candidates = [c for c in choices if c.backend != primary.backend]
    if not candidates:
        return None
    # Best case: unique exact persistent device path in the two backends.
    if primary.path:
        same = [c for c in candidates if c.path and c.path.casefold() == primary.path.casefold()]
        if len(same) == 1:
            return same[0]
        if len(same) > 1:
            return None
    # A unique VID/PID can identify a camera only when no duplicates exist.
    if primary.vid is not None and primary.pid is not None:
        peers = [c for c in choices if c.vid == primary.vid and c.pid == primary.pid]
        matches = [c for c in peers if c.backend != primary.backend]
        if len(peers) == 2 and len(matches) == 1:
            return matches[0]
    # Name matching only when the name is unique PER backend; avoid two USB
    # devices of the same type silently being interchanged.
    same_name = [c for c in choices if c.name.strip().casefold() == primary.name.strip().casefold()]
    matches = [c for c in same_name if c.backend != primary.backend]
    if len(same_name) == 2 and len(matches) == 1 and primary.name.strip():
        return matches[0]
    return None


def manual_choice(index: int, backend: str):
    if type(index) is not int or index < 0 or index > 99 or backend not in {"dshow", "msmf"}:
        raise ValueError("请选择有效的相机编号和后端")
    return CameraChoice(index, backend, "手动编号（需目视确认）")
