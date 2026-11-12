#!/usr/bin/env python3
"""HeT OvenCapture Desk v1.2.0-experimental — Windows x64 GUI.

No oven heating control, firmware access, automatic camera scanning or network calls.
Camera is explicitly selected by name/backend; manual-index fallback stays available.
"""
from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from capture_core import PhotoStore, VideoRecorder, frame_properties, fourcc_to_text, now_local, safe_stem, write_exclusive
from camera_devices import CameraChoice, list_cameras, compatible_alternate, manual_choice
from inference_engine import ModelContractError, ModelEngine, read_image_unicode, file_sha256, OVEN_MODES
from model_catalog import MANUAL_LABEL, read_catalog, resolve_profile

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    RootClass = TkinterDnD.Tk
    DND_AVAILABLE = True
except ImportError:
    RootClass = tk.Tk
    DND_AVAILABLE = False

APP = "HeT OvenCapture Desk"
VERSION = "1.2.2-r2-experimental"
OVEN_MODE_LABEL = "首轮烤箱分类（实验·9类）"
OVEN_ROUND2_MODE_LABEL = "二轮烤箱分类（实验·12类）"
BACKENDS = {"自动兼容（同设备）": "auto", "DirectShow (dshow)": "dshow", "Media Foundation (msmf)": "msmf"}
RESOLUTIONS = ("1920×1080", "1280×720", "640×480", "224×224", "自动 (不请求)")
PREVIEW_SIZES = ("自适应", "960×540", "640×360", "480×270", "224×224")
FORMAT_LABELS = ("自动", "MJPG", "YUY2")
DEFAULT_LABELS = ("未标注", "胡萝卜 carrot", "土豆 potato", "西兰花 broccoli", "南瓜 pumpkin", "西红柿 tomato", "白萝卜 white_radish", "黄瓜 cucumber", "甜椒 bell_pepper", "玉米 corn", "其他 / 未知")


def settings_path():
    base = os.environ.get("APPDATA")
    return Path(base) / "HeT-OvenCapture" / "settings.json" if base else Path.home() / ".het_oven_capture" / "settings.json"


def load_settings():
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(data):
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def create_events():
    return queue.Queue(maxsize=200)


def post_event(events, kind, payload):
    try:
        events.put_nowait((kind, payload))
    except queue.Full:
        # Do not block the camera worker on a full UI log queue.
        pass


class CaptureWorker(threading.Thread):
    """Owns all VideoCapture / VideoRecorder calls; GUI never touches cap."""
    def __init__(self, camera_options, resolution, fourcc, output_folder, context,
                 frame_queue, events, commands):
        super().__init__(daemon=True, name="OvenCamera")
        self.camera_options = list(camera_options)
        self.resolution, self.fourcc = resolution, fourcc
        self.output_folder, self.context = Path(output_folder), dict(context)
        self.frames, self.events, self.commands = frame_queue, events, commands
        self.stop_flag = threading.Event()
        self.backend_active = None
        self.capture = None
        self.recording = None
        self.burst = None
        self.save_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="OvenPhotoSave")
        self.save_slots = threading.BoundedSemaphore(100)

    def stop(self):
        self.stop_flag.set()

    def _open_one(self, camera):
        cap_type = cv2.CAP_DSHOW if camera.backend == "dshow" else cv2.CAP_MSMF
        cap = cv2.VideoCapture(camera.index, cap_type)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("device could not be opened")
        try:
            if self.fourcc != "自动":
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
            if self.resolution:
                w, h = self.resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            for i in range(36):
                if self.stop_flag.is_set():
                    raise RuntimeError("user cancelled")
                ok, image = cap.read()
                if ok and image is not None and image.ndim == 3 and image.shape[2] == 3:
                    return cap, image
                time.sleep(.08)
            raise RuntimeError("已打开设备，但连续约3秒未获得有效帧：可换后端、格式或采集分辨率")
        except Exception:
            cap.release()
            raise

    def _submit_photo(self, image, data):
        if image is None:
            post_event(self.events,"log","尚未收到图像帧，无法拍照")
            return False
        if not self.save_slots.acquire(blocking=False):
            post_event(self.events,"log","保存任务太多，请稍后重试；不会丢弃已提交的任务")
            return False
        frame = image.copy()
        output = Path(data.get("output", str(self.output_folder))).expanduser()
        ctx = dict(data.get("context", {}))
        ctx["driver_properties_at_capture"] = frame_properties(self.capture, cv2)
        ctx["captured_at_host_approx"] = now_local()
        include = bool(data.get("include_metadata", False))
        fmt = str(data.get("image_format", "PNG"))
        def task():
            try:
                photo, meta = PhotoStore(output).save(frame, ctx, include_metadata=include, image_format=fmt)
                post_event(self.events, "photo_saved", (str(photo), str(meta) if meta else None,
                                                       frame.shape[1],frame.shape[0]))
            except Exception as exc:
                post_event(self.events, "log", f"保存照片失败：{type(exc).__name__}: {exc}")
            finally:
                self.save_slots.release()
        self.save_pool.submit(task)
        return True

    def _tick_burst(self, frame):
        info = self.burst
        if info is None or time.monotonic() < info["due"]:
            return
        if not self._submit_photo(frame, info["data"]):
            self.burst = None
            post_event(self.events,"burst_end", "连拍中止：保存队列繁忙")
            return
        info["done"] += 1
        post_event(self.events,"burst_progress", (info["done"],info["count"]))
        if info["done"] >= info["count"]:
            self.burst = None
            post_event(self.events,"burst_end",f"已提交 {info['done']} 张到保存队列")
        else:
            info["due"] = time.monotonic() + info["interval"]

    def _handle_command(self, command, data, current_frame):
        if command == "properties":
            if self.backend_active != "dshow":
                post_event(self.events, "log", "相机属性页只支持 DirectShow；请先用 DShow 打开相机。")
            else:
                post_event(self.events, "log", "正在请求驱动属性窗口。完成后回到预览。")
                try:
                    self.capture.set(cv2.CAP_PROP_SETTINGS, 1)
                except Exception as exc:
                    post_event(self.events, "log", f"驱动属性页不可用: {exc}")
            post_event(self.events, "properties", frame_properties(self.capture, cv2))
        elif command == "inspect":
            post_event(self.events, "properties", frame_properties(self.capture, cv2))
        elif command == "photo":
            self._submit_photo(current_frame, data)
        elif command == "burst_start":
            count, interval = data.get("count"), data.get("interval")
            if self.burst is not None:
                post_event(self.events,"log","已有连拍任务，先停止再重新开始")
            elif (type(count) is not int or count < 2 or count > 50 or
                    not isinstance(interval, (int, float)) or not .2 <= interval <= 10):
                post_event(self.events,"log","连拍参数无效：2–50张、间隔0.2–10秒")
            else:
                self.burst = {"data":data, "count":count,"interval":float(interval),"done":0,"due":0.0}
                post_event(self.events,"burst_started",count)
        elif command == "burst_cancel":
            if self.burst is not None:
                done = self.burst["done"]
                self.burst = None
                post_event(self.events,"burst_end",f"已停止连拍，前 {done} 张已提交保存的照片仍会写完")
        elif command == "record_start":
            if self.recording is not None:
                post_event(self.events, "log", "录像已在进行")
                return
            try:
                if current_frame is None:
                    raise RuntimeError("还没有真实帧")
                self.recording = VideoRecorder(Path(data.get("output", str(self.output_folder))), current_frame,
                    float(data.get("fps", 15)), context=data.get("context", {}),
                    include_metadata=bool(data.get("include_metadata",False)))
                post_event(self.events, "recording", True)
                post_event(self.events, "log", "已开始录制原尺寸相机画面 (AVI/MJPEG，无音频)")
            except Exception as exc:
                post_event(self.events, "log", f"录像启动失败: {type(exc).__name__}: {exc}")
        elif command == "record_stop":
            self._finish_recording()

    def _finish_recording(self):
        if self.recording is None:
            return
        recorder = self.recording
        self.recording = None
        try:
            out = recorder.stop()
            if out:
                path, _, summary = out
                post_event(self.events, "log", f"录像保存: {path.name} | {summary['frame_count']} 帧 | {summary['elapsed_wall_s']}秒")
                post_event(self.events, "saved_path", str(path))
        except Exception as exc:
            post_event(self.events, "log", f"录像保存失败: {type(exc).__name__}: {exc}")
        finally:
            post_event(self.events, "recording", False)

    def run(self):
        cap = None
        try:
            first = None
            attempts = []
            for camera in self.camera_options:
                if self.stop_flag.is_set():
                    return
                post_event(self.events, "log", f"尝试 {camera.name}，后端 {camera.backend}，编号 {camera.index}")
                try:
                    cap, first = self._open_one(camera)
                    self.backend_active = camera.backend
                    self.selected_camera = camera
                    if camera != self.camera_options[0]:
                        post_event(self.events, "log", "已切换到同一设备已匹配的另一后端；仍请目视确认画面。")
                    break
                except Exception as exc:
                    attempts.append(f"{backend}: {exc}")
                    post_event(self.events, "log", f"{backend} 未成功: {exc}")
            if cap is None or first is None:
                raise RuntimeError("两个后端均无法取帧。" + "; ".join(attempts))
            self.capture = cap
            info = {
                "device_index": self.selected_camera.index, "device_name": self.selected_camera.name, "backend": self.backend_active,
                "actual_decoded_width": int(first.shape[1]), "actual_decoded_height": int(first.shape[0]),
                "driver_width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                "driver_height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                "reported_fourcc": fourcc_to_text(cap.get(cv2.CAP_PROP_FOURCC)),
                "driver_fps": cap.get(cv2.CAP_PROP_FPS),
                "properties_observed_only": frame_properties(cap, cv2),
            }
            post_event(self.events, "camera_ready", info)
            if float(np.mean(first)) < 4:
                post_event(self.events,"log","画面几乎全黑：请确认镜头、照明以及自动/手动曝光设置")
            fail_count = 0
            last = first
            while not self.stop_flag.is_set():
                for _ in range(12):
                    try:
                        command, data = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    self._handle_command(command, data, last)
                try:
                    ok, frame = cap.read()
                except cv2.error as exc:
                    ok, frame = False, None
                    post_event(self.events, "log", f"摄像头读帧异常: {exc}")
                if not ok or frame is None:
                    fail_count += 1
                    if fail_count >= 15:
                        raise RuntimeError("连续15次读取失败：检查 USB、权限、后端和独占应用")
                    time.sleep(.07)
                    continue
                if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                    raise RuntimeError("摄像头返回不支持的图像布局")
                fail_count = 0
                last = frame
                self._tick_burst(frame)
                if self.recording is not None:
                    try:
                        self.recording.write(frame)
                    except Exception as exc:
                        post_event(self.events, "log", f"录像错误，正在关闭文件: {exc}")
                        self._finish_recording()
                try:
                    if self.frames.full():
                        self.frames.get_nowait()
                    self.frames.put_nowait((frame, time.time()))
                except queue.Empty:
                    pass
                except queue.Full:
                    pass
        except Exception as exc:
            if not self.stop_flag.is_set():
                post_event(self.events, "camera_error", f"{type(exc).__name__}: {exc}")
        finally:
            if self.burst is not None:
                post_event(self.events,"burst_end", "摄像头结束；已提交的照片继续保存")
                self.burst = None
            self._finish_recording()
            self.save_pool.shutdown(wait=True)
            if cap is not None:
                cap.release()
            post_event(self.events, "camera_stopped", None)


def windows_camera_inventory():
    """Read names only, do NOT claim PnP order equals OpenCV capture index."""
    if sys.platform != "win32":
        return ["此功能只在 Windows 可用"]
    cmd = "Get-PnpDevice -Class Camera,Image -ErrorAction SilentlyContinue | Select-Object FriendlyName,Status | ConvertTo-Json -Compress"
    try:
        res = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", cmd],
            capture_output=True, text=True, timeout=15)
        if res.returncode:
            return [f"Windows 设备列表读取失败: {res.stderr.strip()[:180]}"]
        obj = json.loads(res.stdout.strip() or "[]")
        obj = obj if isinstance(obj, list) else [obj]
        return [f"{row.get('FriendlyName','(unnamed)')} / {row.get('Status','UNKNOWN')}" for row in obj if isinstance(row, dict)] or ["未枚举到设备"]
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return [f"读取 Windows 设备列表失败: {exc}"]


class DesktopApp:
    def __init__(self, root, catalog_path=None):
        self.root = root
        self.root.title(f"{APP} v{VERSION} · 食材图像采集与模型测试")
        self.root.geometry("1370x850")
        self.root.minsize(1060, 670)
        self.root.configure(bg="#F4F6FA")
        self.style = ttk.Style(root)
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#F4F6FA")
        self.style.configure("TLabel", background="#F4F6FA", foreground="#17283B", font=("Microsoft YaHei UI", 10))
        self.style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=(9, 7))
        self.style.configure("Accent.TButton", background="#086AC5", foreground="white", font=("Microsoft YaHei UI",10,"bold"))
        self.style.map("Accent.TButton", background=[("active", "#0552A0")])
        self.style.configure("TEntry", padding=5)
        self.style.configure("TCombobox", padding=4)
        self.style.configure("TNotebook", background="#F4F6FA")
        self.style.configure("TNotebook.Tab", font=("Microsoft YaHei UI", 10), padding=(12,7))
        self.settings = load_settings()
        app_dir=Path(sys.executable).parent if getattr(sys,'frozen',False) else Path(__file__).parent
        try:self.model_profiles=read_catalog(catalog_path or app_dir/'experimental_models.json')
        except ModelContractError as exc:
            self.model_profiles={}
            messagebox.showwarning('实验列表不可用',str(exc))
        self.last_model_profile=MANUAL_LABEL
        self.frames = queue.Queue(maxsize=2)
        self.events = create_events()
        self.commands = queue.Queue()
        self.worker = None
        self.latest_frame = None
        self.latest_capture_ts = 0
        self.camera_info = {}
        self.model = None
        self.model_loading = False
        self.inference_running = False
        self.camera_ready = False
        self.recording = False
        self.burst_running = False
        self.camera_choices = []
        self.named_choices = {}
        self.frame_count = 0
        self.last_preview_t = 0
        self.photo_image = None
        self.serial_port = None
        self.serial_thread = None
        self.serial_stop = threading.Event()
        self._closing = False
        self._vars()
        self._make_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._poll)
        if sys.platform=="win32":self.root.after(450,self.refresh_cameras)
        self._log("欢迎使用。先确认摄像头设备索引；不自动扫描或打开摄像头。")
        self._log("相机曝光与增益默认只读取；只有点击“驱动相机设置”才可能修改。")

    def _vars(self):
        self.device = tk.StringVar(value=str(self.settings.get("device_index", "")))
        self.camera_choice = tk.StringVar(value="请刷新设备列表，选择摄像头")
        old_backend=self.settings.get("backend", "自动兼容（同设备）")
        if old_backend == "自动兼容 (DShow→MSMF)":old_backend="自动兼容（同设备）"
        self.backend = tk.StringVar(value=old_backend)
        self.camformat = tk.StringVar(value=self.settings.get("format", "自动"))
        self.resolution = tk.StringVar(value=self.settings.get("resolution", "1920×1080"))
        self.preview_size = tk.StringVar(value="自适应")
        self.output = tk.StringVar(value=self.settings.get("output", str(Path.home() / "Pictures" / "HeT-OvenCapture")))
        self.session_id = tk.StringVar(value=f"session-{datetime.now():%Y%m%d}-{safe_stem('s')[-12:]}")
        self.camera_alias = tk.StringVar(value=self.settings.get("camera_alias", "oven-camera-A"))
        self.sample_id = tk.StringVar(value="")
        self.photo_mode = tk.StringVar(value=self.settings.get("photo_mode","快速拍照（不存JSON）"))
        self.photo_format = tk.StringVar(value=self.settings.get("photo_format","PNG"))
        self.burst_count = tk.StringVar(value="5")
        self.burst_interval = tk.StringVar(value="0.5")
        self.candidate = tk.StringVar(value="未标注")
        self.placement = tk.StringVar(value="center")
        self.lighting = tk.StringVar(value="oven-light")
        self.video_fps = tk.StringVar(value="15")
        self.camera_status = tk.StringVar(value="摄像头：未连接")
        self.photo_status = tk.StringVar(value="原尺寸取帧 · 图片不裁剪、不覆盖")
        self.model_file = tk.StringVar(value="")
        self.model_profile = tk.StringVar(value=MANUAL_LABEL)
        self.label_file = tk.StringVar(value="")
        self.model_task = tk.StringVar(value="YOLO 分类 (RGB/255)")
        self.network_size = tk.StringVar(value="224")
        self.model_state = tk.StringVar(value="模型：未加载")
        self.port_name = tk.StringVar(value="")
        self.baudrate = tk.StringVar(value="115200")

    def _make_ui(self):
        top = ttk.Frame(self.root, padding=(18, 14, 18, 8))
        top.pack(side="top", fill="x")
        ttk.Label(top, text="HeT OvenCapture", font=("Microsoft YaHei UI", 19, "bold")).pack(side="left")
        ttk.Label(top, text="   Windows 相机采集 / MNN·ONNX 模型实验", foreground="#516477").pack(side="left", pady=6)
        ttk.Label(top, text="不控制加热 · 原始解码帧 · 防覆盖", foreground="#227845").pack(side="right")
        main = ttk.Panedwindow(self.root, orient="horizontal")
        main.pack(fill="both", expand=True, padx=12, pady=(4,12))
        side = ttk.Frame(main, padding=(8,6,8,8), width=390)
        main.add(side, weight=0)
        right = ttk.Frame(main, padding=(8,6,8,8))
        main.add(right, weight=4)
        tabs = ttk.Notebook(side)
        tabs.pack(fill="both", expand=True)
        self.capture_tab = self._scrollable_tab(tabs, "相机 / 采集")
        self.model_tab = self._scrollable_tab(tabs, "模型 / 图集")
        self.serial_tab = self._scrollable_tab(tabs, "串口 (可选)")
        self._build_capture_tab()
        self._build_model_tab()
        self._build_serial_tab()
        ttk.Label(right, textvariable=self.camera_status, font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w", pady=(0,8))
        preview_frame = tk.Frame(right, bg="black", highlightthickness=1, highlightbackground="#B9C4CD")
        preview_frame.pack(fill="both", expand=True)
        self.preview = tk.Label(preview_frame, bg="black", fg="#DCE8F5", text="等待打开摄像头\n\n选择编号及后端 → 打开摄像头", font=("Microsoft YaHei UI", 13))
        self.preview.pack(fill="both", expand=True)
        quickbar=ttk.Frame(right)
        quickbar.pack(fill="x",pady=(8,0))
        ttk.Label(quickbar,text="实物编号").pack(side="left",padx=(0,6))
        ttk.Entry(quickbar,textvariable=self.sample_id,width=25).pack(side="left",padx=(0,10))
        ttk.Label(quickbar,text="例：carrot_001 → carrot_001_0001.png",foreground="#536579").pack(side="left")
        buttonbar=ttk.Frame(right)
        buttonbar.pack(fill="x", pady=(9,8))
        ttk.Button(buttonbar, text="保存照片 (S)", style="Accent.TButton", command=self.save_photo).pack(side="left", padx=(0,6))
        self.burst_btn = ttk.Button(buttonbar, text="开始连拍", command=self.toggle_burst)
        self.burst_btn.pack(side="left", padx=6)
        self.record_btn = ttk.Button(buttonbar, text="开始录像", command=self.toggle_recording)
        self.record_btn.pack(side="left", padx=6)
        ttk.Button(buttonbar, text="驱动相机设置", command=self.open_camera_settings).pack(side="left", padx=6)
        ttk.Button(buttonbar, text="打开保存目录", command=self.open_output_folder).pack(side="left", padx=6)
        ttk.Label(right, textvariable=self.photo_status, foreground="#536579").pack(anchor="w", pady=(0,8))
        ttk.Label(right, text="操作日志（只保存在本机）", font=("Microsoft YaHei UI",10,"bold")).pack(anchor="w")
        logframe = ttk.Frame(right)
        logframe.pack(fill="x", pady=(5,0))
        self.log_box = tk.Text(logframe, height=9, wrap="word", bg="#13263C", fg="#DAE8F4",
             insertbackground="white", borderwidth=0, font=("Consolas", 9), padx=10, pady=7, state="disabled")
        scroll=ttk.Scrollbar(logframe, orient="vertical", command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log_box.pack(side="left", fill="both", expand=True)
        self.root.bind_all("<KeyPress-s>", self._key_save)
        self.root.bind_all("<KeyPress-S>", self._key_save)

    def _scrollable_tab(self, notebook, label):
        panel=ttk.Frame(notebook)
        notebook.add(panel, text=label)
        canvas=tk.Canvas(panel, bg="#F4F6FA", highlightthickness=0)
        scroll=ttk.Scrollbar(panel, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner=ttk.Frame(canvas, padding=10)
        window=canvas.create_window((0,0), anchor="nw", window=inner)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        def wheel(e):
            if e.delta:
                canvas.yview_scroll(-int(e.delta / 120) or (-1 if e.delta > 0 else 1), "units")
        # Bind only inside this scrollable panel to avoid hijacking scroll events elsewhere.
        def track_children(widget):
            widget.bind("<MouseWheel>", wheel, add="+")
            for child in widget.winfo_children():
                track_children(child)
        def register_wheel(_=None):
            track_children(inner)
            canvas.bind("<MouseWheel>", wheel)
        panel.bind("<Enter>", register_wheel, add="+")
        return inner

    def _row(self, parent, label, widget_factory, width=12):
        row=ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, width=width).pack(side="left", anchor="center")
        widget=widget_factory(row)
        widget.pack(side="left", fill="x", expand=True)
        return widget

    def _entry(self, parent, label, variable, width=12):
        return self._row(parent, label, lambda row: ttk.Entry(row, textvariable=variable), width)

    def _combo(self, parent, label, variable, options, readonly=True, width=12):
        return self._row(parent,label,lambda row: ttk.Combobox(row, textvariable=variable, values=options,
                state="readonly" if readonly else "normal"),width)

    def _build_capture_tab(self):
        p=self.capture_tab
        ttk.Label(p, text="摄像头", font=("Microsoft YaHei UI",12,"bold")).pack(anchor="w", pady=(2,8))
        self.camera_picker = self._combo(p,"实际摄像头",self.camera_choice,[],readonly=True)
        self.camera_picker.bind("<<ComboboxSelected>>", self._camera_selected)
        ttk.Button(p,text="刷新摄像头列表（只读）",command=self.refresh_cameras).pack(fill="x",pady=(1,7))
        self._entry(p,"手动编号",self.device)
        ttk.Label(p,text="识别不到设备时才填手动编号；自动模式只匹配同一设备。",
                  foreground="#536579", wraplength=310).pack(anchor="w",pady=(0,5))
        self._combo(p,"连接后端",self.backend,list(BACKENDS))
        self._combo(p,"图像格式",self.camformat,FORMAT_LABELS)
        self._combo(p,"采集分辨率",self.resolution,RESOLUTIONS)
        self._combo(p,"显示尺寸",self.preview_size,PREVIEW_SIZES)
        r=ttk.Frame(p);r.pack(fill="x",pady=(8,9))
        self.connect_btn=ttk.Button(r,text="打开摄像头",style="Accent.TButton",command=self.toggle_camera)
        self.connect_btn.pack(side="left",fill="x",expand=True,padx=(0,3))
        ttk.Button(r,text="读取曝光 / 增益",command=self.inspect_camera).pack(side="left",fill="x",expand=True,padx=(3,0))
        ttk.Separator(p).pack(fill="x",pady=9)
        ttk.Label(p,text="文件保存",font=("Microsoft YaHei UI",12,"bold")).pack(anchor="w",pady=(2,7))
        out_row=ttk.Frame(p); out_row.pack(fill="x",pady=3)
        ttk.Entry(out_row,textvariable=self.output).pack(side="left",fill="x",expand=True)
        ttk.Button(out_row,text="选择…",command=self.choose_output).pack(side="left",padx=(4,0))
        self._entry(p,"摄像头别名",self.camera_alias)
        self._entry(p,"拍摄会话",self.session_id)
        self._entry(p,"实物编号",self.sample_id)
        ttk.Label(p,text="照片命名：实物编号_0001.png，自动递增、不覆盖。",
                  foreground="#536579",wraplength=310).pack(anchor="w")
        self._combo(p,"拍照模式",self.photo_mode,["快速拍照（不存JSON）","数据集采集（保存JSON）"])
        self._combo(p,"照片格式",self.photo_format,["PNG","JPG"])
        self._combo(p,"连拍数量",self.burst_count,["2","5","10","20","50"],readonly=False)
        self._combo(p,"连拍间隔/秒",self.burst_interval,["0.25","0.5","1","2"],readonly=False)
        self._combo(p,"候选食材",self.candidate,DEFAULT_LABELS,readonly=False)
        self._entry(p,"摆放位置",self.placement)
        self._entry(p,"光照条件",self.lighting)
        self._combo(p,"录像帧率",self.video_fps,["10","15","20","25","30"],readonly=True)
        ttk.Label(p,text="照片、录像不会覆盖旧文件。\n保存完整解码帧；数据集标签仍需人工复核。",foreground="#566B7A",justify="left").pack(anchor="w",pady=(10,0))

    def _build_model_tab(self):
        p=self.model_tab
        ttk.Label(p,text="模型测试（可选）",font=("Microsoft YaHei UI",12,"bold")).pack(anchor="w",pady=(2,9))
        ttk.Label(p,text="支持 ONNX/MNN 的指定输出契约。\nPyTorch .pt 请继续使用原 Gradio 项目。",foreground="#536579",justify="left").pack(anchor="w",pady=(0,9))
        picker=self._combo(p,'实验模型',self.model_profile,[MANUAL_LABEL,*self.model_profiles])
        picker.bind('<<ComboboxSelected>>',self.select_model_profile)
        self._file_row(p,"模型文件",self.model_file,[('Model files','*.onnx *.mnn'),('All files','*.*')])
        self._file_row(p,"类别文件",self.label_file,[('Labels','*.txt *.json'),('All files','*.*')])
        self._combo(p,"模型类型",self.model_task,[OVEN_MODE_LABEL,OVEN_ROUND2_MODE_LABEL,"YOLO 分类 (RGB/255)","EfficientNet 分类 (ImageNet)","YOLOv8/11 检测 (原始输出)"])
        self._combo(p,"网络输入",self.network_size,["224","320","416","512","640"],readonly=False)
        ttk.Label(p,textvariable=self.model_state,foreground="#235884",wraplength=315).pack(anchor="w",pady=(9,7))
        ttk.Button(p,text="载入真实模型",style="Accent.TButton",command=self.load_model).pack(fill="x",pady=(0,5))
        ttk.Button(p,text="识别当前摄像头画面",command=self.infer_current).pack(fill="x",pady=5)
        ttk.Button(p,text="图集测试：选择 JPG/PNG 文件夹",command=self.test_gallery).pack(fill="x",pady=5)
        ttk.Label(p,text="烤箱模型：按导出版本选择首轮9类或二轮12类、224 和随附 classes.json。\n烤箱模型尚无独立验证，只显示实验预测。\n图集测试输出预测与 CSV；没有真值不显示准确率。\n检测仅支持 YOLOv8/11 raw xywh，无法解析即拒绝。",foreground="#536579",justify="left",wraplength=315).pack(anchor="w",pady=(12,4))

    def _file_row(self,parent,label,var,types):
        row=ttk.Frame(parent);row.pack(fill="x",pady=5)
        ttk.Label(row,text=label,width=9).pack(side="left")
        field=ttk.Entry(row,textvariable=var)
        field.pack(side="left",fill="x",expand=True)
        def browse():
            name=filedialog.askopenfilename(filetypes=types)
            if name:var.set(name)
        ttk.Button(row,text="浏览",command=browse).pack(side="left",padx=(3,0))
        if DND_AVAILABLE:
            try:
                field.drop_target_register(DND_FILES)
                def on_drop(e):
                    paths = self.root.tk.splitlist(e.data)
                    if paths: var.set(paths[0])
                field.dnd_bind('<<Drop>>', on_drop)
            except Exception:
                pass

    def _build_serial_tab(self):
        p=self.serial_tab
        ttk.Label(p,text="串口（仅连接 / 只读监测）",font=("Microsoft YaHei UI",12,"bold")).pack(anchor="w",pady=(2,12))
        self.serial_combo=self._combo(p,"串口",self.port_name,[],readonly=False)
        ttk.Button(p,text="刷新串口列表",command=self.refresh_serials).pack(fill="x",pady=5)
        self._combo(p,"波特率",self.baudrate,["9600","19200","38400","57600","115200","230400"],readonly=False)
        self.serial_btn=ttk.Button(p,text="打开串口（不发送控制命令）",command=self.toggle_serial)
        self.serial_btn.pack(fill="x",pady=10)
        ttk.Label(p,text="此功能不执行任何加热、继电器、风机或固件命令。\n未获得企业通信协议与授权前，不应连接\n可能会被串口打开动作影响的控制端口。",foreground="#9A5B23",justify="left").pack(anchor="w",pady=9)
        self.refresh_serials()

    def _log(self,msg):
        stamp=datetime.now().strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end",f"[{stamp}] {msg}\n")
        if int(self.log_box.index("end-1c").split('.')[0])>260:
            self.log_box.delete("1.0","60.0")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _current_context(self):
        label=self.candidate.get().strip()
        label_key=label.split()[-1] if label and label!="未标注" else None
        return {"camera_alias":self.camera_alias.get().strip(),
                "session_id":self.session_id.get().strip(),
                "physical_sample_id":self.sample_id.get().strip() or None,
                "candidate_class_key_unreviewed":label_key,
                "placement":self.placement.get().strip(),"lighting":self.lighting.get().strip(),
                "device_index":self.camera_info.get("device_index"),
                "device_name":self.camera_info.get("device_name"),
                "backend":self.camera_info.get("backend"),
                "fourcc_reported":self.camera_info.get("reported_fourcc"),
                "driver_properties_last_observed":self.camera_info.get("properties_observed_only",{}),
                "human_label_confirmed":False}

    def refresh_cameras(self):
        if self.worker and self.worker.is_alive():
            self._log("请先关闭摄像头，再刷新设备列表")
            return
        self._log("正在读取摄像头名称与各后端编号；不会打开任何视频流")
        def task():
            try:
                items = list_cameras(cv2)
                post_event(self.events,"camera_list",(items,None))
            except Exception as exc:
                post_event(self.events,"camera_list",([],str(exc)))
        threading.Thread(target=task,daemon=True).start()

    def _camera_selected(self,_event=None):
        selected=self.named_choices.get(self.camera_choice.get())
        if selected is None:
            return
        self.device.set(str(selected.index))
        self.backend.set("DirectShow (dshow)" if selected.backend=="dshow" else "Media Foundation (msmf)")
        self._log(f"已选 {selected.name} / {selected.backend} #{selected.index}，请确认预览身份")

    def choose_output(self):
        directory=filedialog.askdirectory(initialdir=str(Path(self.output.get()).expanduser().parent))
        if directory:self.output.set(directory)

    def _get_camera_inputs(self):
        text = self.camera_choice.get().strip()
        if self.camera_choices and text not in self.named_choices and text != "手动编号（下面填写）":
            raise ValueError("请先从实际摄像头列表选择设备；需要手动编号请选择“手动编号”")
        named = self.named_choices.get(text)
        desired = BACKENDS.get(self.backend.get())
        if desired is None:
            raise ValueError("请选择摄像头后端")
        if named is not None:
            if desired == "auto":
                chosen = [named]
                alternate = compatible_alternate(named,self.camera_choices)
                if alternate is not None:
                    chosen.append(alternate)
            elif desired == named.backend:
                chosen = [named]
            else:
                alternate=compatible_alternate(named,self.camera_choices)
                if alternate is None or alternate.backend != desired:
                    raise ValueError("无法确认同一相机在该后端的设备编号；请从设备列表中选对应后端")
                chosen=[alternate]
        else:
            try:
                index=int(self.device.get().strip())
                if not 0<=index<100: raise ValueError
            except ValueError:
                raise ValueError("请先从实际摄像头列表选择设备，或输入手动编号 0..99")
            if desired == "auto":
                raise ValueError("手动编号不能自动切换后端（不同后端编号不一定相同）；请明确选择 DSHOW 或 MSMF")
            chosen=[manual_choice(index,desired)]
        res=self.resolution.get()
        if res=="自动 (不请求)":
            shape=None
        else:
            parts=res.replace("×","x").split("x")
            shape=(int(parts[0]),int(parts[1]))
        output=Path(self.output.get().strip()).expanduser()
        if not self.output.get().strip():
            raise ValueError("请选择图片保存目录")
        output.mkdir(parents=True,exist_ok=True)
        fmt=self.camformat.get()
        if fmt not in FORMAT_LABELS:
            raise ValueError("无效的摄像头格式")
        return chosen,shape,fmt,output

    def toggle_camera(self):
        if self.worker and self.worker.is_alive():
            self._log("请求关闭摄像头…")
            self.worker.stop()
            self.connect_btn.configure(state="disabled")
            return
        try:
            chosen,shape,fmt,out=self._get_camera_inputs()
        except (ValueError,OSError) as exc:
            messagebox.showwarning("摄像头配置",str(exc));return
        self.latest_frame=None
        self.camera_ready=False
        self.recording=False
        self.frame_count=0
        self.camera_status.set(f"正在打开 {chosen[0].name} / {chosen[0].backend} #{chosen[0].index} …")
        self.connect_btn.configure(text="停止摄像头")
        self._save_user_settings()
        self.commands=queue.Queue()
        self.worker=CaptureWorker(chosen,shape,fmt,out,self._current_context(),self.frames,self.events,self.commands)
        self.worker.start()

    def inspect_camera(self):
        if self.worker and self.worker.is_alive() and self.camera_ready:
            self.commands.put(("inspect",{}))
        else:
            self._log("摄像头未就绪，无法读取曝光属性")

    def open_camera_settings(self):
        if self.worker and self.worker.is_alive() and self.camera_ready:
            self.commands.put(("properties",{}))
        else:
            messagebox.showwarning("摄像头", "请先打开摄像头并等待取帧成功")

    def save_photo(self):
        if not self.camera_ready or not self.worker or not self.worker.is_alive():
            self._log("未取到摄像头真实画面，不能保存")
            return
        if not self.output.get().strip():
            self._log("请选择保存路径")
            return
        if self.photo_mode.get().startswith("数据集采集") and not self.sample_id.get().strip():
            messagebox.showwarning("实物编号","数据集采集模式请先填写实物编号")
            return
        self.commands.put(("photo", self._save_payload()))

    def _save_payload(self):
        return {"context": self._current_context(), "output": self.output.get().strip(),
                "include_metadata": self.photo_mode.get().startswith("数据集采集"),
                "image_format":self.photo_format.get()}

    def toggle_burst(self):
        if self.burst_running:
            self.commands.put(("burst_cancel",{}))
            return
        if not self.camera_ready or not self.worker or not self.worker.is_alive():
            self._log("请先打开摄像头再连拍")
            return
        if self.photo_mode.get().startswith("数据集采集") and not self.sample_id.get().strip():
            messagebox.showwarning("实物编号","数据集采集模式请先填写实物编号")
            return
        try:
            count=int(self.burst_count.get().strip())
            interval=float(self.burst_interval.get().strip())
            if not 2<=count<=50 or not .2<=interval<=10:
                raise ValueError
        except ValueError:
            messagebox.showwarning("连拍设置","连拍数量 2–50，间隔 0.2–10 秒")
            return
        self.commands.put(("burst_start",{**self._save_payload(),"count":count,"interval":interval}))

    def _key_save(self,event):
        focus = self.root.focus_get()
        if isinstance(focus,(tk.Entry,tk.Text,ttk.Entry,ttk.Combobox,tk.Spinbox)):
            return
        self.save_photo()

    def toggle_recording(self):
        if not self.worker or not self.worker.is_alive() or not self.camera_ready:
            self._log("摄像头未就绪，无法录像")
            return
        if self.recording:
            self.commands.put(("record_stop",{}))
            self.record_btn.configure(state="disabled")
        else:
            try:
                fps=float(self.video_fps.get())
                if not 1<=fps<=60:raise ValueError
            except ValueError:
                messagebox.showerror("录像", "无效录像帧率")
                return
            self.commands.put(("record_start",{"fps":fps,"context":self._current_context(),
                "include_metadata":self.photo_mode.get().startswith("数据集采集"),"output":self.output.get().strip()}))
            self.record_btn.configure(state="disabled")

    def open_output_folder(self):
        path=Path(self.output.get().strip()).expanduser()
        if not path.is_dir():
            messagebox.showwarning("保存目录", "文件夹还不存在。先选择合法保存路径。")
            return
        if sys.platform == "win32":
            os.startfile(str(path))
        else:
            messagebox.showinfo("目录",str(path))

    def _model_mode(self):
        return {OVEN_MODE_LABEL:"oven_round1", OVEN_ROUND2_MODE_LABEL:"oven_round2", "YOLO 分类 (RGB/255)":"yolo_classify",
                "EfficientNet 分类 (ImageNet)":"efficientnet_imagenet",
                "YOLOv8/11 检测 (原始输出)":"yolo_detect"}[self.model_task.get()]

    def select_model_profile(self,event=None):
        if self.model_loading or self.inference_running:
            self.model_profile.set(self.last_model_profile)
            self._log('模型任务进行中，请完成后再切换模型')
            return
        label=self.model_profile.get()
        if label==MANUAL_LABEL:
            self.last_model_profile=label
            return
        self.model=None
        self.model_state.set('模型：未就绪，请载入并验证所选实验模型')
        try:
            profile=resolve_profile(self.model_profiles[label])
            self.model_file.set(str(profile['model']));self.label_file.set(str(profile['classes']))
            self.model_task.set(OVEN_ROUND2_MODE_LABEL);self.network_size.set('224')
            self.last_model_profile=label
        except (KeyError,OSError,ModelContractError) as exc:
            self.model_profile.set(self.last_model_profile)
            self._log('实验模型选择失败：'+str(exc))

    def load_model(self):
        if self.model_loading or self.inference_running:
            self._log("模型任务进行中，请完成后再切换模型")
            return
        self.model=None
        self.model_state.set("模型：未就绪，等待验证新模型")
        path,labels=self.model_file.get().strip(),self.label_file.get().strip()
        if not path or not labels:
            messagebox.showinfo("模型", "请选择模型文件 (.onnx 或 .mnn) 和类别文件 (.txt 或 .json)")
            return
        try:
            mode=self._model_mode()
            input_size=int(self.network_size.get().strip())
            expected_checkpoint=None
            selected=getattr(self,'model_profile',None)
            if selected is not None and selected.get()!=MANUAL_LABEL:
                profile=resolve_profile(self.model_profiles[selected.get()])
                expected_checkpoint=profile['checkpoint_sha256']
                if Path(path).resolve()!=profile['model'].resolve() or Path(labels).resolve()!=profile['classes'].resolve() or mode!='oven_round2' or input_size!=224:
                    raise ModelContractError('所选实验模型路径/模式已改变；请重新选择，或切回手动选择')
        except (ValueError, KeyError, OSError, ModelContractError) as exc:
            messagebox.showwarning("模型", "无效的网络输入尺寸、模式或实验模型："+str(exc))
            return
        self.model_loading=True
        self.model_state.set("模型：正在验证输入输出及加载权重…")
        def task():
            try:
                engine=ModelEngine(path,labels,mode,input_size)
                if expected_checkpoint is not None and engine.contract['source_checkpoint_sha256']!=expected_checkpoint:
                    raise ModelContractError('Loaded checkpoint differs from selected experiment')
                post_event(self.events,"model_ready",engine)
            except Exception as exc:
                post_event(self.events,"model_error",f"{type(exc).__name__}: {exc}")
        threading.Thread(target=task,daemon=True).start()

    def _start_predict(self,frame,tag="摄像头当前帧"):
        engine=self.model
        if engine is None or self.model_loading:
            messagebox.showwarning("模型未就绪", "先载入真实 ONNX/MNN 模型和匹配的类别文件。不会产生模拟预测。")
            return
        if self.inference_running:
            self._log("预测任务进行中，请稍后")
            return
        self.inference_running=True
        def task():
            try:
                t0=time.perf_counter()
                result=engine.predict(frame)
                result["latency_wall_ms"] = round((time.perf_counter()-t0)*1000,2)
                post_event(self.events,"predict",(tag,result))
            except Exception as exc:
                post_event(self.events,"predict_error",f"{tag}: {type(exc).__name__}: {exc}")
        threading.Thread(target=task,daemon=True).start()

    def infer_current(self):
        if self.latest_frame is None:
            self._log("当前没有真实帧，无法预测")
            return
        self._start_predict(self.latest_frame.copy())

    def test_gallery(self):
        engine=self.model
        if engine is None or self.model_loading:
            messagebox.showwarning("模型未就绪", "请先载入模型和类别文件，否则不能运行图集测试。")
            return
        if self.inference_running:
            return
        folder=filedialog.askdirectory(title="选择已有 JPG/PNG 图集目录")
        if not folder:return
        if self.model_loading or self.inference_running or self.model is not engine:
            self._log("模型状态已变化，请重新启动图集测试")
            return
        files=[p for p in sorted(Path(folder).iterdir()) if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png"}]
        if not files:
            messagebox.showwarning("图集测试", "所选文件夹内没有 JPG/PNG")
            return
        if len(files)>500:
            messagebox.showwarning("图集测试", f"已发现 {len(files)} 张图片。为避免不受控的大批量处理，本次最多处理前 500 张。")
            files=files[:500]
        self.inference_running=True
        self._log(f"图集测试开始：{len(files)} 张，推理结果仅作为预测，不等于真实准确率")
        target = Path(self.output.get().strip()).expanduser()
        def task():
            rows=[]
            for index,path in enumerate(files):
                try:
                    result=engine.predict_file(path)
                    if result["task"]=="classification":
                        top=result["top3"][0]
                        row={"image":str(path),"status":"OK","task":"classification","predicted_label":top["label"],"score":top["score"],"detections":"","full_result":json.dumps(result,ensure_ascii=False)}
                    else:
                        objects=result["detections"]
                        row={"image":str(path),"status":"OK","task":"detection","predicted_label":objects[0]["label"] if objects else "", "score":objects[0]["score"] if objects else "", "detections":len(objects),"full_result":json.dumps(result,ensure_ascii=False)}
                except Exception as exc:
                    row={"image":str(path),"status":str(exc),"task":"","predicted_label":"","score":"","detections":"","full_result":""}
                rows.append(row)
                if (index+1)%20==0:
                    post_event(self.events,"log",f"图集完成 {index+1}/{len(files)}")
            try:
                target.mkdir(parents=True,exist_ok=True)
                csvfile=target/(safe_stem("gallery_predictions")+".csv")
                import io
                sio=io.StringIO()
                writer=csv.DictWriter(sio,fieldnames=["image","status","task","predicted_label","score","detections","full_result"])
                writer.writeheader();writer.writerows(rows)
                write_exclusive(csvfile,[sio.getvalue().encode("utf-8-sig")])
                post_event(self.events,"gallery_done",(len(rows),sum(r["status"]=="OK" for r in rows),str(csvfile)))
            except Exception as exc:
                post_event(self.events,"predict_error",f"图集结果保存失败: {exc}")
        threading.Thread(target=task,daemon=True).start()

    def refresh_serials(self):
        try:
            from serial.tools import list_ports
            ports = [p.device for p in list_ports.comports()]
        except ImportError:
            ports=[]
            if hasattr(self,"log_box"):self._log("本构建未包含 pyserial，串口功能不可用")
        self.serial_combo.configure(values=ports)
        if ports and not self.port_name.get(): self.port_name.set(ports[0])

    def toggle_serial(self):
        if self.serial_port is not None:
            self.serial_stop.set()
            try:self.serial_port.close()
            except Exception:pass
            self.serial_port=None
            self.serial_btn.configure(text="打开串口（不发送控制命令）")
            self._log("串口已关闭")
            return
        if not messagebox.askyesno("确认串口连接", "仅允许在已确认用途和安全范围的端口上连接。\n程序只读，不发送任何控制命令。\n继续？"):
            return
        try:
            import serial
            name=self.port_name.get().strip()
            baud=int(self.baudrate.get())
            if not name or baud<1:raise ValueError("必须选择真实串口与有效波特率")
            ser=serial.Serial(port=None,baudrate=baud,timeout=.15)
            ser.port=name
            ser.dtr=False;ser.rts=False
            ser.open()
            self.serial_port=ser
            self.serial_stop.clear()
            self.serial_btn.configure(text="关闭串口")
            self._log(f"串口已连接 {name} / {baud}；只读，不发送指令")
            def reader():
                while not self.serial_stop.is_set():
                    try:
                        data=ser.read(128)
                        if data:
                            post_event(self.events,"log","串口接收(hex): "+data[:80].hex(" "))
                    except Exception as exc:
                        post_event(self.events,"log",f"串口读取异常: {exc}")
                        break
            self.serial_thread=threading.Thread(target=reader,daemon=True)
            self.serial_thread.start()
        except Exception as exc:
            messagebox.showerror("串口打开失败",str(exc))

    def _poll(self):
        if self._closing:return
        for _ in range(50):
            try:
                kind,payload=self.events.get_nowait()
            except queue.Empty:
                break
            if kind=="log":self._log(str(payload))
            elif kind=="camera_list":
                entries,err=payload
                self.camera_choices=entries
                self.named_choices={x.display:x for x in entries}
                labels=["手动编号（下面填写）"]+list(self.named_choices)
                self.camera_picker.configure(values=labels)
                if err:
                    self._log("设备枚举不可用："+err+"；可使用手动编号")
                elif not entries:
                    self._log("未发现可枚举的设备；可使用手动编号和指定后端")
                else:
                    for item in entries:self._log("摄像头："+item.display)
                    preferred = self.settings.get("camera_name","")
                    preferred_backend=BACKENDS.get(self.settings.get("backend",""))
                    matches=[x for x in entries if x.name==preferred and (preferred_backend in {None,"auto",x.backend})]
                    if len(matches)==1:
                        self.camera_choice.set(matches[0].display)
                        self._camera_selected()
                    else:
                        self.camera_choice.set("手动编号（下面填写）" if not entries else "请从上方列表选择摄像头")
            elif kind=="burst_started":
                self.burst_running=True
                self.burst_btn.configure(text="停止连拍")
                self._log(f"开始连拍：{payload} 张，拍摄将随真实相机帧进行")
            elif kind=="burst_progress":
                self.photo_status.set(f"连拍 {payload[0]}/{payload[1]}（已排队保存）")
            elif kind=="burst_end":
                self.burst_running=False
                self.burst_btn.configure(text="开始连拍")
                self._log(str(payload))
            elif kind=="camera_ready":
                self.camera_info=payload
                self.camera_ready=True
                self.connect_btn.configure(text="停止摄像头",state="normal")
                self.camera_status.set(f"已连接 {payload['device_name']} · {payload['backend']} #{payload['device_index']} · {payload['actual_decoded_width']}×{payload['actual_decoded_height']}")
                self._log(f"实际取帧 {payload['actual_decoded_width']}×{payload['actual_decoded_height']}，FOURCC={payload['reported_fourcc']}，驱动 FPS={payload['driver_fps']}(仅报告值)")
            elif kind=="camera_error":
                self._log("摄像头错误: "+str(payload))
                self.camera_status.set("摄像头错误：检查日志并尝试其他后端")
            elif kind=="camera_stopped":
                self.camera_ready=False
                self.burst_running=False
                self.burst_btn.configure(text="开始连拍")
                self.recording=False
                self.connect_btn.configure(text="打开摄像头",state="normal")
                self.record_btn.configure(text="开始录像",state="normal")
                if not self.camera_status.get().startswith("摄像头错误"):
                    self.camera_status.set("摄像头：已停止")
            elif kind=="properties":
                self.camera_info["properties_observed_only"]=payload
                self._log("相机属性（0/-1 可能表示驱动不支持）："+json.dumps(payload,ensure_ascii=False))
            elif kind=="recording":
                self.recording=bool(payload)
                self.record_btn.configure(text="停止录像" if self.recording else "开始录像",state="normal")
            elif kind=="photo_saved":
                name,meta,w,h=payload
                self._log(f"已保存 {w}×{h}：{Path(name).name}"+(f"；元数据：{Path(meta).name}" if meta else "（无JSON）"))
                self.photo_status.set("已保存："+Path(name).name)
            elif kind=="saved_path":
                self.photo_status.set("录像已保存："+Path(payload).name)
            elif kind=="model_ready":
                self.model=payload
                self.model_loading=False
                scope=" · 实验预测 / 无独立验证" if payload.mode in OVEN_MODES else ""
                self.model_state.set(f"模型：已加载 {payload.path.name} · {len(payload.labels)} 类 · {payload.height}×{payload.width}{scope}")
                self._log("模型加载成功。输出必须匹配类别数量和选择的解析协议。")
            elif kind=="model_error":
                self.model=None
                self.model_loading=False
                self.model_state.set("模型：加载失败；详见日志")
                self._log("模型加载失败: "+str(payload))
            elif kind=="predict":
                tag,result=payload
                self.inference_running=False
                if result["task"]=="classification":
                    summary=" | ".join(f"{r['label']} {r['score']:.1%}" for r in result["top3"])
                else:
                    det=result["detections"]
                    summary="、".join(f"{r['label']} {r['score']:.1%} {r['xyxy']}" for r in det[:10]) or "无检测框"
                self._log(f"真实模型预测 {tag}：{summary}；耗时 {result['latency_wall_ms']}ms（不含模型加载）")
            elif kind=="predict_error":
                self.inference_running=False
                self._log("预测失败: "+str(payload))
            elif kind=="gallery_done":
                self.inference_running=False
                total,ok,path=payload
                self._log(f"图集测试结束 {ok}/{total} 张执行成功，CSV={path}（没有真值不计算准确率）")
        latest=None
        while True:
            try:latest=self.frames.get_nowait()
            except queue.Empty:break
        if latest:
            frame,ts=latest
            self.latest_frame=frame
            self.latest_capture_ts=ts
            self.frame_count+=1
        if self.latest_frame is not None and time.monotonic()-self.last_preview_t>0.065:
            self._draw_preview(self.latest_frame)
            self.last_preview_t=time.monotonic()
        self.root.after(45,self._poll)

    def _draw_preview(self,frame):
        try:
            total_w,total_h=self.preview.winfo_width()-8,self.preview.winfo_height()-8
            if total_w<60 or total_h<60:
                return
            mode=self.preview_size.get()
            if mode!="自适应":
                w,h=map(int,mode.replace("×","x").split("x"))
                total_w,total_h=min(total_w,w),min(total_h,h)
            h,w=frame.shape[:2]
            scale=min(total_w/w,total_h/h,1.5)
            new_w,new_h=max(1,int(w*scale)),max(1,int(h*scale))
            scaled=cv2.resize(frame,(new_w,new_h),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
            rgb=cv2.cvtColor(scaled,cv2.COLOR_BGR2RGB)
            photo=ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.photo_image=photo
            self.preview.configure(image=photo,text="")
        except Exception as exc:
            self._log("预览失败: "+str(exc))

    def _save_user_settings(self):
        try:
            save_settings({"device_index":self.device.get(),"backend":self.backend.get(),
                "format":self.camformat.get(),"resolution":self.resolution.get(),
                "output":self.output.get(),"camera_alias":self.camera_alias.get(),
                "photo_mode":self.photo_mode.get(),"photo_format":self.photo_format.get(),
                "camera_name":self.camera_info.get("device_name",self.settings.get("camera_name",""))})
        except OSError as exc:
            self._log(f"无法保存本地设置: {exc}")

    def _on_close(self):
        if self.recording:
            if not messagebox.askyesno("录像进行中", "正在录像。退出时将尝试写入并关闭 AVI 文件，确定退出？"):
                return
        self._closing=True
        if self.worker and self.worker.is_alive():
            self.worker.stop()
            self.worker.join(timeout=20)
        if self.serial_port is not None:
            self.serial_stop.set()
            try:self.serial_port.close()
            except Exception:pass
        self._save_user_settings()
        self.root.destroy()


def healthcheck(report_file):
    """Non-camera packaging smoke test; no PnP access, not a teammate compatibility proof."""
    import hashlib
    import tempfile
    items = {}
    try:
        import numpy
        import PIL
        with tempfile.TemporaryDirectory(prefix="het-healthcheck-") as td:
            path = Path(td) / "中文路径" / "测试"
            frame = np.zeros((32, 64, 3), dtype=np.uint8)
            frame[:, :, 1] = 82
            photo, sidecar = PhotoStore(path).save(frame, {"healthcheck":True,"physical_sample_id":"test_001"})
            decoded = read_image_unicode(photo)
            assert np.array_equal(decoded, frame)
            assert json.loads(sidecar.read_text("utf-8"))["sha256"] == hashlib.sha256(photo.read_bytes()).hexdigest()
            assert photo.name == "test_001_0001.png"
            second, sidecar2 = PhotoStore(path).save(frame, {"physical_sample_id":"test_001"}, include_metadata=False)
            assert second.name == "test_001_0002.png" and sidecar2 is None
            items["safe_png_unicode_sha"] = "PASS"
            items["numbering_optional_json"] = "PASS"
        items["opencv_version"] = cv2.__version__
        items["numpy_version"] = numpy.__version__
        items["pillow_version"] = PIL.__version__
        items["tkinter"] = "AVAILABLE"
        try:
            import cv2_enumerate_cameras
            items["camera_enumeration"] = "IMPORT_OK"
        except Exception as exc:
            items["camera_enumeration"] = "UNAVAILABLE: " + str(exc)
        try:
            import onnxruntime
            items["onnxruntime"] = onnxruntime.__version__
        except Exception as exc:
            items["onnxruntime"] = "UNAVAILABLE: " + str(exc)
        try:
            import MNN
            items["MNN"] = "IMPORT_OK"
        except Exception as exc:
            items["MNN"] = "UNAVAILABLE: " + str(exc)
        status = "PASS" if items["safe_png_unicode_sha"] == "PASS" else "FAIL"
    except Exception as exc:
        status = "FAIL"
        items["exception"] = f"{type(exc).__name__}: {exc}"
    report = {"status": status, "checks":items,
              "note":"Packaged-runtime smoke only. Camera/serial/video and model validation require physical Windows machines."}
    target=Path(report_file).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if sys.stdout is not None:
        print(json.dumps(report, ensure_ascii=False))
    return 0 if status=="PASS" else 1


def model_selfcheck(model_path, classes_path, image_path, report_file, mode='oven_round1'):
    """Load a real export and predict a file without constructing Tk or opening hardware."""
    report={'status':'FAIL', 'model_ready':False, 'mode':mode,
            'independent_validation':False, 'camera_opened':False,
            'note':'Experimental file inference only; no independent validation or camera test.'}
    try:
        if mode not in OVEN_MODES:
            raise ModelContractError("Model selfcheck requires an explicit oven mode")
        engine=ModelEngine(model_path, classes_path, mode, 224)
        started=time.perf_counter()
        result=engine.predict_file(image_path)
        elapsed_ms=round((time.perf_counter()-started)*1000,2)
        report.update(status='PASS', model_ready=True, result=result,
                      model_sha256=file_sha256(model_path), image_sha256=file_sha256(image_path),
                      latency_wall_ms=elapsed_ms,
                      timing_scope='Image decode + preprocessing + CPU inference + output validation; excludes model load and file hashes')
    except Exception as exc:
        report['error']=f"{type(exc).__name__}: {exc}"
    target=Path(report_file).expanduser()
    # A report path must not replace a selected model, its contract, or its image.
    if target.resolve() in {Path(p).expanduser().resolve() for p in (model_path,classes_path,image_path)}:
        if sys.stderr is not None: print('Report must not overwrite an input file', file=sys.stderr)
        return 1
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    except OSError as exc:
        if sys.stderr is not None: print(f'Cannot write report: {exc}', file=sys.stderr)
        return 1
    if sys.stdout is not None:
        print(json.dumps(report, ensure_ascii=False))
    return 0 if report['status']=='PASS' else 1


def main():
    # Never import unknown vendor SDK or change oven power/heating controls.
    if len(sys.argv)>=2 and sys.argv[1]=="--healthcheck":
        if len(sys.argv)!=3:
            raise SystemExit("usage: HeT-OvenCapture --healthcheck <report.json>")
        raise SystemExit(healthcheck(sys.argv[2]))
    if len(sys.argv)>=2 and sys.argv[1]=="--model-selfcheck":
        import argparse
        parser=argparse.ArgumentParser(description="No-camera real-model check for an experimental oven export")
        parser.add_argument('--model-selfcheck', action='store_true')
        parser.add_argument('--mode', choices=list(OVEN_MODES), default='oven_round1', help='Oven export version; default preserves round-1 usage')
        parser.add_argument('--model', required=True, help='Exported ONNX/MNN model; .pt is not accepted')
        parser.add_argument('--classes', required=True, help='Matching het-oven-classifier/v1 or v2 classes.json')
        parser.add_argument('--image', required=True, help='Real JPG/PNG image to predict')
        parser.add_argument('--out', required=True, help='JSON report (PASS or FAIL); exit code 0 or 1')
        args=parser.parse_args()
        raise SystemExit(model_selfcheck(args.model,args.classes,args.image,args.out,mode=args.mode))
    import argparse
    parser=argparse.ArgumentParser(description='Local capture and approved experimental models')
    parser.add_argument('--model-catalog',type=Path,help='Optional local model list; does not auto-load or change defaults')
    args=parser.parse_args()
    root=RootClass()
    DesktopApp(root,catalog_path=args.model_catalog)
    root.mainloop()


if __name__=="__main__":
    main()
