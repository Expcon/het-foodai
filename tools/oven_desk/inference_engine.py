"""Explicit, fail-closed ONNX / MNN model testing for HeT OvenCapture.

Supports ONE IMAGE classification or YOLOv8/YOLO11 *raw* detection output only.
No assumption that an arbitrary .mnn/.onnx has this input/output contract.
Do not use this module to control an oven or select cooking parameters.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import threading

import cv2
import numpy as np
from PIL import Image, ImageOps


def class_keys_sha256(keys):
    """Match the export's ordered-key fingerprint without importing training code."""
    return hashlib.sha256(json.dumps(keys, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


OVEN_FORMAT = "het-oven-classifier/v1"
OVEN_KEYS = ['asparagus', 'bacon', 'broccoli', 'chicken_wing', 'croissant',
             'french_fries', 'potato', 'salmon', 'toast']
OVEN_CLASS_SHA = class_keys_sha256(OVEN_KEYS)
OVEN_ROUND2_FORMAT = "het-oven-classifier/v2"
OVEN_ROUND2_KEYS = OVEN_KEYS + ['corn', 'sausage', 'steak']
OVEN_ROUND2_CLASS_SHA = class_keys_sha256(OVEN_ROUND2_KEYS)
OVEN_MODES = {'oven_round1': (OVEN_FORMAT, OVEN_KEYS),
              'oven_round2': (OVEN_ROUND2_FORMAT, OVEN_ROUND2_KEYS)}
OVEN_FORMAT_MODES = {value[0]: mode for mode, value in OVEN_MODES.items()}
OVEN_PREPROCESS = dict(size=224, color='RGB', resize='bilinear-contain-center-pad',
                       pad=114, scale='1/255', layout='NCHW')


class ModelContractError(RuntimeError):
    pass


def _oven_spec(mode):
    if mode not in OVEN_MODES:
        raise ModelContractError("Unrecognized oven mode")
    return OVEN_MODES[mode]


def read_image_unicode(path, mode=None):
    """OpenCV imread may fail with Chinese path names on Windows."""
    if mode in OVEN_MODES:
        try:
            with Image.open(path) as source:
                rgb = np.asarray(ImageOps.exif_transpose(source).convert('RGB'))
            return np.ascontiguousarray(rgb[:, :, ::-1])
        except (OSError, ValueError) as exc:
            raise ModelContractError(f"Cannot decode image: {path}") from exc
    data = Path(path).read_bytes()
    out = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if out is None:
        raise ModelContractError(f"Cannot decode image: {path}")
    return out


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as exc:
        raise ModelContractError(f"Invalid class contract: {path}") from exc


def load_oven_contract(path, model_path, mode='oven_round1'):
    expected_format, keys = _oven_spec(mode)
    obj = _read_json(path)
    if not isinstance(obj, dict) or obj.get('format') != expected_format:
        raise ModelContractError(f"{mode} requires the exported {expected_format} classes.json")
    if obj.get('class_keys') != keys or obj.get('class_list_sha256') != class_keys_sha256(keys):
        raise ModelContractError(f"{mode} ordered class keys/hash mismatch")
    rows = obj.get('classes')
    if (not isinstance(rows, list) or len(rows) != len(keys) or
        any(not isinstance(row, dict) or type(row.get('id')) is not int or
            row['id'] != i or row.get('key') != keys[i] for i, row in enumerate(rows))):
        raise ModelContractError(f"{mode} class IDs/keys must match the explicit experimental order")
    if (obj.get('preprocess') != OVEN_PREPROCESS or obj.get('input_shape') != [1,3,224,224] or
        obj.get('output_shape') != [1,len(keys)] or obj.get('output_kind') != 'classification-probabilities' or
        obj.get('independent_validation') is not False):
        raise ModelContractError(f"{mode} preprocessing/output/experimental-scope mismatch")
    source_sha = obj.get('source_checkpoint_sha256')
    if not isinstance(source_sha, str) or not re.fullmatch('[0-9a-f]{64}', source_sha):
        raise ModelContractError("Source checkpoint SHA256 is required")
    models = obj.get('models')
    entry = models.get(Path(model_path).suffix.lower().lstrip('.')) if isinstance(models, dict) else None
    expected_sha = entry.get('sha256') if isinstance(entry, dict) else None
    if not isinstance(expected_sha, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha):
        raise ModelContractError("This model format has no approved SHA256 in classes.json")
    if file_sha256(model_path) != expected_sha:
        raise ModelContractError("Model SHA256 differs from classes.json; model not ready")
    return obj


def _reject_oven_in_generic_mode(model_path, labels_path, labels):
    for mode, (_, keys) in OVEN_MODES.items():
        if labels == keys:
            raise ModelContractError(f"Oven classes require {mode} mode; generic modes crop differently")
    paths = {Path(labels_path), Path(model_path).parent/'classes.json', Path(model_path).with_suffix('.json')}
    for path in paths:
        if path.suffix.lower() != '.json' or not path.is_file():
            continue
        try:
            obj = _read_json(path)
        except ModelContractError:
            if path == Path(labels_path): raise
            continue
        mode = OVEN_FORMAT_MODES.get(obj.get('format')) if isinstance(obj, dict) else None
        if mode is None:
            continue
        entries = obj.get('models', {})
        entry = entries.get(Path(model_path).suffix.lower().lstrip('.'), {}) if isinstance(entries, dict) else {}
        if path == Path(labels_path) or (isinstance(entry, dict) and entry.get('sha256') == file_sha256(model_path)):
            raise ModelContractError(f"This oven contract/model requires {mode} mode")


def load_labels(path):
    path = Path(path)
    if not path.is_file():
        raise ModelContractError("Class labels file is missing")
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(obj, dict) and "classes" in obj:
            obj = obj["classes"]
        if isinstance(obj, list):
            if all(isinstance(k, dict) for k in obj):
                for i, row in enumerate(obj):
                    if row.get("id", i) != i:
                        raise ModelContractError("Class IDs must be ordered 0..N-1")
                labels = [str(x.get("zh") or x.get("name") or x.get("key") or "") for x in obj]
            else:
                labels = [str(x) for x in obj]
        elif isinstance(obj, dict):
            try:
                labels = [str(obj[str(i)]) for i in range(len(obj))]
            except KeyError as exc:
                raise ModelContractError("JSON numeric class keys must be consecutive") from exc
        else:
            raise ModelContractError("JSON label file must be an array or classes object")
    else:
        labels = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        # Allow names in '0 carrot' format only with consecutive indices.
        import re
        m = [re.fullmatch(r"(\d+)[\s:]+(.+)", s) for s in labels]
        if m and all(x for x in m):
            if [int(x[1]) for x in m] != list(range(len(m))):
                raise ModelContractError("Class IDs in text file must be 0..N-1")
            labels = [x[2].strip() for x in m]
    if not labels or any(not x or x.isspace() for x in labels):
        raise ModelContractError("No valid nonblank class names")
    if len(labels) != len(set(labels)):
        raise ModelContractError("Duplicate class names")
    return labels


def _resize_shortest_center(frame_bgr, height, width, imagenet):
    if height != width:
        raise ModelContractError("Classification prototype expects square network input")
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    shortest = round(height * 1.14) if imagenet else height
    w, h = pil.size
    if min(w, h) <= 0:
        raise ModelContractError("Invalid image dimensions")
    ratio = shortest / min(w, h)
    rw, rh = max(1, round(w * ratio)), max(1, round(h * ratio))
    pil = pil.resize((rw, rh), resample=Image.Resampling.BILINEAR)
    x, y = (rw - width) // 2, (rh - height) // 2
    pil = pil.crop((x, y, x + width, y + height))
    tensor = np.asarray(pil, dtype=np.float32) / 255.0
    if imagenet:
        tensor = (tensor - np.array((0.485, 0.456, 0.406), dtype=np.float32)) / np.array((0.229, 0.224, 0.225), dtype=np.float32)
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])


def _letterbox(frame_bgr, height, width):
    h0, w0 = frame_bgr.shape[:2]
    if min(h0, w0) <= 0:
        raise ModelContractError("Invalid image size")
    gain = min(height / h0, width / w0)
    nw, nh = round(w0 * gain), round(h0 * gain)
    scaled = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = width - nw, height - nh
    left, top = dw // 2, dh // 2
    padded = cv2.copyMakeBorder(scaled, top, dh - top, left, dw - left, cv2.BORDER_CONSTANT, value=(114,114,114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...], dtype=np.float32) / 255.0
    return tensor, {"gain": gain, "pad_left": left, "pad_top": top, "orig_w": w0, "orig_h": h0}


def prepare_input(frame_bgr, mode: str, height: int, width: int):
    if mode in OVEN_MODES:
        if (height, width) != (224, 224):
            raise ModelContractError(f"{mode} oven input must be 224x224")
        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8 or
            frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or min(frame_bgr.shape[:2]) < 1):
            raise ModelContractError(f"{mode} input must be a nonempty uint8 BGR frame")
        try:
            rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            resized = ImageOps.contain(rgb, (224,224), Image.Resampling.BILINEAR)
            canvas = Image.new('RGB', (224,224), (114,114,114))
            canvas.paste(resized, ((224-resized.width)//2, (224-resized.height)//2))
        except ValueError as exc:
            raise ModelContractError("Image aspect ratio cannot be resized by the training transform") from exc
        tensor = np.asarray(canvas, dtype=np.float32) / 255.0
        return np.ascontiguousarray(tensor.transpose(2,0,1)[None]), None
    if mode == "efficientnet_imagenet":
        return _resize_shortest_center(frame_bgr, height, width, True), None
    if mode == "yolo_classify":
        return _resize_shortest_center(frame_bgr, height, width, False), None
    if mode == "yolo_detect":
        return _letterbox(frame_bgr, height, width)
    raise ModelContractError("Unrecognized preprocessing mode")


def parse_classification(output, labels):
    arr = np.asarray(output)
    if arr.ndim == 2 and arr.shape[0] == 1:
        scores = arr[0]
    elif arr.ndim == 1:
        scores = arr
    else:
        raise ModelContractError(f"Classification output must be [1,C] or [C], got {arr.shape}")
    if len(scores) != len(labels) or not np.isfinite(scores).all():
        raise ModelContractError("Output class count differs from labels or contains NaN/Inf")
    scores = scores.astype(np.float64)
    if scores.min() >= -1e-5 and scores.max() <= 1.00001 and abs(float(scores.sum()) - 1) < 1e-3:
        probs = np.clip(scores, 0, 1)
    else:
        exps = np.exp(scores - np.max(scores))
        probs = exps / exps.sum()
    indices = np.argsort(-probs)[:3]
    return [{"id": int(i), "label": labels[int(i)], "score": float(probs[i])} for i in indices]


def parse_oven_classification(output, labels, mode='oven_round1'):
    expected_format, keys = _oven_spec(mode)
    count = len(keys)
    arr = np.asarray(output)
    if arr.shape != (1,count) or arr.dtype != np.float32 or len(labels) != count:
        raise ModelContractError(f"{mode} output must be float32 [1,{count}] with matching labels, got {arr.dtype} {arr.shape}")
    if (not np.isfinite(arr).all() or np.any(arr < 0) or np.any(arr > 1) or
        abs(float(arr.astype(np.float64).sum()) - 1) > 1e-4):
        raise ModelContractError(f"{mode} output must be probabilities in [0,1] summing to 1; logits are rejected")
    scores = arr[0]
    indices = np.argsort(-scores, kind='stable')[:3]
    return {'task':'classification',
            'top3':[{'id':int(i), 'label':labels[int(i)], 'score':float(scores[i])} for i in indices],
            'probabilities':scores.tolist(), 'independent_validation':False,
            'mode':mode, 'contract_format':expected_format, 'class_list_sha256':class_keys_sha256(keys)}


def parse_yolo_detection(output, labels, context, conf=0.25, iou=0.45, limit=100):
    """YOLOv8/11 raw xywh+class scores only (no objectness, no built-in NMS)."""
    arr = np.asarray(output)
    nc = len(labels)
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ModelContractError("YOLO raw detection output must be batch=1 rank-3")
    if arr.shape[1] == nc + 4:
        pred = arr[0].T
    elif arr.shape[2] == nc + 4:
        pred = arr[0]
    else:
        raise ModelContractError(f"Expected raw YOLO output [1,{nc+4},N] or [1,N,{nc+4}], got {arr.shape}; check model version/NMS")
    if not np.isfinite(pred).all():
        raise ModelContractError("Invalid detection scores/boxes")
    if not context or not 0 < context["gain"]:
        raise ModelContractError("Missing letterbox parameters")
    candidate_boxes, candidate_scores, candidate_ids = [], [], []
    for row in pred:
        cls_id = int(np.argmax(row[4:]))
        score = float(row[4+cls_id])
        if score < conf:
            continue
        cx, cy, w, h = map(float, row[:4])
        x1 = (cx - w/2 - context["pad_left"]) / context["gain"]
        y1 = (cy - h/2 - context["pad_top"]) / context["gain"]
        x2 = (cx + w/2 - context["pad_left"]) / context["gain"]
        y2 = (cy + h/2 - context["pad_top"]) / context["gain"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(context["orig_w"], x2), min(context["orig_h"], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        candidate_boxes.append([float(x1), float(y1), float(x2-x1), float(y2-y1)])
        candidate_scores.append(score)
        candidate_ids.append(cls_id)
    # Class-aware NMS: never suppress overlapping boxes from different classes.
    keep = []
    for cls_id in sorted(set(candidate_ids)):
        indices = [i for i, v in enumerate(candidate_ids) if v == cls_id]
        idx_out = cv2.dnn.NMSBoxes([candidate_boxes[i] for i in indices], [candidate_scores[i] for i in indices], conf, iou)
        for local_index in np.array(idx_out).reshape(-1).tolist():
            keep.append(indices[int(local_index)])
    keep.sort(key=lambda i: -candidate_scores[i])
    return [{"id": candidate_ids[i], "label": labels[candidate_ids[i]],
             "score": candidate_scores[i], "xyxy": [int(round(candidate_boxes[i][0])), int(round(candidate_boxes[i][1])),
                 int(round(candidate_boxes[i][0]+candidate_boxes[i][2])), int(round(candidate_boxes[i][1]+candidate_boxes[i][3]))]}
            for i in keep[:limit]]


class ModelEngine:
    def __init__(self, model_path, labels_path, mode: str, input_size=224):
        self.path = Path(model_path)
        if not self.path.is_file():
            raise ModelContractError("Model file not found")
        if mode not in OVEN_MODES and mode not in {'yolo_classify', 'efficientnet_imagenet', 'yolo_detect'}:
            raise ModelContractError("Unrecognized preprocessing mode")
        self.contract = load_oven_contract(labels_path, self.path, mode) if mode in OVEN_MODES else None
        self.labels = list(self.contract['class_keys']) if self.contract else load_labels(labels_path)
        if not self.contract:
            _reject_oven_in_generic_mode(self.path, labels_path, self.labels)
        self.mode = mode
        self.lock = threading.Lock()
        self.session = None
        self.model = None
        self.input_name = None
        self.height, self.width = int(input_size), int(input_size)
        if self.contract and self.height != 224:
            raise ModelContractError(f"{mode} oven input must be 224x224")
        if self.height < 64 or self.height > 2048:
            raise ModelContractError("Input size outside supported range")
        if mode == "yolo_detect" and self.height % 32:
            raise ModelContractError("YOLO detection input size must be divisible by 32")
        ext = self.path.suffix.lower()
        if ext == ".onnx":
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ModelContractError("ONNX Runtime is not bundled in this build") from exc
            self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
            metadata = self.session.get_modelmeta().custom_metadata_map
            required_mode = OVEN_FORMAT_MODES.get(metadata.get('het_contract_format'))
            if required_mode is not None and mode != required_mode:
                raise ModelContractError(f"This ONNX model requires {required_mode} mode")
            inputs = self.session.get_inputs()
            if len(inputs) != 1:
                raise ModelContractError("Only single-input models are supported")
            self.input_name = inputs[0].name
            shape = inputs[0].shape
            if len(shape) != 4 or shape[1] != 3 or inputs[0].type != "tensor(float)":
                raise ModelContractError("Only float32 NCHW single-RGB-input ONNX models supported")
            if self.contract:
                outputs = self.session.get_outputs()
                if (shape != [1,3,224,224] or len(outputs) != 1 or
                    outputs[0].shape != [1,len(self.labels)] or outputs[0].type != 'tensor(float)'):
                    raise ModelContractError(f"{mode} ONNX must have static float32 [1,3,224,224] -> [1,{len(self.labels)}]")
            self.height = shape[2] if isinstance(shape[2], int) else self.height
            self.width = shape[3] if isinstance(shape[3], int) else self.width
        elif ext == ".mnn":
            try:
                import MNN
            except ImportError as exc:
                raise ModelContractError("MNN is not included in this build; use an ONNX model or install MNN build") from exc
            self.mnn = MNN
            self.runtime = MNN.nn.create_runtime_manager(({"backend":"CPU", "precision":"high", "numThread":2},))
            self.model = MNN.nn.load_module_from_file(str(self.path), [], [], runtime_manager=self.runtime, rearrange=True)
            info = self.model.get_info()
            if len(info.get("inputNames", [])) != 1 or len(info.get("outputNames", [])) != 1:
                raise ModelContractError("MNN model must have one input and one output")
            # Generic MNN Python bindings do not reliably expose static input dimensions;
            # operator must explicitly select network input size from model documentation.
        else:
            raise ModelContractError("Supported models: .onnx, .mnn; .pt belongs to the original PyTorch demo")
        if self.contract:
            # Readiness requires real execution. This synthetic zero tensor is only a contract check.
            parse_oven_classification(self._forward(np.zeros((1,3,224,224), np.float32)), self.labels, self.mode)
            expected = self.contract['models'][ext.lstrip('.')]['sha256']
            if file_sha256(self.path) != expected:
                raise ModelContractError("Model changed while loading; model not ready")

    def _forward(self, arr):
        if self.session is not None:
            outputs = self.session.run(None, {self.input_name:arr})
            if len(outputs) != 1:
                raise ModelContractError("Only single-output models supported")
            return outputs[0]
        var = self.mnn.expr.const(arr, arr.shape, self.mnn.expr.NCHW)
        outputs = self.model.onForward([var])
        if len(outputs) != 1:
            raise ModelContractError("MNN output count is not one")
        return np.asarray(outputs[0].read()).copy()

    def predict_file(self, path):
        return self.predict(read_image_unicode(path, mode=self.mode))

    def predict(self, frame_bgr):
        with self.lock:
            arr, context = prepare_input(frame_bgr, self.mode, self.height, self.width)
            output = self._forward(arr)
            if self.contract:
                return parse_oven_classification(output, self.labels, self.mode)
            if self.mode == "yolo_detect":
                return {"task":"detection", "detections": parse_yolo_detection(output, self.labels, context)}
            return {"task":"classification", "top3": parse_classification(output, self.labels)}
