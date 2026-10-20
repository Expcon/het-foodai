"""Checkpoint-backed adapter using the training loader and validation transform.

Readiness requires strict weight loading and a finite full-model forward pass.
Timing excludes image decoding/validation and initial model loading; includes
preprocessing, H2D, forward, softmax, Top-3 and D2H, with CUDA synchronization.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from foodai.contracts import ClassConfig, ContractError, hash_file, load_classes
from foodai.inference_contract import (
    ClassPrediction, ModelNotReadyError, PredictionResult, predict_unconfigured,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLASSES_PATH = ROOT / "configs" / "classes.json"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}
TOP_K = 3


def _contract_not_ready_message() -> str:
    try:
        predict_unconfigured(None)
    except ModelNotReadyError as exc:
        return str(exc)
    raise AssertionError("predict_unconfigured 必须抛出 ModelNotReadyError")


NOT_READY_MESSAGE = _contract_not_ready_message()


class CheckpointContractError(RuntimeError):
    """Checkpoint weights, metadata or execution validation failed."""


class InvalidImageError(ValueError):
    """Input is not a fully decodable JPEG/PNG."""


@dataclass(frozen=True)
class AdapterInfo:
    state: str
    message: str
    checkpoint: str | None = None
    model_id: str | None = None
    num_classes: int | None = None
    input_size: int | None = None
    class_list_sha256: str | None = None
    checkpoint_sha256: str | None = None
    device: str | None = None


def validate_image_file(path: str | Path) -> Path:
    from PIL import Image, UnidentifiedImageError
    p = Path(path)
    if not p.is_file():
        raise InvalidImageError(f"文件不存在或不是文件：{p}")
    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise InvalidImageError(f"不支持的格式 {p.suffix or '(无后缀)'}：只接受 JPG / JPEG / PNG")
    if p.stat().st_size == 0:
        raise InvalidImageError(f"文件为空（0 字节）：{p.name}")
    try:
        with Image.open(p) as img:
            # MPO is a JPEG container in 38 delivered photos; use its first frame as training does.
            if img.format not in {"JPEG", "PNG", "MPO"}:
                raise InvalidImageError(f"不支持的图片编码：{img.format}；只接受 JPEG / PNG")
            img.verify()
        with Image.open(p) as img:
            img.load()
            if img.width <= 0 or img.height <= 0:
                raise InvalidImageError(f"图片尺寸非法：{p.name}")
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(f"无法解码为图片：{p.name}（文件可能已损坏）") from exc
    return p


def _load_rgb_image(source):
    from PIL import Image
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    with Image.open(validate_image_file(source)) as image:
        return image.convert("RGB")


class PredictionAdapter:
    """Load only trusted project checkpoints; absent model remains explicitly not ready."""
    def __init__(self, checkpoint=None, classes_path=None, expect_checkpoint_sha256=None,
                 device: str | None = None):
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.classes_path = Path(classes_path) if classes_path else DEFAULT_CLASSES_PATH
        self.expect_checkpoint_sha256 = (expect_checkpoint_sha256 or "").strip().lower() or None
        self.device = device
        self._info: AdapterInfo | None = None
        self._metadata = None
        self._transform = None
        self._model = None
        self._device = None

    def status(self) -> AdapterInfo:
        if self._info is None:
            self._info = self._inspect()
        return self._info

    @property
    def is_ready(self) -> bool:
        return self.status().state == "ready"

    def _inspect(self) -> AdapterInfo:
        try:
            return self._inspect_ready()
        except ModelNotReadyError as exc:
            return AdapterInfo(state="not_ready", message=str(exc),
                               checkpoint=str(self.checkpoint) if self.checkpoint else None)
        except CheckpointContractError as exc:
            return AdapterInfo(state="contract_error", message=str(exc),
                               checkpoint=str(self.checkpoint) if self.checkpoint else None)

    def _inspect_ready(self) -> AdapterInfo:
        if self.checkpoint is None:
            raise ModelNotReadyError(NOT_READY_MESSAGE)
        if not self.checkpoint.is_file():
            raise ModelNotReadyError(f"{NOT_READY_MESSAGE}\ncheckpoint 不存在：{self.checkpoint}\n"
                                     "请让模型 owner 提供 best.pt，再用 --checkpoint 指定。")
        if self.checkpoint.stat().st_size == 0:
            raise ModelNotReadyError(f"{NOT_READY_MESSAGE}\ncheckpoint 是空文件：{self.checkpoint}")
        classes = self._load_classes()
        try:
            import torch
            from foodai.checkpoint import load_classifier_checkpoint
            from foodai.vision import get_transforms
            digest = hash_file(self.checkpoint)
            if self.expect_checkpoint_sha256 and not digest.startswith(self.expect_checkpoint_sha256):
                raise CheckpointContractError("checkpoint SHA256 前缀不符")
            model, metadata = load_classifier_checkpoint(self.checkpoint, classes)
            device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            model = model.to(device).eval()
            transform = get_transforms(metadata["input_size"], train=False)
            with torch.inference_mode():
                logits = model(torch.zeros(1, 3, metadata["input_size"], metadata["input_size"], device=device))
            if tuple(logits.shape) != (1, metadata["stage"]) or not torch.isfinite(logits).all().item():
                raise CheckpointContractError("模型前向验证失败：输出形状不符或含非有限数值")
        except (ContractError, ImportError, OSError, RuntimeError, ValueError) as exc:
            raise CheckpointContractError(f"模型加载或验证失败：{exc}") from exc
        # Publish state only after weights, preprocessing and device execution all succeed.
        self._model, self._metadata, self._transform, self._device = model, metadata, transform, device
        model_id = str(metadata.get("model_id") or f"efficientnet_v2_s-stage{metadata['stage']}-{self.checkpoint.parent.name}")
        return AdapterInfo(state="ready", message=f"模型已加载并验证：{model_id}", checkpoint=str(self.checkpoint),
                           model_id=model_id, num_classes=metadata["stage"], input_size=metadata["input_size"],
                           class_list_sha256=metadata["class_list_sha256"], checkpoint_sha256=digest, device=str(device))

    def _load_classes(self) -> ClassConfig:
        try:
            return load_classes(self.classes_path, require_frozen=True)
        except (ContractError, OSError) as exc:
            raise ModelNotReadyError(f"{NOT_READY_MESSAGE}\n类别配置不可用：{self.classes_path}\n{exc}\n"
                                     "需要三名成员共同确认并冻结 configs/classes.json 后才能推理。") from exc

    def load(self) -> None:
        info = self.status()
        if info.state != "ready":
            raise ModelNotReadyError(info.message)

    def predict(self, image) -> PredictionResult:
        self.load()
        import torch
        rgb = _load_rgb_image(image)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        start = time.perf_counter()
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            if not torch.isfinite(probs).all().item():
                raise RuntimeError("模型输出含非有限分数；未生成预测")
            values, indices = probs.topk(min(TOP_K, probs.numel()))
        values, indices = values.cpu().tolist(), indices.cpu().tolist()
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        latency_ms = (time.perf_counter() - start) * 1000.0
        active = self._metadata["class_list"]
        predictions = tuple(ClassPrediction(active[i]["id"], active[i]["key"], active[i]["zh"], float(v))
                            for v, i in zip(values, indices))
        info = self.status()
        return PredictionResult(model_id=info.model_id, class_list_sha256=info.class_list_sha256,
                                predictions=predictions, top1_id=predictions[0].class_id,
                                latency_ms=latency_ms, status="ok")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Compatibility name for existing adapter callers; use the shared file hash."""
    return hash_file(path)


def as_contract_dict(result: PredictionResult) -> dict:
    return {
        "model_id": result.model_id, "class_list_sha256": result.class_list_sha256,
        "predictions": [{"class_id": p.class_id, "class_key": p.class_key,
                         "label_zh": p.label_zh, "score": float(p.score)} for p in result.predictions],
        "top1_id": result.top1_id, "latency_ms": float(result.latency_ms), "status": result.status,
    }
