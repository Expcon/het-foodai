"""Real-checkpoint prediction adapter for docs/INFERENCE_CONTRACT.md (v0.1).

设计原则（对应 AGENTS.md 与 INFERENCE_CONTRACT 的硬要求）
--------------------------------------------------------
1. 未就绪就抛 `ModelNotReadyError`，**绝不返回数值**，也不做任何 fallback 打分。
2. 类别列表必须与冻结 `configs/classes.json` 的**有序前缀**逐字一致：
   8 类 checkpoint 可以配 12/16 类配置，但顺序、id、key、zh 一个字都不能差。
3. 预处理直接复用 `foodai.vision.get_transforms(image_size, train=False)`，
   与训练侧验证集是同一份代码，不做"看起来一样"的重写。
4. `latency_ms` 只统计 preprocess + forward + softmax，不含文件上传与 UI 渲染。
5. 本模块 import 时不依赖 torch：只有真正加载权重时才 import，
   这样"缺失 checkpoint / 契约不符 / 坏图片"这些状态可以纯 stdlib 测试。

本模块不计算任何 accuracy。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from foodai.contracts import ClassConfig, ContractError, class_list_hash, load_classes
from foodai.inference_contract import (
    ClassPrediction,
    ModelNotReadyError,
    PredictionResult,
    predict_unconfigured,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLASSES_PATH = ROOT / "configs" / "classes.json"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}
TOP_K = 3


def _contract_not_ready_message() -> str:
    """直接取契约函数里的原文，保证 UI 与 `predict_unconfigured` 一字不差。"""
    try:
        predict_unconfigured(None)
    except ModelNotReadyError as exc:
        return str(exc)
    raise AssertionError("predict_unconfigured 必须抛出 ModelNotReadyError")


NOT_READY_MESSAGE = _contract_not_ready_message()


class CheckpointContractError(RuntimeError):
    """checkpoint 的类别/hash/结构与冻结契约不一致。"""


class InvalidImageError(ValueError):
    """上传的文件不是可解码的 JPG/PNG。"""


@dataclass(frozen=True)
class AdapterInfo:
    state: str                 # ready | not_ready | contract_error
    message: str
    checkpoint: str | None = None
    model_id: str | None = None
    num_classes: int | None = None
    input_size: int | None = None
    class_list_sha256: str | None = None
    checkpoint_sha256: str | None = None
    device: str | None = None


# ----------------------------------------------------------------------
# 图片校验（不依赖 torch）
# ----------------------------------------------------------------------
def validate_image_file(path: str | Path) -> Path:
    """校验后缀、非空、可完整解码；任何一项不满足都抛 InvalidImageError。"""
    from PIL import Image, UnidentifiedImageError

    p = Path(path)
    if not p.exists():
        raise InvalidImageError(f"文件不存在：{p}")
    if not p.is_file():
        raise InvalidImageError(f"不是文件：{p}")
    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise InvalidImageError(
            f"不支持的格式 {p.suffix or '(无后缀)'}：只接受 JPG / JPEG / PNG"
        )
    if p.stat().st_size == 0:
        raise InvalidImageError(f"文件为空（0 字节）：{p.name}")
    try:
        with Image.open(p) as img:
            img.verify()          # 检测截断/损坏
        with Image.open(p) as img:
            img.load()
            if img.width <= 0 or img.height <= 0:
                raise InvalidImageError(f"图片尺寸非法：{p.name}")
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(
            f"无法解码为图片：{p.name}（文件可能已损坏或扩展名与实际格式不符）"
        ) from exc
    return p


def _load_rgb_image(source):
    from PIL import Image

    if isinstance(source, Image.Image):
        return source.convert("RGB")
    return Image.open(validate_image_file(source)).convert("RGB")


# ----------------------------------------------------------------------
# Adapter
# ----------------------------------------------------------------------
class PredictionAdapter:
    """加载真实 checkpoint 并按契约输出 Top-3。

    Parameters
    ----------
    checkpoint:
        训练脚本产出的 `best.pt`。为 None 时永远是 not-ready。
    classes_path:
        默认 `configs/classes.json`（必须 status=frozen）。
    expect_checkpoint_sha256:
        checkpoint 文件 SHA256 的期望前缀（大小写不敏感）。None 表示不校验。
    device:
        "cpu" / "cuda" / None(自动)。
    """

    def __init__(self, checkpoint=None, classes_path=None, expect_checkpoint_sha256=None,
                 device: str | None = None):
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.classes_path = Path(classes_path) if classes_path else DEFAULT_CLASSES_PATH
        self.expect_checkpoint_sha256 = (expect_checkpoint_sha256 or "").strip().lower() or None
        self.device = device
        self._info: AdapterInfo | None = None
        self._classes: ClassConfig | None = None
        self._metadata: dict[str, Any] | None = None
        self._transform = None
        self._model = None

    # ---------------- 状态 ----------------
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
            raise ModelNotReadyError(
                f"{NOT_READY_MESSAGE}\n\ncheckpoint 不存在：{self.checkpoint}\n"
                "请让模型 owner 提供 runs/<run_id>/best.pt，再用 --checkpoint 指定。"
            )
        if self.checkpoint.stat().st_size == 0:
            raise ModelNotReadyError(f"{NOT_READY_MESSAGE}\n\ncheckpoint 是空文件：{self.checkpoint}")

        classes = self._load_classes()
        digest = sha256_file(self.checkpoint)
        if self.expect_checkpoint_sha256 and not digest.startswith(self.expect_checkpoint_sha256):
            raise CheckpointContractError(
                f"checkpoint SHA256 前缀不符：期望 {self.expect_checkpoint_sha256}，实际 {digest[:16]}…\n"
                f"文件：{self.checkpoint}"
            )

        metadata = self._load_metadata()
        active = self._validate_metadata(metadata, classes)
        self._classes = classes
        self._metadata = metadata

        model_id = metadata.get("model_id") or (
            f"{metadata.get('model_name', 'unknown')}-stage{len(active)}-"
            f"{self.checkpoint.parent.name}"
        )
        return AdapterInfo(
            state="ready",
            message=f"模型已加载：{model_id}（{len(active)} 类）",
            checkpoint=str(self.checkpoint),
            model_id=str(model_id),
            num_classes=len(active),
            input_size=int(metadata["input_size"]),
            class_list_sha256=metadata["class_list_sha256"],
            checkpoint_sha256=digest,
            device=self.device or "auto",
        )

    # ---------------- 校验 ----------------
    def _load_classes(self) -> ClassConfig:
        try:
            return load_classes(self.classes_path, require_frozen=True)
        except ContractError as exc:
            raise ModelNotReadyError(
                f"{NOT_READY_MESSAGE}\n\n类别配置不可用：{self.classes_path}\n{exc}\n"
                "需要数据 owner 提交并冻结 configs/classes.json（status=frozen）后才能推理。"
            ) from exc

    def _load_metadata(self) -> dict[str, Any]:
        run_info = self.checkpoint.parent / "run_info.json"
        if run_info.is_file():
            try:
                return json.loads(run_info.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError) as exc:
                raise CheckpointContractError(f"run_info.json 解析失败：{exc}") from exc
        # 没有 run_info.json 才回落到读 checkpoint 内嵌 metadata（需要 torch）
        try:
            import torch
        except Exception as exc:  # pragma: no cover - 环境缺 torch 时
            raise CheckpointContractError(
                "找不到 run_info.json，且当前环境无法导入 torch 读取 checkpoint 内嵌 metadata。\n"
                "请在训练产物目录保留 run_info.json，或在含 torch 的项目 .venv 中运行。"
            ) from exc
        try:
            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise CheckpointContractError(f"checkpoint 无法读取：{exc}") from exc
        if not isinstance(payload, dict):
            raise CheckpointContractError("checkpoint 结构不是 dict")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise CheckpointContractError("checkpoint 缺少 metadata 字典")
        return metadata

    def _validate_metadata(self, metadata: dict[str, Any], classes: ClassConfig) -> list[dict]:
        if "class_list" not in metadata:
            raise CheckpointContractError("metadata 缺少 class_list")
        active = metadata["class_list"]
        if not isinstance(active, list) or not active:
            raise CheckpointContractError("metadata.class_list 必须是非空列表")
        if len(active) > len(classes.classes):
            raise CheckpointContractError(
                f"checkpoint 声明 {len(active)} 类，超过冻结配置的 {len(classes.classes)} 类"
            )
        for idx, item in enumerate(active):
            if not isinstance(item, dict):
                raise CheckpointContractError(f"class_list[{idx}] 必须是对象")
            expected = classes.classes[idx]
            got = (item.get("id"), item.get("key"), item.get("zh"))
            want = (expected.id, expected.key, expected.zh)
            if got != want:
                raise CheckpointContractError(
                    f"类别契约不符（第 {idx} 项）：checkpoint={got}，冻结配置={want}。"
                    "8/12/16 阶段必须共用同一有序类别前缀。"
                )
        expected_hash = class_list_hash(classes.classes[:len(active)])
        declared = metadata.get("class_list_sha256")
        if declared and declared != expected_hash:
            raise CheckpointContractError(
                f"class_list_sha256 不符：checkpoint={declared}，按冻结配置重算={expected_hash}"
            )
        if "input_size" not in metadata:
            raise CheckpointContractError("metadata 缺少 input_size")
        try:
            int(metadata["input_size"])
        except (TypeError, ValueError) as exc:
            raise CheckpointContractError(f"input_size 非法：{metadata['input_size']}") from exc
        return active

    # ---------------- 推理 ----------------
    def load(self) -> None:
        """按需加载权重（首次 predict 前调用）。torch 只在这里 import。"""
        if self._model is not None:
            return
        info = self.status()
        if info.state != "ready":
            raise ModelNotReadyError(info.message)

        import torch
        from foodai.vision import get_model, get_transforms

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(info.num_classes, pretrained=False)
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model_state") or payload.get("state_dict")
        if state is None:
            raise CheckpointContractError("checkpoint 里找不到 model_state/state_dict")
        model.load_state_dict(state)
        model.eval().to(device)

        self._model = model
        self._device = device
        self._transform = get_transforms(int(info.input_size), train=False)
        self._info = AdapterInfo(**{**info.__dict__, "device": device})

    def predict(self, image) -> PredictionResult:
        info = self.status()
        if info.state != "ready":
            raise ModelNotReadyError(info.message)
        self.load()

        import torch

        t0 = time.perf_counter()
        rgb = _load_rgb_image(image)
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        k = min(TOP_K, probs.numel())
        values, indices = torch.topk(probs, k)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        active = self._metadata["class_list"]
        predictions = tuple(
            ClassPrediction(
                class_id=int(active[int(i)]["id"]),
                class_key=str(active[int(i)]["key"]),
                label_zh=str(active[int(i)]["zh"]),
                score=float(v),
            )
            for v, i in zip(values.tolist(), indices.tolist())
        )
        return PredictionResult(
            model_id=str(info.model_id),
            class_list_sha256=str(info.class_list_sha256),
            predictions=predictions,
            top1_id=predictions[0].class_id,
            latency_ms=latency_ms,
            status="ok",
        )


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def as_contract_dict(result: PredictionResult) -> dict:
    """转成 docs/INFERENCE_CONTRACT.md 定义的 JSON 兼容结构（字段不多不少）。"""
    return {
        "model_id": result.model_id,
        "class_list_sha256": result.class_list_sha256,
        "predictions": [
            {"class_id": p.class_id, "class_key": p.class_key,
             "label_zh": p.label_zh, "score": float(p.score)}
            for p in result.predictions
        ],
        "top1_id": result.top1_id,
        "latency_ms": float(result.latency_ms),
        "status": result.status,
    }
