"""Stable interface for the app owner; real model adapter to be implemented by that owner."""
from __future__ import annotations
from dataclasses import dataclass


class ModelNotReadyError(RuntimeError):
    """No verified model/checkpoint is available yet."""


@dataclass(frozen=True)
class ClassPrediction:
    class_id: int
    class_key: str
    label_zh: str
    score: float


@dataclass(frozen=True)
class PredictionResult:
    model_id: str
    class_list_sha256: str
    predictions: tuple[ClassPrediction, ...]
    top1_id: int
    latency_ms: float
    status: str = "ok"


def predict_unconfigured(_image):
    """Explicit placeholder used until the application owner loads a real model."""
    raise ModelNotReadyError("模型尚未训练／未加载。此处没有模拟识别分数。")
