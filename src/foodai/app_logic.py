"""UI 展示逻辑（纯函数，不 import gradio / torch）——便于无界面单测。

状态文案要求：发生了什么 / 为什么 / 怎么修。
全文件不出现"准确率/accuracy"，因为本应用没有评测结果，只有单张图的 softmax 分数。
"""
from __future__ import annotations

from typing import Iterable

from foodai.inference_contract import PredictionResult
from foodai.predictor import AdapterInfo, InvalidImageError

SCORE_NOTE = "score 是 softmax 类别分数，不是校准后的正确概率。"


def banner(kind: str, title: str, detail: str = "", hint: str = "") -> str:
    """kind: info | ok | warn | error"""
    icon = {"info": "ℹ️", "ok": "✅", "warn": "⚠️", "error": "⛔"}.get(kind, "ℹ️")
    text = f"{icon} **{title}**"
    if detail:
        text += f"\n\n{detail}"
    if hint:
        text += f"\n\n> {hint}"
    return text


def adapter_banner(info: AdapterInfo) -> tuple[str, bool]:
    """返回 (横幅 markdown, 预测按钮是否可用)。"""
    if info.state == "ready":
        return banner(
            "ok", "模型已就绪",
            f"`{info.model_id}` · {info.num_classes} 类 · 输入 {info.input_size}px · 设备 {info.device or 'auto'}",
            f"class_list_sha256: `{info.class_list_sha256}`\n\n"
            f"checkpoint_sha256: `{(info.checkpoint_sha256 or '')[:16]}…`",
        ), True
    if info.state == "contract_error":
        return banner("error", "checkpoint 与冻结契约不一致，已拒绝加载", info.message), False
    return banner("warn", "模型未就绪（未配置 / 未训练）", info.message), False


def image_banner(path) -> tuple[str, bool]:
    """上传时立即校验，坏图在点击识别之前就能看到原因。"""
    if not path:
        return banner("info", "等待上传", "支持 JPG / JPEG / PNG，建议单张、主体居中。"), False
    try:
        from foodai.predictor import validate_image_file
        validate_image_file(path)
    except InvalidImageError as exc:
        return banner("error", "输入不是有效图片", str(exc), "请重新导出为 JPG 或 PNG 后再试。"), False
    return banner("ok", "图片已载入", f"`{path}`"), True


def format_top3(result: PredictionResult) -> list[list]:
    """Top-3 表格行：排名 / 中文名 / class_key / score。"""
    rows: list[list] = []
    for rank, pred in enumerate(result.predictions[:3], start=1):
        rows.append([rank, pred.label_zh, pred.class_key, f"{pred.score:.4f}"])
    return rows


def latency_markdown(result: PredictionResult) -> str:
    return (f"**推理时延**：{result.latency_ms:.1f} ms  \n"
            f"<sub>本机测量，包含预处理、H2D、前向、softmax、Top-3及D2H，CUDA已同步；不含图片解码/校验、首次加载、上传和页面渲染。"
            f"桌面数字不代表板端性能。</sub>")


def run_prediction(adapter, image_path) -> tuple[str, list[list], str]:
    """点击识别时的完整流程；任何失败都返回可读状态，不返回分数。"""
    try:
        result = adapter.predict(image_path)
    except InvalidImageError as exc:
        return banner("error", "输入不是有效图片", str(exc)), [], ""
    except Exception as exc:  # ModelNotReadyError / CheckpointContractError / 其它
        name = type(exc).__name__
        return banner("error", "无法完成识别", f"`{name}`: {exc}",
                      "没有输出任何预测分数。"), [], ""

    return (
        banner("ok", f"识别完成：{result.predictions[0].label_zh}",
               f"model: `{result.model_id}`", f"{SCORE_NOTE}"),
        format_top3(result),
        latency_markdown(result),
    )


def assert_no_accuracy_claim(texts: Iterable[str]) -> None:
    """自检用：任何 UI 文案都不得暗示准确率。"""
    banned = ("准确率", "accuracy", "ACCURACY")
    for text in texts:
        for word in banned:
            if word in text:
                raise AssertionError(f"UI 文案不能出现 {word!r}: {text}")
