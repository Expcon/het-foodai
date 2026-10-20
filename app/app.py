r"""HeT FoodAI 本地演示：Gradio 上传 + Top-3 + 时延 + 明确状态。

启动（需要真实 checkpoint）：
    .\.venv\Scripts\python.exe app\app.py --checkpoint runs\<run_id>\best.pt

只检查状态、不启动界面：
    .\.venv\Scripts\python.exe app\app.py --check --checkpoint runs\<run_id>\best.pt

没有 checkpoint / 没有冻结 classes.json 时，明确定位为"模型未就绪"，
不显示任何模拟分数。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from foodai.app_logic import (adapter_banner, banner, image_banner, latency_markdown,  # noqa: E402
                             run_prediction)
from foodai.predictor import DEFAULT_CLASSES_PATH, PredictionAdapter  # noqa: E402


def build_demo(adapter: PredictionAdapter):
    import gradio as gr

    initial, enabled = adapter_banner(adapter.status())

    with gr.Blocks(title="HeT FoodAI · 食材识别", analytics_enabled=False) as demo:
        gr.Markdown("# HeT FoodAI · 本地食材识别演示\n"
                    "单张主食材图片分类（JPG / PNG）。本页面不展示任何评测准确率。")
        status_md = gr.Markdown(initial)

        with gr.Row():
            with gr.Column(scale=1):
                image = gr.File(type="filepath", file_types=[".jpg", ".jpeg", ".png"],
                                label="上传食材图片（JPG / PNG）")
                preview = gr.Image(label="图片预览", interactive=False, height=320)
                upload_note = gr.Markdown()
            with gr.Column(scale=1):
                predict_btn = gr.Button("开始识别", variant="primary", interactive=enabled)
                result_md = gr.Markdown()
                table = gr.Dataframe(
                    headers=["排名", "中文类别", "class_key", "score"],
                    datatype=["number", "str", "str", "str"],
                    interactive=False, value=[], label="Top-3",
                )
                latency = gr.Markdown()

        def on_upload(path):
            text, _ok = image_banner(path)
            return text, "", [], "", path if _ok else None

        def on_predict(path):
            if not path:
                return banner("warn", "请先上传图片", "支持 JPG / JPEG / PNG。"), [], ""
            return run_prediction(adapter, path)

        # Serialize image changes with prediction; lock both controls until the entire event finishes.
        image.change(on_upload, inputs=image, outputs=[upload_note, result_md, table, latency, preview],
                     concurrency_id="foodai-image", concurrency_limit=1)
        predict_btn.click(lambda: (gr.update(interactive=False), gr.update(interactive=False)),
                          outputs=[image, predict_btn], queue=False, api_name=False).then(
            on_predict, inputs=image, outputs=[result_md, table, latency], api_name="on_predict",
            concurrency_id="foodai-image", concurrency_limit=1,
        ).then(lambda: (gr.update(interactive=True), gr.update(interactive=enabled)),
               outputs=[image, predict_btn], queue=False, api_name=False)
    return demo


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="HeT FoodAI 本地 Gradio 演示")
    ap.add_argument("--checkpoint", default=None, help="训练产出的 best.pt；缺省则演示未就绪状态")
    ap.add_argument("--classes", default=str(DEFAULT_CLASSES_PATH),
                    help="冻结类别配置，默认 configs/classes.json")
    ap.add_argument("--expect-checkpoint-sha256", default=None,
                    help="checkpoint 文件 SHA256 的期望前缀（可选）")
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--check", action="store_true", help="只打印状态并以 JSON 退出，不启动界面")
    args = ap.parse_args(argv)

    adapter = PredictionAdapter(checkpoint=args.checkpoint, classes_path=args.classes,
                                expect_checkpoint_sha256=args.expect_checkpoint_sha256,
                                device=args.device)
    info = adapter.status()

    if args.check:
        print(json.dumps(info.__dict__, ensure_ascii=False, indent=2))
        return 0 if info.state == "ready" else 1

    build_demo(adapter).launch(server_name=args.host, server_port=args.port, share=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

