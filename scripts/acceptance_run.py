r"""端到端验收脚本（真实 checkpoint / 真实图片）。

只做验收，不改模型、不改类别、不写训练数据。
输出 JSON：每个文件的 Top-3、重复推理时延、错误路径结果。

用法：
  .\.venv\Scripts\python.exe scripts\acceptance_run.py ^
      --checkpoint runs\food-v1.1.0-baseline\stage8-20261020-164500-40a26d\best.pt ^
      --classes configs\classes.json --images-dir "..\测试图片" --device cpu --repeats 5
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodai.inference_contract import ModelNotReadyError  # noqa: E402
from foodai.predictor import (InvalidImageError, PredictionAdapter,  # noqa: E402
                              as_contract_dict)


def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--classes", default=str(ROOT / "configs" / "classes.json"))
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "runs" / "acceptance" / "acceptance.json"))
    args = ap.parse_args(argv)

    import torch
    adapter = PredictionAdapter(checkpoint=args.checkpoint, classes_path=args.classes,
                                device=args.device)
    status = adapter.status()
    print(json.dumps(status.__dict__, ensure_ascii=False, indent=2))
    if status.state != "ready":
        print("BLOCKED: adapter 不是 ready，无法进行真实预测", file=sys.stderr)
        return 2

    images = list_images(Path(args.images_dir))
    print(f"图片数量：{len(images)}")

    # 预热一次（不计入统计）
    warmup = images[0]
    adapter.predict(str(warmup))

    records = []
    for image in images:
        latencies, last = [], None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            last = adapter.predict(str(image))
            latencies.append((time.perf_counter() - t0) * 1000.0)
        payload = as_contract_dict(last)
        rec = {
            "file": image.name,
            "top3": [
                {"rank": i + 1, "class_id": p["class_id"], "class_key": p["class_key"],
                 "label_zh": p["label_zh"], "score": p["score"]}
                for i, p in enumerate(payload["predictions"])
            ],
            "latency_ms_internal_median": round(last.latency_ms, 2),
            "latency_ms_wall_median": round(statistics.median(latencies), 2),
            "latency_ms_wall_p95": round(statistics.quantiles(latencies, n=20)[18], 2) if len(latencies) > 1 else round(latencies[0], 2),
            "repeats": args.repeats,
        }
        records.append(rec)
        top = rec["top3"][0]
        print(f"{image.name:45s} -> {top['label_zh']}({top['class_key']}) "
              f"{top['score']:.4f}  内部 {rec['latency_ms_internal_median']} ms  "
              f"墙钟 {rec['latency_ms_wall_median']} ms")

    # 错误路径 1：损坏图片
    error_cases = {}
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "corrupt.jpg"
        bad.write_text("this is not an image", encoding="utf-8")
        try:
            adapter.predict(str(bad))
            error_cases["corrupt_image"] = "FAIL: 没有报错"
        except InvalidImageError as exc:
            error_cases["corrupt_image"] = f"PASS: {exc}"
        except Exception as exc:
            error_cases["corrupt_image"] = f"UNEXPECTED {type(exc).__name__}: {exc}"

    # 错误路径 2：缺失 checkpoint
    missing = PredictionAdapter(checkpoint=None, classes_path=args.classes, device=args.device)
    error_cases["missing_checkpoint_state"] = missing.status().state
    try:
        missing.predict(str(warmup))
        error_cases["missing_checkpoint_predict"] = "FAIL: 没有报错"
    except ModelNotReadyError as exc:
        error_cases["missing_checkpoint_predict"] = f"PASS: {exc}"
    except Exception as exc:
        error_cases["missing_checkpoint_predict"] = f"UNEXPECTED {type(exc).__name__}: {exc}"

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(adapter.status().device),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": status.checkpoint_sha256,
        "model_id": status.model_id,
        "num_classes": status.num_classes,
        "input_size": status.input_size,
        "class_list_sha256": status.class_list_sha256,
        "images": records,
        "error_cases": error_cases,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")
    print(json.dumps(error_cases, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

