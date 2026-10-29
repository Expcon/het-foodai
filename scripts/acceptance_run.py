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
import math
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


def percentile95(samples):
    """Nearest-rank empirical P95, including the one-sample case."""
    return sorted(samples)[math.ceil(.95 * len(samples)) - 1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--classes", default=str(ROOT / "configs" / "classes.json"))
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "runs" / "acceptance" / "acceptance.json"))
    args = ap.parse_args(argv)
    if args.repeats < 1:
        ap.error("--repeats must be a positive integer")
    folder = Path(args.images_dir)
    if not folder.is_dir():
        ap.error("--images-dir must be an existing directory")
    images = list_images(folder)
    if not images:
        ap.error("--images-dir contains no JPG/JPEG/PNG files")

    import torch
    adapter = PredictionAdapter(checkpoint=args.checkpoint, classes_path=args.classes,
                                device=args.device)
    status = adapter.status()
    print(json.dumps(status.__dict__, ensure_ascii=False, indent=2))
    if status.state != "ready":
        print("BLOCKED: adapter 不是 ready，无法进行真实预测", file=sys.stderr)
        return 2

    print(f"图片数量：{len(images)}")

    warmup = images[0]

    records = []
    for image in images:
        adapter.predict(str(image))  # One warm-up per image, excluded from both statistics.
        latencies, internal_latencies, last = [], [], None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            last = adapter.predict(str(image))
            latencies.append((time.perf_counter() - t0) * 1000.0)
            internal_latencies.append(float(last.latency_ms))
        payload = as_contract_dict(last)
        rec = {
            "file": image.name,
            "top3": [
                {"rank": i + 1, "class_id": p["class_id"], "class_key": p["class_key"],
                 "label_zh": p["label_zh"], "score": p["score"]}
                for i, p in enumerate(payload["predictions"])
            ],
            "latency_ms_internal_median": round(statistics.median(internal_latencies), 2),
            "latency_ms_wall_median": round(statistics.median(latencies), 2),
            "latency_ms_wall_p95": round(percentile95(latencies), 2),
            "latency_ms_internal_samples": internal_latencies,
            "latency_ms_wall_samples": latencies,
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
        "checkpoint": Path(args.checkpoint).name,
        "checkpoint_sha256": status.checkpoint_sha256,
        "model_id": status.model_id,
        "num_classes": status.num_classes,
        "input_size": status.input_size,
        "class_list_sha256": status.class_list_sha256,
        "measurement": {
            "statistics_version": 2,
            "warmup_per_image": 1,
            "repeats_per_image": args.repeats,
            "median_method": "statistics.median; even N averages the two middle samples",
            "p95_method": "nearest_rank: sorted_samples[ceil(0.95*N)-1]",
            "internal_scope": "preprocess, H2D, forward, softmax, Top-3, D2H; CUDA synchronized; excludes decode/validation and initial loading",
            "wall_scope": "perf_counter around adapter.predict; includes decode/validation, internal scope and Python result assembly; excludes initial loading, warmup, upload/render",
            "small_sample_note": "Descriptive samples, not a stable performance benchmark; P95 is max when N < 20",
        },
        "images": records,
        "error_cases": error_cases,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")
    print(json.dumps(error_cases, ensure_ascii=False, indent=2))
    passed = (error_cases["corrupt_image"].startswith("PASS:")
              and error_cases["missing_checkpoint_state"] == "not_ready"
              and error_cases["missing_checkpoint_predict"].startswith("PASS:"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

