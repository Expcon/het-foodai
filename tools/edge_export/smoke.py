"""Official YOLO11 nano tooling specimens; never a HeT model or accuracy evaluation.

Run with runs/phase2-edge/.venv. Only --allow-downloads permits the four fixed
official URLs; SDK workers are offline and cannot auto-install packages.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .offline_guard import install, isolated_environment

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "yolo11n-cls.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt",
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    "bus.jpg": "https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.230/ultralytics/assets/bus.jpg",
    "zidane.jpg": "https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.230/ultralytics/assets/zidane.jpg",
}
PINS = {"ultralytics": "8.3.230", "torch": "2.8.0+cpu", "torchvision": "0.23.0+cpu", "MNN": "3.2.0", "onnx": "1.19.1", "onnxruntime": "1.23.2", "numpy": "2.2.6"}
SOURCE_SHA256 = {
    "yolo11n-cls.pt": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "yolo11n.pt": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "bus.jpg": "c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63",
    "zidane.jpg": "16d73869e3267a7d4ed00de8e860833bd1657c1b252e94c0c348277adc7b6edb",
}


def sandbox_path(path):
    path = path.resolve()
    sandbox = (ROOT / "runs/phase2-edge").resolve()
    if not path.is_relative_to(sandbox) or path == sandbox:
        raise ValueError("output must be a subdirectory of runs/phase2-edge")
    return path


def describe(path):
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def cleaned(text):
    for path, replacement in ((str(ROOT), "<workspace>"), (str(Path.home()), "<home>")):
        for variant in (path, path.replace("\\", "/")):
            text = re.sub(re.escape(variant), replacement, text, flags=re.IGNORECASE)
    return text


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def worker(args):
    # Prevent SDK logging, auto-installation, shell use and networking, even on failure.
    install()
    for path in (args.model, args.tensor, args.contract, args.output):
        if path is not None:
            sandbox_path(path)
    import numpy as np

    task = args.task
    if args.action in {"torchscript", "onnx", "prepare"}:
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS

        SETTINGS.update({"sync": False})
        model = YOLO(str(args.model), task=task)
        size = 224 if task == "classify" else 320
        if args.action != "prepare":
            options = dict(format=args.action, imgsz=size, batch=1, device="cpu", half=False, dynamic=False, nms=False)
            if args.action == "onnx":
                options.update(opset=17, simplify=False)
            model.export(**options)
            return
        import torch
        from PIL import Image

        model.model.cpu().eval().float().fuse()
        contract = dict(specimen=args.model.stem, task=task, input_layout="NCHW", input_dtype="float32", input_shape=[1, 3, size, size], num_classes=len(model.names), output_kind="classification-probabilities" if task == "classify" else "yolo11-detect-raw")
        for image in (args.model.parent / "bus.jpg", args.model.parent / "zidane.jpg"):
            if task == "classify":
                from ultralytics.data.augment import classify_transforms

                tensor = classify_transforms(size)(Image.open(image).convert("RGB")).unsqueeze(0)
                contract["preprocessing"] = "Ultralytics8.3.230 classify_transforms: RGB shortest-edge resize224 bilinear, center crop224, ToTensor, mean0/std1"
            else:
                import cv2
                from ultralytics.data.augment import LetterBox

                pixels = LetterBox((size, size), auto=False)(image=cv2.imread(str(image)))
                array = np.ascontiguousarray(pixels[..., ::-1].transpose(2, 0, 1))
                tensor = torch.from_numpy(array).float().unsqueeze(0) / 255
                contract["preprocessing"] = "Ultralytics8.3.230 LetterBox320x320 auto=False center=True scaleup=True padding114; BGR->RGB, BCHW /255"
            stem = args.model.stem + "-" + image.stem
            np.save(args.model.parent / (stem + "-input.npy"), tensor.numpy(), allow_pickle=False)
            with torch.inference_mode():
                output = model.model(tensor)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                output = output.numpy()
            contract["output_shape"] = list(output.shape)
            np.save(args.model.parent / (stem + "-pytorch.npy"), output, allow_pickle=False)
        contract["names"] = model.names
        write_json(args.model.with_suffix(".contract.json"), contract)
    elif args.action == "convert":
        # Importing MNN.tools starts a telemetry logger. Call the same native binding directly.
        import _tools

        if args.output.suffix != ".mnn" or args.output.resolve() == args.model.resolve():
            raise ValueError("conversion output must be a separate .mnn artifact")
        args.output.unlink(missing_ok=True)  # A stale prior export must not conceal converter failure.
        argv = ["mnnconvert", "-f", args.framework, "--modelFile", str(args.model), "--MNNModel", str(args.output), "--bizCode", "HeT-official-tooling-specimen-only"]
        if args.weight_bits:
            argv += ["--weightQuantBits", str(args.weight_bits)]
        _tools.mnnconvert(argv)
        if not args.output.is_file() or args.output.stat().st_size == 0:
            raise RuntimeError("native converter returned without a nonempty output model")
    elif args.action == "infer":
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        tensor = np.load(args.tensor, allow_pickle=False)
        if args.model.suffix == ".mnn":
            from tools.edge_runtime.mnn_cpu import run_mnn

            result = run_mnn(args.model, tensor, contract)
        elif args.model.suffix == ".onnx":
            import onnxruntime as ort

            session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
            result = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        else:
            import torch

            with torch.inference_mode():
                result = torch.jit.load(str(args.model)).eval()(torch.from_numpy(tensor))
                if isinstance(result, (tuple, list)):
                    result = result[0]
                result = result.numpy()
        from tools.edge_runtime.mnn_cpu import validate_output

        validate_output(result, contract)
        np.save(args.output, result, allow_pickle=False)


def predictions(array, contract):
    """Classification top5 or exact pinned official YOLO11 class-aware NMS, input coordinates."""
    import numpy as np

    if contract["task"] == "classify":
        ids = np.argsort(array[0])[-5:][::-1]
        return [{"class_id": int(i), "name": contract["names"][str(i)], "probability": float(array[0, i])} for i in ids]
    import torch
    from ultralytics.utils.nms import non_max_suppression

    kept = non_max_suppression(torch.from_numpy(array.copy()), conf_thres=0.25, iou_thres=0.45, nc=contract["num_classes"], max_det=100)[0].tolist()
    return [{"xyxy_input_pixels": row[:4], "confidence": row[4], "class_id": int(row[5]), "name": contract["names"][str(int(row[5]))]} for row in kept]


def main(args):
    isolated_environment(ROOT)
    versions = {package: importlib.metadata.version(package) for package in PINS}
    if versions != PINS:
        raise RuntimeError(f"version mismatch: expected {PINS}, observed {versions}")
    work = sandbox_path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    (work / "ultralytics-settings").mkdir(exist_ok=True)
    os.environ.update(YOLO_AUTOINSTALL="false", YOLO_CONFIG_DIR=str(work / "ultralytics-settings"), YOLO_OFFLINE="true", PYTHONIOENCODING="utf-8", PYTHONUTF8="1", OMP_NUM_THREADS="1")
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), purpose="official public tooling specimens, NOT HeT model choice/training/accuracy", platform={"system": platform.system(), "machine": platform.machine(), "python": platform.python_version()}, versions=versions, sources=[], stages=[], comparisons=[], artifacts=[], io_contracts=[])

    def stage(name, *options):
        command = [str(Path(sys.executable).resolve()), "-m", "tools.edge_export.smoke", "--worker", *map(str, options)]
        started = time.monotonic()
        try:
            completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
            code, output = completed.returncode, completed.stdout + completed.stderr
        except subprocess.TimeoutExpired as exc:
            code, output = 124, "worker exceeded 240 seconds\n" + str(exc)
        log = work / (name + ".log")
        log.write_text(cleaned(output), encoding="utf-8")
        record = {"name": name, "exit_code": code, "seconds": round(time.monotonic() - started, 3), "command": ["<isolated-python>", *[cleaned(arg) for arg in command[1:]]], "log": log.relative_to(ROOT).as_posix()}
        report["stages"].append(record)
        write_json(work / "export_results.json", report)
        print(f"{name}: exit={code}", flush=True)
        return code == 0

    for name, url in SOURCES.items():
        dest = work / name
        if not dest.exists():
            if not args.allow_downloads:
                raise RuntimeError(f"missing {name}; review fixed official sources and use --allow-downloads")
            print(f"download official specimen: {name}", flush=True)
            temporary = dest.with_suffix(dest.suffix + ".download")
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
            temporary.replace(dest)
        descriptor = describe(dest)
        if descriptor["sha256"] != SOURCE_SHA256[name]:
            raise RuntimeError(f"official specimen bytes changed: {name}; review upstream, do not load untrusted weights")
        report["sources"].append({**descriptor, "source_url": url, "license_context": "Ultralytics AGPL-3.0 / enterprise licensing; sample photos provided in official repo; original photo rights not independently audited; not redistributed in git"})
    stage("offline_guard_self_test", "--action", "guard")
    for task, name in (("classify", "yolo11n-cls"), ("detect", "yolo11n")):
        model = work / (name + ".pt")
        common = ["--model", model, "--task", task]
        if not stage(name + "-prepare", "--action", "prepare", *common):
            continue
        formats = []
        jit = model.with_suffix(".torchscript")
        if stage(name + "-torchscript", "--action", "torchscript", *common):
            formats.append(jit)
            direct = work / (name + "-direct-fp32.mnn")
            if stage(name + "-direct-TORCH-fp32", "--action", "convert", "--framework", "TORCH", "--model", jit, "--output", direct):
                formats.append(direct)
        onnx = model.with_suffix(".onnx")
        if stage(name + "-onnx", "--action", "onnx", *common):
            formats.append(onnx)
            for bits in (0, 8):
                dest = work / (name + ("-onnx-weight8.mnn" if bits else "-onnx-fp32.mnn"))
                if stage(dest.stem, "--action", "convert", "--framework", "ONNX", "--weight-bits", bits, "--model", onnx, "--output", dest):
                    formats.append(dest)
        import numpy as np

        contract_file = model.with_suffix(".contract.json")
        contract = json.loads(contract_file.read_text(encoding="utf-8"))
        report["io_contracts"].append({key: value for key, value in contract.items() if key != "names"})
        for image in ("bus", "zidane"):
            stem = name + "-" + image
            tensor = work / (stem + "-input.npy")
            reference = np.load(work / (stem + "-pytorch.npy"), allow_pickle=False)
            for artifact in formats:
                result_file = work / (stem + "-" + artifact.stem + artifact.suffix.replace(".", "-") + "-output.npy")
                if not stage(result_file.stem, "--action", "infer", "--model", artifact, "--tensor", tensor, "--contract", contract_file, "--output", result_file):
                    continue
                actual = np.load(result_file, allow_pickle=False)
                difference = np.abs(actual - reference)
                report["comparisons"].append({"task": task, "sample": image, "format": artifact.name, "input": describe(tensor), "output_shape": list(actual.shape), "max_absolute_error": float(difference.max()), "mean_absolute_error": float(difference.mean()), "allclose_atol1e-4_rtol1e-4": bool(np.allclose(actual, reference, atol=1e-4, rtol=1e-4)), "pytorch_predictions": predictions(reference, contract), "export_predictions": predictions(actual, contract), "postprocess": "classification top5" if task == "classify" else "Ultralytics8.3.230 non_max_suppression conf0.25 iou0.45 class-aware max_det100, coordinates in 320x320 input; no claim of original-image boxes"})
        report["artifacts"].extend(describe(path) for path in [jit, onnx, *formats] if path.is_file() and path not in [ROOT])
    report["artifacts"] = list({item["path"]: item for item in report["artifacts"]}.values())
    report["quantization"] = "weight8 uses --weightQuantBits8 storage compression, no activation calibration, no integer-kernel/speed claim; no synthetic/random calibration; no HeT dataset accessed"
    report["target_validation"] = "Windows AMD64 CPU only; no Ubuntu, ARMv7, RK3506, device memory/latency or hardware validation"
    report["stage_timing_note"] = "seconds include whole Python worker startup and model loading; not a warmed inference benchmark"
    report["mnn_license_observation"] = {"installed_wheel_metadata": importlib.metadata.metadata("MNN").get("License"), "upstream_readme": "Apache 2.0", "source_url": "https://github.com/alibaba/MNN#license", "limitation": "wheel metadata and upstream license label differ; verify the exact redistribution bundle before deployment"}
    unexpected_failures = [s["name"] for s in report["stages"] if s["exit_code"] != 0 and not s["name"].endswith("-direct-TORCH-fp32")]
    complete = not unexpected_failures and len(report["comparisons"]) >= 16
    report["status"] = "completed_with_direct_route_blocked_and_measured_numerical_drift" if complete else "incomplete"
    report["unexpected_failures"] = unexpected_failures
    write_json(work / "export_results.json", report)
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "runs/phase2-edge/export-smoke")
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--action", choices=["guard", "prepare", "torchscript", "onnx", "convert", "infer"])
    parser.add_argument("--task", choices=["classify", "detect"])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--framework", choices=["TORCH", "ONNX"])
    parser.add_argument("--weight-bits", type=int, choices=[0, 8], default=0)
    parser.add_argument("--tensor", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    isolated_environment(ROOT)
    if args.worker:
        if args.action == "guard":
            from .offline_guard import self_test

            self_test()
        else:
            worker(args)
    else:
        main(args)
