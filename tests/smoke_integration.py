"""SYNTHETIC data -> training -> checkpoint -> predictor -> local Gradio checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smoke_training import main as training_smoke

EVIDENCE = ROOT / "runs/integration-20261020"


def demo_checks(scratch, run_dir, classes, *, browser_check=False):
    from gradio_client import Client, handle_file
    from foodai.predictor import PredictionAdapter, as_contract_dict
    image = scratch / "val-0.png"
    broken = scratch / "SYNTHETIC-broken.png"
    broken.write_bytes(b"SYNTHETIC damaged PNG, not an image")
    adapter = PredictionAdapter(run_dir / "best.pt", classes, device="cuda")
    assert adapter.is_ready, adapter.status().message
    reference = as_contract_dict(adapter.predict(image))
    assert len(reference["predictions"]) == 3
    del adapter
    import torch
    torch.cuda.empty_cache()
    processes = []
    logs = []
    try:
        for name, port, extra in (("ready", 7861, ["--checkpoint", str(run_dir / "best.pt"),
                                                 "--classes", str(classes), "--device", "cuda"]),
                                  ("missing", 7862, [])):
            log = (EVIDENCE / f"gradio-{name}.log").open("w", encoding="utf-8")
            logs.append(log)
            entry = [str(ROOT / "app/app.py")]
            if name == "ready" and browser_check:
                gate = EVIDENCE / "ui-gate"
                gate.mkdir(exist_ok=True)
                for flag in ("block", "started", "release"):
                    (gate / flag).unlink(missing_ok=True)
                entry = [str(Path(__file__)), "--serve-gated", str(gate), "--"]
            process = subprocess.Popen([sys.executable, *entry, "--port", str(port), *extra],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "GRADIO_ANALYTICS_ENABLED": "False", "PYTHONUTF8": "1"})
            processes.append(process)
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Gradio {name} exited; see gradio-{name}.log")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/config", timeout=1) as response:
                        config = json.load(response)
                    break
                except OSError:
                    time.sleep(.2)
            else:
                raise RuntimeError(f"Gradio {name} startup timed out")
            buttons = [component["props"] for component in config["components"] if component["type"] == "button"]
            assert buttons[0]["interactive"] == (name == "ready")
        ready = Client("http://127.0.0.1:7861", verbose=False)
        uploaded = ready.predict(handle_file(str(image)), api_name="/on_upload")
        assert "图片已载入" in uploaded[0]
        prediction = ready.predict(handle_file(str(image)), api_name="/on_predict")
        assert "识别完成" in prediction[0] and len(prediction[1]["data"]) == 3
        assert "CUDA已同步" in prediction[2]
        reference_labels = [row["label_zh"] for row in reference["predictions"]]
        assert [row[1] for row in prediction[1]["data"]] == reference_labels
        invalid = ready.predict(handle_file(str(broken)), api_name="/on_upload")
        assert "输入不是有效图片" in invalid[0] and invalid[2]["data"] == []
        invalid_prediction = ready.predict(handle_file(str(broken)), api_name="/on_predict")
        assert "输入不是有效图片" in invalid_prediction[0] and invalid_prediction[1]["data"] == []
        missing = Client("http://127.0.0.1:7862", verbose=False)
        unavailable = missing.predict(handle_file(str(image)), api_name="/on_predict")
        assert "无法完成识别" in unavailable[0] and unavailable[1]["data"] == [] and not unavailable[2]
        evidence = {"synthetic_only": True, "program_chain": "PASS", "normal_upload_top3": "PASS",
            "adapter_demo_top3_match": "PASS", "corrupt_upload_and_prediction": "PASS",
            "missing_model_no_scores_button_disabled": "PASS", "checkpoint": str(run_dir / "best.pt"),
            "classes": str(classes), "image": str(image), "broken_image": str(broken),
            "ready_url": "http://127.0.0.1:7861", "missing_url": "http://127.0.0.1:7862"}
        (EVIDENCE / "demo-verification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print("PASS: real local Gradio HTTP upload/Top-3/corrupt/missing-model paths; all inputs SYNTHETIC", flush=True)
        if browser_check:
            done = EVIDENCE / "browser-done"
            done.unlink(missing_ok=True)
            print("BROWSER_READY: 7861 synthetic checkpoint; 7862 missing model; see demo-verification.json", flush=True)
            deadline = time.monotonic() + 1200
            while not done.exists() and time.monotonic() < deadline:
                time.sleep(.5)
            assert done.exists(), "Browser confirmation timed out"
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-check", action="store_true", help="Keep local servers up for browser verification")
    parser.add_argument("--serve-gated", type=Path, help=argparse.SUPPRESS)
    args, remaining = parser.parse_known_args()
    if args.serve_gated:
        # Test-only barrier around the real predictor, for observing disabled browser controls.
        sys.path.insert(0, str(ROOT))
        from foodai.predictor import PredictionAdapter
        from app.app import main as app_main
        original_predict = PredictionAdapter.predict
        def gated_predict(self, image):
            if (args.serve_gated / "block").exists():
                (args.serve_gated / "started").touch()
                deadline = time.monotonic() + 180
                while not (args.serve_gated / "release").exists() and time.monotonic() < deadline:
                    time.sleep(.05)
                if not (args.serve_gated / "release").exists():
                    raise RuntimeError("Synthetic browser gate timed out")
            return original_predict(self, image)
        PredictionAdapter.predict = gated_predict
        sys.exit(app_main(remaining[1:] if remaining[:1] == ["--"] else remaining))
    if remaining:
        parser.error("Unexpected arguments")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    training_smoke(after_training=lambda *paths: demo_checks(*paths, browser_check=args.browser_check))
