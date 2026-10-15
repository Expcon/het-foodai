# Sprint 0 — three parallel tracks (start 2026-10-15)

> Review these tasks together. No teammate should need another teammate's trained model to make initial progress.

## Task A — 杨熙珂, dataset (branch `feat/data-inventory`)

1. Audit the proposed 8 food categories for strict semantic match and usable source; propose replacements where availability is weak.
2. List URLs, access date, permissions, true *independent* originals, whether images are already augmented, raw/cooked mismatch.
3. Extend `scripts/validate_manifest.py` or add `scripts/audit_images.py` for corrupted image detection, per-class/source counts, exact and perceptual duplicate checks.
4. Propose frozen `configs/classes.json`, source manifest, group-safe split and data-version `v1` for joint approval.
5. Submit a PR with code + `docs/DATA_SOURCES.md` + honest audit report. Do NOT push raw images.

**Acceptance:** running the script reproduces counts; sample photos manually spot-checked; each split has no known group/SHA leakage; unavailable data clearly labeled.

## Task M — group lead, training (branch `feat/model-baseline`)

1. Verify project `.venv` and GPU-supported PyTorch only in local env; log exact versions and `cuda.is_available()`.
2. Smoke-check repository unit tests. Review `scripts/train_classifier.py`; work on stable training checkpoint/logging and reproducibility. No real training on draft class config.
3. Once dataset v1 is frozen: perform first 8-class pre-trained EfficientNetV2-S run, store logs (not weight files in git), use train+val only.
4. Record accuracy, macro-average, per-class errors on validation; independently run *one* frozen test assessment after model choice.
5. Follow with 8→12→16 incremental experiments using all previously seen data as permitted; compare cumulative joint training vs sequential fine-tuning.

**Acceptance:** reproducible command, class/manifest hashes, actual checkpoints, logs and evaluation evidence. Never fabricate performance.

## Task B — 孙璞月, inference/demo (branch `feat/app-shell`)

1. Implement Gradio UI accepting JPEG/PNG, with legible image preview, top-3 labels/scores and clear untrained state.
2. Implement a predictor adapter following `docs/INFERENCE_CONTRACT.md`, initially returning `ModelNotReadyError` instead of mock accuracy.
3. Build input validation and at least success/error/not-ready test scenarios using fixture assets labeled synthetic.
4. Once a real checkpoint arrives, load model, compare Python predictions with lead's reference results, check model/config hashes and measure CPU/GPU latency.
5. Make deployment research **after** target hardware and memory are confirmed; draft ONNX export as an optional later phase.

**Acceptance:** app launches and responds gracefully to invalid / absent model; later loads an actual checkpoint and reports real results.

## Coordination and code review

- Branches/PRs instead of competing edits on `main`.
- Changes to class order, manifest columns or predictor output are **interface changes**: discuss in group first.
- Week 1: demonstrate one valid 8-class data pipeline + genuine model + UI. A date is only a planning target.
- Each PR includes changed files, how to run, one expected success, one expected failure, and results.
- No external hardware deployment or cloud funding is required for this initial MVP.
