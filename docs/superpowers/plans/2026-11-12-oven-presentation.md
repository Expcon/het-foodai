# OvenCapture Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task. Steps use checkbox syntax.

**Goal:** Deliver a real Windows recognition-first UI without changing frozen model or camera behavior.

**Architecture:** DesktopApp continues to own workers and validated ModelEngine. A pure RecognitionSession binds async work to copied frames/models; a Tk presentation shell handles the four pages and result display.

**Tech Stack:** Existing Python3.12, Tkinter/ttk, Pillow, OpenCV, ONNX Runtime, PyInstaller; no new dependencies.

**Spec:** docs/superpowers/specs/2026-11-12-oven-presentation-design.md

## Global Constraints

- Independent branch/output; preserve v1.2.2 and all historical checkpoint/ONNX bytes.
- No inference_engine/camera_devices/capture_core/model_catalog contract redesign.
- No auto camera/serial opening or heating control; no invented scores, timings or accuracy.
- Native screenshots and real EXE tests required; untested hardware/DPI environments explicitly remain unverified.

## Review Focus

- Image change A→B→A while A worker runs must reject the earlier A completion (generation, not SHA alone).
- Editing model config during load must reject that load's eventual ready event.
- Source/model changes and corrupt images clear old result/save state, even when a task is busy.
- Saved frame must be the inference snapshot, never the newest live preview frame.
- A long source path/label and scaled fonts must not hide the recognize/error/save controls.

### Task 1: Bind request/result state

**Files:** new tools/oven_desk/recognition_session.py; new tools/oven_desk/tests/test_recognition_session.py.

**Interfaces:** `RecognitionSession.set_source(kind, label, frame=None, original_sha256=None)`, `invalidate_model()`, `begin(frame, model_identity)`, `finish(request, result, latency_ms)`, `fail(request, reason)`, `save_result(folder)`.

- [x] RED: synthetic tests for copied frames, busy reject, A→B→A, stale success/error, model invalidation, exact saved PNG/JSON hashes and exclusive names.
- [x] GREEN: implement only the state/persistence logic; retain core write_exclusive.
- [x] Run Desk tests and confirm real failure/green evidence; record ledger.

### Task 2: Presentation shell and worker integration

**Files:** new tools/oven_desk/presentation.py; modify oven_capture_gui.py; new tests/test_presentation.py and real Tk smoke script.

**Interfaces:** `Presentation(app)` creates tabs/header/home/result/log widgets, displays source/result/error, leaves app.preview/capture buttons usable by original camera flow. DesktopApp routes model and prediction events through generation/config guards.

- [x] RED: UI/state tests for default tab, image import without camera, invalidation on image/config/disconnect, stale completion, repeated clicks, visible errors and current-frame snapshot.
- [x] GREEN: build shell; split advanced controls; import JPG/PNG through existing decoder; bind actual measured results and exclusive save actions. Keep camera worker/core and engine bytes unchanged.
- [x] Run original GUI/model/capture tests and new real Tk source smoke, comparing v2/R1/R2 predictions to saved baseline.
- [x] Measure layout at both window sizes and 100/125/150% Tk scales; record actual DPI separately.

### Task 3: Windows delivery and native acceptance

**Files:** build_windows.ps1 version/output protection; usage and regression report. All binary/model/screenshot artifacts in runs/oven-desk-presentation-20261112 only.

- [x] Build new version with existing isolated PyInstaller environment and copied unchanged v2/R1/R2 payloads/catalog.
- [x] Run full project tests, packaged healthcheck/model checks/errors and real native EXE clicks/import/recognize/model switch/save.
- [x] Capture four real program states and viewport evidence; do not claim hardware acceptance.
- [x] SHA/ZIP CRC/relocation checks; independently review whole change; fix concrete critical/important issues with regression tests.
- [x] Commit only source/tests/docs; report exact artifact paths and remaining unverified environments. Never merge main or publish Release.
