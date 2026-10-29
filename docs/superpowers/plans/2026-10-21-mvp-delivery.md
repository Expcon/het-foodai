# MVP v1.0 Delivery Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline; final independent review only.

**Goal:** Deliver a reproducible integration/mvp-v1.0-delivery draft PR; missing real assets remain BLOCKED.
**Architecture:** Start at Sun's real-model branch; merge Yang's v1.1.0 history and byte-preserved data. Retain shared safe loader, transforms and UI. Fix only acceptance statistics and delivery documentation.
**Tech Stack:** Windows Python 3.12, project venv, PyTorch/torchvision, Gradio, unittest.
**Spec:** User's six-task final delivery request; AGENTS.md; DATA_CONTRACT and INFERENCE_CONTRACT.

## Global constraints
- Never merge main or publish a Release; retain both source histories.
- Frozen v1.1.0 classes: carrot,potato,broccoli,pumpkin,tomato,white_radish,cucumber,bell_pepper.
- Manifest SHA256: 6bef9776c278d8efac86008a11c47386cb7d95b7164073b9bd07e84cb5cf2821.
- Checkpoint SHA256: 41056b3ea90d5fa718a18f48cf5182373d1010c5185cb426ba5c949b1ec1f4c5.
- No weights, raw images, venv, secrets or new personal filesystem paths in Git; historical acceptance JSON stays unchanged.

## Review focus
- Wrong last-sample median; P95 outside observed samples; per-image warm-up stated accurately.
- Empty/missing image directory and repeats <= 0 fail clearly before loading weights.
- Data version/class mapping/hash mismatch blocks real acceptance.
- Historical acceptance JSON must never be presented as corrected remeasurement.
- Fresh clone must work without untracked source files or host-only imports.

## Tasks and verification
- [x] Create isolated worktree from origin/feat/app-real-model-demo 8d6fda0; merge origin/feat/data-inventory 524879a retaining history. No conflicts; protected model/app files unchanged. Baseline unittest: 63 PASS.
- [x] Add tests/test_acceptance.py: main() deterministic measurements expose last-sample and exclusive-P95 errors; reject empty/missing directories and invalid repeats with exit 2. Run RED.
- [x] Fix scripts/acceptance_run.py: collect every internal latency; median; nearest-rank P95 ceil(.95*N); record raw samples, exact scope, per-image warmup and repeats; checkpoint basename in new reports. Run GREEN and full suite.
- [x] Verify Release cache/digests, v1.1.0 manifest bytes/counts/files and frozen classes. Verify real checkpoint + embedded metadata only if actual file supplied.
- [x] Update README.md and docs/MVP_V1_DELIVERY.md with portable setup, restore/check/UI commands, honest baseline provenance, source restrictions and remaining work. Annotate historical timing limitations without editing JSON.
- [x] Actual local Gradio and CUDA acceptance when real checkpoint exists; otherwise record BLOCKED. Fresh clone final branch in independent ignored directory, create fresh project venv and test deployment.
- [x] Whole-branch review, full tests and explicit branch-only Draft PR handoff prepared; actual PR creation/link recorded by the final task response. No merge/Release.

## Rulings / ledger
User supplied execution method and explicitly authorized shared README/config updates; continue autonomously.
Real checkpoint not found in project/worktrees/Downloads; asked for artifact path while independent work proceeds.
Historical reports are evidence from Sun's environment, not current-machine remeasurements.

Ledger update 2026-10-22: supplied Baseline tar restored with path whitelist; model/config/manifest match. CPU/CUDA ready exit 0, corrected acceptance exit 0; fixed-weight test recheck 1067/1067. Final unit suite 68/68 (19.319s). Review found false-success exit on error-case failure; RED reproduced, gate fixed and reviewer confirmed. Browser real JPG/empty input PASS; origin permission later declined, further UI work stopped and not marked PASS. Fresh-clone deployment remains pending.

Final ledger: fresh GitHub clone at 6ffe216 with its own venv (no system site packages), torch2.8.0+cpu/torchvision0.23.0+cpu. Restore-before check exit1/not_ready; restored check exit0/ready; real 9-file CPU acceptance exit0; Gradio startup GET200/ready. Full clean suite 68 tests, 67 passed/1 CUDA-only skipped, 0 failures, 12.487s. Code fingerprints match the main integration verification. Further browser work was declined; no workaround attempted. Candidate is ready for course demonstration review; source licensing/semantic/generalization/edge limits remain.
