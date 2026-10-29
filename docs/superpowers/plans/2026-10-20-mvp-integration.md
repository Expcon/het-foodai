# MVP integration implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prove data/training/inference/Demo interoperability on integration/mvp-v1, keeping real training BLOCKED until team freeze.
**Architecture:** Preserve existing contracts and versioned data bytes. Reuse the training checkpoint loader and validation transform from the Demo.
**Tech Stack:** Existing project Python 3.12 venv, PyTorch/CUDA, Pillow, Gradio, unittest.
**Spec:** User four-phase integration request; AGENTS.md; DATA_CONTRACT.md; INFERENCE_CONTRACT.md.

## Global constraints
- Only integration/mvp-v1; retain member history; never merge main.
- No taxonomy semantics change or formal freeze. Synthetic frozen classes only in isolated scratch.
- No system installs. Reuse D:/Projects/HeT_FoodAI/.venv/Scripts/python.exe.
- Verify local data before any authorized Release download.

## Review focus
- Corrupt checkpoint or sidecar-only metadata must never cause ready.
- Strict safe loader rejection must not leave partial model state.
- Synchronize GPU timing and state decode/transfer scope.
- Upload failures and changes must clear stale results.
- Preserve all dataset manifest, class proposal and summary SHA256 bytes.

## Tasks
- [x] Fresh refs, clean main, contracts and branch audit; baseline 8/8 pass.
- [x] Native isolated worktree, integration/mvp-v1 from origin/main f229bc2.
- [x] Merge training 9c9b60a, data f9481c2, app 22df321 in that order; zero conflicts.
- [x] Add RED tests for real readiness, shared loader, mapping mismatch and CUDA timing.
- [x] Minimal adapter/UI corrections; full unit suite GREEN.
- [x] Local package search, hash validation and data contract audit; no metadata edits.
- [x] Full synthetic GPU training, save/load, adapter Top-3 and Gradio browser paths.
- [x] Whole-branch review; report actual tests, BLOCKED items and PR #1/#2/#3 verdicts.
- [x] Prepare branch-only commit and draft-PR handoff; program checks pass. Never merge main.

## Rulings
User explicitly requested autonomous execution, so implement inline without another plan approval pause.
The original worktree remains on main and unchanged. Safe loading and readiness are integration fixes authorized across the app/model boundary; no public contract edit is needed.

## Verification ledger

Final suite: 60/60, exit 0 (17.183s); full synthetic GPU/Gradio/browser smoke exit 0; real-data SHA and decode PASS. Formal training CLI exit 2 as expected. Dataset metadata exact bytes unchanged; ordinary text is Git-normalized because native checkout uses CRLF. Whole-branch review resolved corrupt-parser and in-flight upload findings; real browser control-lock/restore verified. Branch-only handoff follows this verified checkpoint.
