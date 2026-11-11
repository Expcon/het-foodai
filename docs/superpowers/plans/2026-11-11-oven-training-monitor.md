# Oven Training Monitor Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline; one read-only whole-change review at the end. The user has already authorized autonomous implementation; do not pause for another design approval.

**Goal:** Catch final-eval per-class failures during exploratory training and prepare an isolated BN statistics comparison.

**Architecture:** Extend the existing `fit_exploratory`; preserve its nested Dataset and all callers. A small monitor handles fixed eval, BN snapshots and epoch/batch evidence. A controlled runner validates recipes and physical holdout provenance before forwarding to the same trainer.

**Tech Stack:** Existing Python3.12 / torch / torchvision / Ultralytics8.3.230 project environment; no new dependencies.

**Spec:** User request dated2026-11-11 and `docs/OVEN_V2_MODE_AUDIT.md`.

## Constraints and decisions

- Never overwrite old runs, raw data, classes, manifests, checkpoint or ONNX. No Desk/product inference changes.
- Two arms differ only in BN running-statistics policy. BN affine parameters remain equally trainable; full affine freezing is outside this comparison.
- Same pretrained SHA, seed42,224/RGB/pad114,AdamW/lr0.001/wd0.0005,batch32,augmentation,drop_last=False. Thirty epochs are recipe definitions, not authorization to run them now.
- Fixed training eval is a training diagnostic, never a substitute validation set. Missing/unverified independent physical/session IDs => BLOCKED; no automatic split or invented IDs.
- Preserve public contracts/configs/requirements. Experimental recipes live under `tools/oven_training/recipes/`.
- Native execution in existing linked worktree, new branch `feat/model-oven-training-monitor`; keep original audit commits.

## Review focus

1. Eval accidentally changes BN buffers, model modes or future RNG/augmentation sequence.
2. Frozen BN is undone by the next epoch's `model.train()` or also freezes affine parameters unintentionally.
3. Class metrics are averaged by batch rather than sample, or absent classes divide by zero.
4. Eval iterators consume the training shuffle RNG; last small batch disappears without evidence.
5. Weak or overlapping physical/session/hash data is accepted as independent validation.

## Tasks

- [x] Write failing tests for fixed eval metrics/mode+BN+RNG preservation, per-epoch frozen BN, exact sample/last-batch logging, recipe pairing and holdout rejection.
- [x] Add `tools/oven_training/monitor.py`: `set_training_mode`, `bn_snapshot`, `bn_delta`, `evaluate_fixed`; keep the epoch file writer in the existing trainer.
- [x] Extend `tools/oven_training/train.py` with optional recipe and qualified validation inputs; epoch0/each-epoch eval, BN and batch trace; same final-only checkpoint policy.
- [x] Add `tools/oven_training/controlled.py`: strict recipes, validation readiness, prepare-only default CLI; require explicit `--run` for training.
- [x] Add paired JSON recipes and an empty physical-ID intake template; report existing pool readiness without modifying it.
- [x] Verify with all unit tests, two short synthetic GPU integration fits, and pinned real pretrained/v2 read-only inspection; never train real food data this turn.
- [x] Write reproducibility/evidence guide and read-only review; commit only the independent branch.

## Execution ledger

- Start: clean `e6d1cb9`; existing linked worktree and project environment reused.
- Ruling: freeze running statistics only, affine parameters trainable in both arms — isolates the confirmed normalization mechanism without changing optimizer membership.
- Ruling: no real independent specimens can be invented from current filenames or visual judgments; deliver intake/gates/readiness instead of a fabricated validation split.
- Ruling: cache deterministic image framing once; retain historical nested Dataset for audit replay compatibility.
- Ruling: each synthetic GPU arm runs2 epochs,65 images — verifies freeze reapplication across epoch boundaries and the last1-image batch; no real food training.
- Baseline183 tests passed with1 skip. TDD red logs precede monitor/CSV/duplicate-guard implementations. Final191 tests:190 passed,1 skip,exit0,55.788s.
- Actual GPU pair:39 BN layers, equal initial states/trainable parameter count and all augmented batches; update counters+3/epoch, frozen running state stable; safe load of synthetic outputs passed.
- Read-only real v2 eval:1996 images,1857 directory agreement, potato59/163 and sausage71/97; all parameters and buffers unchanged. Independent val remainsBLOCKED.
- Review finding: repeated validation image could fake class coverage. Fixed byte/pixel duplicate and conflict guards;8 targeted tests pass; reviewer confirmed closure.
- First smoke module invocation collided with a third-party tests package. Correct direct script passed; new user SDK setting file reverted after metadata verification. Automatic approval rejected removing the new empty directory (`blocked by policy`); left it intact, did not bypass the rejection.
- Historical preservation:2925 dataset +220 field +14 protected files unchanged. No real training, global installs, production inference modifications, push or PR merge.
