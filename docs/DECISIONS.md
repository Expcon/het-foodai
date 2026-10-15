# Decision log

## D-001 — 2026-10-15 — one primary model for Sprint 0
- Confirmed from team chat: group lead trains high-accuracy EfficientNetV2-S candidate; peers build data engineering and app/inference.
- We are **not** simultaneously requiring a second MobileNet model for Sprint 0; can be revisited after a working baseline.
- Rationale: reduce integration cost, preserve independent technical ownership for all three.

## D-002 — 2026-10-15 — dataset self-sourced and frozen for evaluation
- Enterprise supplies no curated training/test dataset per teacher discussion.
- Group must source and verify original images, license and split quality itself.
- Full replay of previous-class data is allowed per teacher discussion; still compare incremental protocol and keep evaluation clean.

## D-003 — 2026-10-15 — hardware later
- Desktop training and PC demo first. Edge board variant and memory not finalized.
- Do not promise RK3506 performance or claim the enterprise will supply a higher-end board.
