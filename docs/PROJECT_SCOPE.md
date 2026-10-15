# Course MVP vs later research phases

**Project goal:** reliable food-category recognition for smart-oven use, emphasizing classification accuracy, reproducible datasets, expansion from 8 to 12 to 16 categories, robustness and eventual edge deployment.

## Sprint 0 (software foundation; current)
- Agree on an ordered eight-class scope; do not mistake the 28 reference examples for a full training dataset.
- Build a group-separated train/validation/test dataset manifest, deduplication and manual review.
- Establish reproducible local training + evaluation and interactive inference shell.

## MVP 1 (first real end-to-end deliverable)
- User supplies a valid JPEG/PNG of a single primary food item.
- Real checkpoint returns Top-1/Top-3 category scores and clear error states.
- Training evidence: data version and frozen split, checkpoint, real accuracy and confusion matrix from a held-out test after model selection.
- UI must clearly indicate limited operating assumptions (mixed foods, unknown classes, lighting).

## Phase 2 (after MVP 1)
- 8→12→16 category expansion with cumulative full-data joint training baseline and optional sequential fine-tuning comparison.
- Robustness tests for cross-source backgrounds, low light, warmer lighting and oven scenes when available.
- Engineering evidence for small vs large model if size/time becomes a constraint.

## Phase 3 (after hardware identity is known)
- Check exact board, RAM, supported runtime, model conversion, quantization and actual latency / peak RAM.
- Do not claim edge feasibility from RTX 5080 timing or from ONNX export alone.

## Current explicit exclusions
- No automatic cooking control, food weighting, portion-size estimation, ingredient counting, multiple-object detection, mobile account, hosted backend, cloud database or real edge-device certification in MVP 1.
- Can revisit scope after team and teacher agreement, but new requirements must have tests.

## Evaluation philosophy
- A high public-web accuracy is not proof of good performance inside a real oven.
- Use same splits and definitions across experiments; never tune on the held-out test.
- Choose evaluation targets with the teacher; do not fabricate any achieved value.
