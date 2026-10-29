# Phase 2 Oven/Edge Foundation Plan

**Spec:** User Phase 2 request (2026-10-29); existing AGENTS/DATA_CONTRACT. User explicitly chose autonomous execution rather than a design approval pause.
**Goal:** Working offline capture/import/audit plus evidenced YOLO/MNN research before expected oven arrival 2026-10-30.
**Architecture:** Add tools/oven_capture independently of MVP. Reuse audit_images pHash/BKTree/image_facts and propose_groups.UnionFind; reuse frozen classes/shared nine-field validator without edits. Optional OpenCV adapter only opens an explicitly selected device. Export experiments and dependencies isolated under ignored runs/phase2-edge.

## Constraints
- Branch feature/phase2-oven-edge-foundation from f77ced3; no main merge/Release.
- Preserve all MVP frozen configs/manifests/code/real weights; do not rerun frozen test to choose new models.
- Raw images, model files, device identifiers, private paths only in Git-ignored storage; never upload automatically.
- No heating, firmware, driver/system/WSL/Docker installs or unknown hardware probes.
- Enterprise facts: YOLO/INT8/Linux/MNN/RK3506 family/1080p camera/joint old+new-data retraining. Exact task, model, camera interface, SKU/RAM/SDK/access remain UNKNOWN.

## Scope / ownership
- Parent: tools/oven_capture/{core,camera,__main__}.py, tests/test_oven_capture.py, capture protocol, evidence/integration.
- Parallel export task: tools/edge_export, tools/edge_runtime, tests/test_edge_*.py, YOLO_MNN_EXPORT_REPORT and actual isolated experiments.
- Parallel platform task: EDGE_DEPLOYMENT_PREP and OVEN_ARRIVAL_CHECKLIST with official sources.

## Steps and checks
- [x] Clean original/MVP workspaces; native independent worktree; protected hashes recorded. Existing unit baseline: 68 PASS, exit0 (37.797s).
- [x] RED capture tests: byte-preserving imports, explicit timezone metadata, pending/manual labels, corruption/orphan/tamper detection, SHA/pixel/near duplicates, transitive sample/session groups, guarded nine-field export and cross-split rejection, finite replaceable frame source, no output into tracked paths.
- [x] Minimal implementation and GREEN tests; offline SYNTHETIC CLI smoke only, clearly labelled. No real-camera PASS claims. Review regression cases failed before fixes, then passed; export now requires a complete local multi-collection snapshot.
- [x] Passive camera inventory saved ignored; two local candidates, identity/resolution UNKNOWN, opened_any_device=false. Missing/unknown interfaces remain BLOCKED/UNKNOWN.
- [x] Official lightweight cls/det PyTorch/MNN export attempts, dependency and shape/size/SHA logs, prediction comparisons; two direct TORCH failures, ONNX/MNN CPU runs, weight8 drift honestly retained. Upstream wrapper environment incident corrected with limitations; see evidence.
- [x] Official Linux/ARMv7/RK3506 research; all 20 physical hardware acceptance rows BLOCKED.
- [x] Read-only integration review and full unittest suite: 99 run, 98 passed, 1 OS-permission skip, exit0. Protected hashes and evidence recorded in foundation report. Images-only interrupted collections also included in catalog discovery after RED regression.
- [x] Final delivery checks and user-authorized route: commit/push only new branch, independent Draft PR; no main merge/Release, preserve worktree. Actual commit/PR identity is recorded by Git/PR and the final response.

## Review focus
No automatic labels/splits; multi-food captures cannot masquerade as classification labels or detection boxes. Same sample/session and duplicate connections are transitive. Path traversal and tracked output protection apply to CLI and manifest export. Corruption fails closed; actual frame resolution is observed, not a requested 1080p guarantee. PNG of a decoded frame is not sensor RAW. Import times are distinct from operator-supplied capture times. New experiments never use frozen MVP test for calibration/selection.

## Ledger
All locally modified work was preserved; original main and MVP were clean. Existing Gradio7860 is user-owned and must not be stopped. Native Ubuntu target and RK board access are not supplied; do not guess a remote host or install a Linux subsystem.
