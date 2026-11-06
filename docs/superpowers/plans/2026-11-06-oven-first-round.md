# Oven first-round exploratory training

Goal: audit the supplied real ZIP, diagnose the unchanged MVP model, train and batch-test a lightweight YOLO classifier, and deliver a reproducible offline field script.

User explicitly requested autonomous execution. Base7054b16 is preserved on its foundation branch; new work uses feat/model-oven-first-round. No hardware, main merge, frozen data/config/weights edits or automatic upload of source media.

Architecture: separate tools/oven_training audit, fixed-config training and inference. Existing source ZIP helpers and image audit are reused. Candidate labels are ZIP folder keys in an experimental namespace, never a replacement public class contract. Missing physical specimen/session identity prevents independent holdout claims; conservative grouping and train-only exploration are allowed by the request. No validation-based selection, no early stopping, final epoch only.

- [x] Data owner task: bounded extraction, image/hash/near-duplicate audit, original split counts, uncertainty groups, full records including exclusions; synthetic regression tests.
- [x] Parent: isolated project .venv with read-only existing CUDA package inheritance; install missing SDK dependencies only here. Lock versions and pre-training hyperparameters.
- [x] Diagnose real Baseline via existing safe adapter. Report overlap classes separately from unsupported source categories.
- [x] Train YOLO11n-cls fixed30 epochs,224 input,AdamW lr0.001 weight_decay0.0005,batch32,seed42,FP32; preserve whole-frame aspect ratio with padding, simple train-only augmentation; last checkpoint only. These settings are engineering choices, not tuned on this dataset's metrics.
- [x] Safe state_dict checkpoint with class/preprocess/data provenance; standalone batch Top3, synchronized timing, missing/corrupt input errors; no fake predictions.
- [x] Actual run logs, resubstitution-only confusion/failures, unit suite, hashes before/after, operating guide and local commit. No source media or weights committed.

Tests target missing identities, ZIP traversal/corruption, duplicates across original splits, invalid/mixed-split manifests, checkpoint metadata/class/preprocess mismatch and malformed field inputs. Model numerical results are real; synthetic tests prove software only.

Final evidence: 115 tests,114 pass/1 OS-permission skip,exit0;568 unique real images trained30 epochs;CPU/CUDA 569-image batch runs exit0;field CLI 7 cases passed as expected. Formal independent evaluation remains blocked.
