# Oven source: unchanged Baseline diagnostic

The unchanged eight-class Baseline completed real CUDA inference on all **569 supplied oven ZIP images**, with **0 prediction errors** and process **exit code 0** on 2026-11-06. This is a descriptive distribution-shift diagnostic, not a formal generalization or independent-test result.

The new source contains nine labels. Only `broccoli` and `potato` occur in the frozen MVP vocabulary. The remaining seven labels cannot be represented by this checkpoint. No nine-class accuracy is calculated, and softmax confidence is not used as an OOD detector.

## Source and model identity

- Input: the real user-supplied oven ZIP audited in `OVEN_ROUND1_DATA_AUDIT.md`; no synthetic images enter this run.
- Archive SHA256: `5a7d0ef8a72d2d85d19ed5b54e2aa0633370a28455c5c8cf9ecd2a2cf7be7d16`.
- Audited rows SHA256: `d0fcd2a66d479a5928e5bd9f69edf6771da2fc8dadd26d2aa81e0e7e50763f5e`.
- Model ID: `stage8-20261020-164500-40a26d`.
- Checkpoint SHA256: `41056b3ea90d5fa718a18f48cf5182373d1010c5185cb426ba5c949b1ec1f4c5`.
- Frozen class-list SHA256: `afdc69e898ef6067b469b9f90eea3dfed84fbb76e7a5910ee1826a62f02326af`.
- Runtime: PyTorch `2.14.1+cu130`, device `cuda`, NVIDIA GeForce RTX 5080 Laptop GPU.
- Completed: `2026-11-06T07:10:51.019625+00:00`.

The manifest has 568 distinct byte hashes and 568 distinct decoded-pixel hashes. One duplicate `chicken_wing` row is excluded from the separate retained-row summary; all 569 rows receive an individual prediction record. Original ZIP directories contain 454 train, 58 val and 57 test images, but those directory names do not establish independence. The audit conservatively places the source in one uncertainty group because physical specimen/session identity is missing and similarity candidates connect groups. The old frozen MVP test set was not evaluated.

Source labels remain unverified supplied labels. Acquisition provenance and license are unknown; raw photos and per-image filenames remain in ignored local artifacts, with redistribution prohibited pending rights verification.

## Observed predictions

These counts describe the 569 supplied rows. Unsupported rows have no correctness denominator.

| Source label | Images | Vocabulary support | Observed Top-1 counts |
|---|---:|---|---|
| asparagus | 62 | Unsupported/OOD label | broccoli 55; cucumber 4; tomato 3 |
| bacon | 63 | Unsupported/OOD label | broccoli 40; tomato 9; pumpkin 5; white_radish 4; potato 3; carrot 2 |
| broccoli | 60 | Supported | broccoli 60 |
| chicken_wing | 63 | Unsupported/OOD label | potato 43; broccoli 17; cucumber 2; carrot 1 |
| croissant | 57 | Unsupported/OOD label | cucumber 29; pumpkin 19; potato 7; carrot 2 |
| french_fries | 60 | Unsupported/OOD label | broccoli 54; cucumber 6 |
| potato | 49 | Supported | potato 44; cucumber 5 |
| salmon | 60 | Unsupported/OOD label | broccoli 25; carrot 25; pumpkin 8; potato 2 |
| toast | 95 | Unsupported/OOD label | broccoli 72; cucumber 23 |

For the **two overlapping labels only**, Top-1 matches the supplied label on **104/109** images: broccoli **60/60**, potato **44/49**. The supplied label occurs in Top-3 on **109/109** images. These are raw within-source diagnostic counts, not independent accuracy estimates. All five supported-label mismatches are potato images predicted as cucumber, with potato in second place. Their Top-1 scores range from 0.5289 to 0.8726. They are inspection candidates; no labels were changed.

There are **460 unsupported-label images**, of which **88** received a Top-1 score of at least 0.90. A closed eight-class softmax can assign high confidence to a forced, unsupported answer. The retained 568-row summary has 459 unsupported rows and the same 88 high-score observations. Its overlap counts are unchanged; chicken_wing has 62 rows, with broccoli predicted on 16 of them.

The result establishes a vocabulary mismatch and records the checkpoint's behavior on this source. It does not establish a causal effect of oven lighting/background, a calibrated rejection threshold, independence of the supplied images, or readiness for deployment. Repeated or similar frames can inflate apparent agreement. No baseline weights, thresholds, transforms, or public class mappings were changed.

## Timing

The existing `PredictionAdapter.predict` supplies CUDA-synchronized internal timing for every source image. It includes preprocessing, host-to-device transfer, forward pass, softmax, Top-3 and device-to-host transfer. It excludes image decoding/validation and model loading. Only the adapter's readiness forward precedes measurement; all 569 source-image timings, including the first, are retained.

| Statistic | Internal latency |
|---|---:|
| Samples | 569 |
| Minimum | 16.7359 ms |
| Median | 22.8039 ms |
| P95, nearest rank | 33.1750 ms |
| Maximum | 115.7316 ms |

These are sequential desktop GPU measurements, not device throughput, end-to-end UI timing, or an embedded-hardware benchmark. This run completed before the new YOLO training process began, so the two task GPU workloads did not overlap.

## Reproduction and local evidence

`tools/oven_training/diagnose_baseline.py` calls the existing safe `PredictionAdapter` and `as_contract_dict`. It verifies the exact approved checkpoint SHA256 and every input image byte hash, restricts inputs to this supplied source, and refuses an existing output directory or an output outside this repository's ignored `runs/` directory. It performs no training or automatic relabeling.

The recorded invocation used the existing project virtual environment interpreter and the existing delivery checkpoint. Set `$ProjectPython` and `$BaselineCheckpoint` to those local files; personal absolute paths are intentionally kept out of this document. A rerun must choose a fresh `--out` directory.

```powershell
& $ProjectPython -u -m tools.oven_training.diagnose_baseline `
  --rows runs/oven-first-round-20261106/candidate_rows.json `
  --checkpoint $BaselineCheckpoint `
  --out runs/oven-first-round-20261106/baseline-diagnostic-NEW-RUN `
  --device cuda
```

Recorded output:

```text
BASELINE ready device=cuda model=stage8-20261020-164500-40a26d input_images=569
PROGRESS 569/569 successful=569 errors=0
EXIT_CODE=0
```

Evidence stays local under `runs/oven-first-round-20261106/`:

- `baseline-diagnostic-20261106.log`: actual progress, final counts/timing and exit code.
- `baseline-diagnostic-20261106/predictions.jsonl`: 569 per-image Top-3 scores, synchronized internal latency, relative source path, labels, hashes, original/new split and uncertainty group.
- `baseline-diagnostic-20261106/summary.json`: all-row and retained-row counts, vocabulary-scoped correctness, limitations, environment and provenance.
- `baseline-diagnostic-20261106/prediction_errors.json`: empty list; no decoding/inference errors occurred.
- `baseline-diagnostic-20261106/overlap_mismatches.json`: five supported-label mismatches for local inspection.

Prediction JSONL SHA256: `06bb20a88b463b2e8549792cc08d7dc6ec4acfa7407da29d2d7ddf0e6faa5ab3`.
Summary SHA256: `c9e12b4cff03d246d1776c1192ae0b5cff7cdfb30534a3df4e7a8042d2296db2`.

Focused verification command: `python -m unittest discover -s tests -p test_oven_round1_diagnostic.py -v`. Actual project-venv run: **4 tests, OK, exit 0**. Tests use explicitly synthetic input/report fixtures to cover changed-byte detection, source/path boundaries, missing checkpoints, output preservation, and exclusion of unsupported labels/duplicate rows from the overlap denominator. They do not measure model quality. Repository-wide verification is reported by the integrating task.
