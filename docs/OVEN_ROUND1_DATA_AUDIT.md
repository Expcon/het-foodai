# Oven first-round ZIP audit — 2026-11-06

The supplied `food_classification_dataset.zip` contains **569 real images in 9 source-labeled classes**. It is not uniformly split 48/6/6 per class. All 569 images decoded successfully as RGB-readable PNGs, 1920×1080. No physical-item IDs or capture-session metadata were available. This audit supports a fixed-configuration **exploratory fit only**, with no validation/test accuracy or generalization claim.

## Source and scope

- Archive size: **422,805,862 bytes**; expanded member size: **476,099,561 bytes**.
- SHA-256: `5a7d0ef8a72d2d85d19ed5b54e2aa0633370a28455c5c8cf9ecd2a2cf7be7d16`.
- Members: 600 = 569 image files + 31 directories.
- Origin: user-supplied ZIP; original acquisition provenance unverified.
- License and redistribution rights: unknown; do not redistribute images pending rights verification.
- Labels come from source folders and have not been independently human-verified.
- No sidecar/text metadata files, PNG metadata keys, or EXIF tags were found.
- Numeric suffixes such as `potato_0008` are file/frame identifiers. They do **not** establish different physical potatoes, sessions, or independent samples.

The archive was read without modification. Images were extracted only under ignored `data/raw/oven_round1_20261106/5a7d0ef8a72d2d85d/`. Existing extracted bytes are checked and never overwritten. Bounds cover ZIP entry counts, expanded sizes, per-file sizes, compression ratios, unsafe paths, Windows aliases, case collisions, encryption, and symlinks.

## Measured counts

IDs below are alphabetical and experimental for this run only. They do not replace the frozen production class mapping.

| Experimental ID | Source class | Original train | Original val | Original test | Total | Unique usable pixels |
|---:|---|---:|---:|---:|---:|---:|
| 0 | asparagus | 50 | 6 | 6 | 62 | 62 |
| 1 | bacon | 50 | 6 | 7 | 63 | 63 |
| 2 | broccoli | 48 | 6 | 6 | 60 | 60 |
| 3 | chicken_wing | 49 | 7 | 7 | 63 | 62 |
| 4 | croissant | 46 | 6 | 5 | 57 | 57 |
| 5 | french_fries | 48 | 6 | 6 | 60 | 60 |
| 6 | potato | 39 | 5 | 5 | 49 | 49 |
| 7 | salmon | 48 | 6 | 6 | 60 | 60 |
| 8 | toast | 76 | 10 | 9 | 95 | 95 |
| **Total** | **9 classes** | **454** | **58** | **57** | **569** | **568** |

## Duplicate and split findings

Byte hashes and decoded RGB pixel hashes each found one two-image duplicate group: `train/chicken_wing/chicken_wing_0051.png` and `train/chicken_wing/chicken_wing_0052.png`. Both were already in train. The latter is retained in the audit rows but excluded from fitting with `exclusion_reason=duplicate_pixels`. **No exact byte/pixel duplicate group crossed the original ZIP splits.**

The reused 64-bit DCT pHash with Hamming radius ≤6 found:

- 1,929 candidate pairs, including the exact duplicate pair.
- 576 nonidentical-pixel candidate pairs across original train/val/test folders.
- 252 of those 576 across-split pairs share the source class; 324 cross class.
- 1,048 cross-class candidate pairs overall.

These are similarity candidates, not 1,929 proven duplicate images or proven physical-item matches. Visual review of a deterministic 45-image contact sheet (five per class) found a common dark tray/oven background, small central food regions, and repeated rotations/arrangements. The strong shared background helps explain cross-class pHash matches and limits pHash specificity. This visual inspection cannot identify which physical item was captured, count independent items, or validate all labels.

| Class | Across-original-split, same-class candidate pairs | Example pair | Hamming distance |
|---|---:|---|---:|
| asparagus | 15 | test/0018 ↔ train/0017 | 0 |
| bacon | 4 | train/0029 ↔ val/0025 | 4 |
| broccoli | 42 | test/0005 ↔ train/0012 | 2 |
| chicken_wing | 38 | train/0008 ↔ val/0032 | 0 |
| croissant | 22 | test/0054 ↔ train/0055 | 0 |
| french_fries | 10 | train/0047 ↔ val/0048 | 0 |
| potato | 77 | train/0008 ↔ val/0034 | 0 |
| salmon | 7 | test/0005 ↔ train/0004 | 0 |
| toast | 37 | test/0069 ↔ train/0070 | 0 |

Example suffixes abbreviate the actual `<class>_<suffix>.png` filenames. All candidate endpoints and distances are recorded in the local JSON audit.

## Conservative grouping and permitted use

Because independence cannot be established, every row has `physical_sample_id: null`. The algorithm joins all rows of the same class into an uncertainty group, then joins global exact-byte, exact-pixel, and pHash candidate connections. For this archive, the cross-class connections produce **one conservative uncertainty component**. This is **not evidence that all photos contain one physical item or belong to one verified session**.

An independent train/validation/test division cannot be justified from these data. The new ZIP directory split is retired for this exploratory run: **568 unique usable images are assigned train; one duplicate is excluded; validation and test have zero images.** No augmentation was used to create these assignments. There are therefore no groups, hashes, or candidates crossing the new fit partitions, but this train-only arrangement provides no held-out evaluation. Choose a fixed training configuration in advance; do not tune it using original ZIP val/test labels or report an accuracy/generalization score from this fit.

The old frozen MVP test set, existing manifests, `configs/`, shared contracts, README, and requirements are untouched. A later formal experiment needs owner-confirmed labels and rights, actual physical-item/session identifiers, independent specimens for each class, and a new frozen group split established before augmentation or model selection.

## Local evidence and reproduction

Generated outputs are gitignored and remain local:

- `runs/oven-first-round-20261106/data_audit.json`: archive identity, class/split counts, hashes, duplicate groups, all pHash candidates, metadata findings, grouping policy, limitations.
- `runs/oven-first-round-20261106/candidate_rows.json`: all 569 rows, including the excluded copy, repo-relative image paths, source labels, original split, experimental IDs, hashes, pHash, uncertainty group, null physical ID, and new split.
- `runs/oven-first-round-20261106/audit_contact_sheet.jpg` and `audit_contact_index.json`: local visual QA selection; not committed or uploaded.
- Candidate rows file SHA-256: `d0fcd2a66d479a5928e5bd9f69edf6771da2fc8dadd26d2aa81e0e7e50763f5e`.

Run with the existing project virtual environment. The ZIP path is supplied locally rather than recorded in committed documentation:

```powershell
python -m tools.oven_training.audit --zip <local-food_classification_dataset.zip> --output runs/oven-first-round-20261106
```

Actual execution exit code: **0**. Relevant output:

```text
ZIP audit: 500/569 images decoded
image_count: 569
original_split_counts: test=57, train=454, val=58
near_pair_count: 1929
near_cross_original_split_pair_count: 576
new_split_counts: excluded=1, train=568
uncertainty_group_count: 1
physical_ids_available: false
```

Synthetic-only unit fixtures cover pixel duplicates under different encodings, original split leakage, label conflicts, absent identity evidence, corrupt images, unsafe ZIP paths, case collisions, symlink entries, size bounds, output confinement, idempotent extraction, and refusal to overwrite modified extracted files. Initial tests failed because the new audit entry point was absent; after implementation all seven passed.

```text
python -m unittest discover -s tests -p test_oven_round1_audit.py -v
Ran 7 tests in 0.287s
OK
exit code: 0

python -m unittest discover -s tests -v
Ran 109 tests in 27.694s
OK (skipped=1)
exit code: 0
```

The skipped existing test is `test_symlinked_image_directory_cannot_escape_collection`: this Windows environment did not permit creation of the test-directory symlink. No real training or inference was performed by this data audit.
