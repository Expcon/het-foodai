# Oven second-round data audit — 2026-11-07

The read-only audit decoded all **1,428 new images** and re-decoded the **569 first-round images**. All 1,997 are readable at 1920×1080. The combined exploratory training candidate set contains **1,996 images** after excluding one exact first-round duplicate. There is **no defensible formal validation or test holdout** in these records.

This report contains aggregate evidence only. Original pictures, per-image records, historical paths, capture identifiers, metadata hashes and contact sheets remain in ignored local directories. The audit does not train a model or alter the source folder, frozen MVP data, shared contracts, requirements or class configuration.

## Scope and source preservation

Only the approved reorganized source directory and the already available first-round candidate list/images were read. The source inventory, organizer verification JSON and organizer README were inspected. Their claims about earlier ZIPs/original folders were not independently verified because those locations were outside this audit scope.

| Check | Observed result |
| --- | ---: |
| New image files | 1,428 |
| Indexed metadata files in class folders | 1,428 |
| CSV records | 2,856 |
| Indexed byte hashes or sizes that disagree | 0 |
| Unindexed files inside class folders | 0 |
| Declared image total and per-class counts | Match |
| Same-name image sidecars | 1,427 |
| Missing same-name sidecars | 1 bacon image |
| Additional class-folder JSON | 1 aggregate photo list |
| All JSON files, including root verification file | 1,429 |
| Non-image files hashed before and after the audit | 1,431 |
| Metadata changed during audit | 0 |
| JSON parse errors | 0 |
| Image/sidecar SHA or dimension mismatches | 0 |

The missing bacon sidecar is documented by the organizer; no substitute was fabricated. The aggregate photo list is not an independent image annotation. Its historical absolute paths were neither followed nor rewritten. Every non-image file's current SHA-256 is preserved in the local provenance output.

New images comprise 1,154 JPEG and 274 PNG files. All 569 first-round files are PNG. No EXIF tags were found in the new images. Five index source prefixes account for 645, 356, 251, 97 and 79 new images; prefixes are acquisition/organization clues, not evidence of independent specimens.

Origin is a user-supplied oven capture collection. Original acquisition ownership, license and redistribution permission remain **unknown**. No new redistribution authorization is established by organizer hash agreement or by this audit.

## Ordered exploratory classes

IDs 0–8 preserve the first-round oven checkpoint order. Corn, sausage and steak are appended at IDs 9–11. The folder 牛排 maps directly to `steak`; it is not silently mapped to another food class. This mapping is scoped to the oven experiment and does not replace the formal vegetable mapping in `configs/classes.json`.

| ID | Folder | Key | New | Round 1 | Retained combined |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | 芦笋 | asparagus | 112 | 62 | 174 |
| 1 | 培根 | bacon | 97 | 63 | 160 |
| 2 | 西兰花 | broccoli | 119 | 60 | 179 |
| 3 | 鸡翅 | chicken_wing | 129 | 63 | 191 |
| 4 | 可颂 | croissant | 110 | 57 | 167 |
| 5 | 薯条 | french_fries | 122 | 60 | 182 |
| 6 | 土豆 | potato | 114 | 49 | 163 |
| 7 | 三文鱼 | salmon | 93 | 60 | 153 |
| 8 | 吐司 | toast | 104 | 95 | 199 |
| 9 | 玉米 | corn | 252 | 0 | 252 |
| 10 | 香肠 | sausage | 97 | 0 | 97 |
| 11 | 牛排 | steak | 79 | 0 | 79 |
| **Total** | | | **1,428** | **569** | **1,996** |

## Labels and metadata reliability

All **1,427 sidecars** have `label_status=pending_manual_review` and `camera_context.human_label_confirmed=false`. Their unreviewed candidate class is `pumpkin` in 644 records, `toast` in 81, and null in 702. The additional image without a sidecar also has no candidate class. All 644 `pumpkin` candidates disagree with the reorganized class; these stale fields are not used as labels.

The organizer explicitly states that all **23 photo mid/photo up images were individually viewed and assigned to toast**. Only those 23 rows receive `trusted_label=true` with `human_reviewed_from_organizer` provenance. Their original pending sidecars are preserved unchanged. This is organizer review, not an independent formal test set. The remaining 1,405 new rows and all 569 old rows keep `trusted_label=false` and `candidate_directory` status. Representative organizer review does not imply every image was reviewed.

Ten fries images have current filenames indicating a high level, while their embedded filenames/sample prefixes indicate a mid level. Their sidecar SHA-256 values match the current image bytes, and the CSV also matches. These are recorded as **historical filename/metadata warnings**, not damaged images. They remain exploratory directory candidates, with **shelf=unknown** because the level evidence conflicts. Neither original metadata nor current filenames were changed.

| Scene field | Available evidence |
| --- | --- |
| Session | 4 source session strings on 1,427 sidecars; missing on 1 image |
| Physical sample | 86 identifier strings, all scene prefixes; zero verified independent specimen count |
| Oven identity | Unknown for all 1,428; a camera alias is not an oven ID |
| Foil | 328 explicit `with_foil`, 317 explicit `without_foil`, 783 unknown |
| Shelf | 454 high, 453 low, 473 mid, 11 up, 37 unknown |
| Viewing angle | Unknown for all 1,428 |
| Lighting | Source says oven-light for 1,427; unknown for 1 |

Foil values come only from explicit tokens; ambiguous `have`/`no` tokens do not establish foil presence. Shelf tokens remain unverified scene annotations. Lighting is a source description, not a measured/calibrated condition. Raw values and evidence accompany every scene value locally.

## Duplicates, perceptual similarity and grouping

| Scope | Exact-byte duplicate groups | Identical-pixel duplicate groups | pHash ≤6 pairs | Cross-class pHash pairs |
| --- | ---: | ---: | ---: | ---: |
| New–new | 0 | 0 | 1,560 | 532 |
| New–round 1 | 0 | 0 | 1,803 | 1,602 |
| Round 1–round 1 | 1 | 1 | 1,929 | 1,048 |

The first-round duplicate group contains two chicken-wing images; one copy is retained. All new–new and new–old perceptual matches have different decoded pixels. Of the old–old perceptual pairs, 1,928 have different pixels and one is the exact duplicate pair.

An audit-only montage of 45 class/scene representatives was inspected. A separate montage of six distance-zero cross-class pairs was inspected: broccoli/chicken-wing, potato/broccoli and potato/chicken-wing within the new set, and sausage/potato, potato/chicken-wing and broccoli/chicken-wing across rounds. These samples show visibly different foods on similar dark oven trays. Thus even pHash distance **zero** is not proof of duplicate images or identical physical food. These visual checks did not promote any additional image to a human-confirmed label.

The actual union-find graph joins same-session evidence, session-scoped physical strings, identical bytes/pixels, pHash candidate pairs and conservative unknown same-class identity across old/new sources. It contains:

| Graph edge reason | Edges considered |
| --- | ---: |
| Same source session | 1,423 |
| Same session plus physical string | 1,341 |
| Same SHA-256 | 1 |
| Same decoded pixels | 1 |
| pHash candidate | 5,292 |
| Same class with unverified independence | 1,985 |

These edges produce **one conservative uncertainty component of 1,997 images**. The component is not a claim that every image depicts one object, nor a specimen count. There is no verified independence on which to divide it into train/validation/test. The audit assigns only `train` or `excluded`; pHash similarity alone never excludes an image. Formal generalization accuracy requires newly captured independent groups with reviewed labels and provenance before freezing a holdout.

## Local artifacts and row contract

All output is under ignored `runs/oven-round2-20261107/audit/`:

- `new_rows.json`: 1,428 new image rows.
- `combined_rows.json`: all 1,997 rows, including exclusion/duplicate provenance; 1,996 train candidates.
- `data_audit.json`: aggregate audit and local duplicate membership.
- `near_pairs.json`: pHash candidate graph edges and scopes.
- `source_metadata_hashes.json`: all 1,431 non-image file hashes before/after the audit.
- `representatives_45.jpg` and its index JSON: local audit thumbnails.
- `phash_cross_class_review.jpg` and its index JSON: six inspected cross-class pairs.
- `audit_run.log` and `unittest_all.log`: actual execution logs.

Each row contains a unique `image_id`, `source` (`new` or `round1`), `relative_path`, `class_key`, `class_id`, byte/pixel SHA-256, pHash, dimensions/format/status, `trusted_label`, `label_status`, `label_evidence`, and `group_id`. New paths are relative to the externally supplied input root; old paths are repository-relative. No source path is inferred from historical absolute metadata paths.

`physical_sample_id`, `session_id` and `oven_id` contain source strings or `unknown`, with separate evidence fields. `physical_identity_verified` remains false. `foil`, `shelf`, `angle` and `lighting` are objects with `value`, `raw` and `evidence`; downstream summaries must group on `value`. `source_collection` preserves the index's collection prefix. `new_split`, `exclusion_reason`, `duplicate_of`, `integrity_errors` and `metadata_warnings` distinguish retained candidates, exact duplicates, errors and nonfatal historical metadata discrepancies.

The definitive combined-row SHA-256 is `d74e39a55f7e4225b4e023d7cf5762379358ac48aef95ab3b7c827378f963181`.

## Verification

The supplied source root is passed locally; no personal filesystem paths are stored in this report.

```powershell
.venv/Scripts/python.exe -m tools.oven_training.round2_audit --input-root $approvedSourceRoot --contacts
.venv/Scripts/python.exe -m unittest discover -s tests -p test_oven_round2_audit.py -v
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Actual final audit exit code: **0**. Relevant output: `New image audit: 1428/1428 decoded`, combined image status `ok: 1997`, and combined splits `train: 1996, excluded: 1`.

Targeted audit tests: **8 passed**, exit code **0**. Fixtures are labeled synthetic and cover append-only IDs, pending sidecars, organizer review provenance, ambiguous/conflicting scene tokens, conservative session/class/duplicate unions, conflicting exact-pixel labels, traversal rejection, real image decoding, historical filename warnings, and unchanged source bytes.

Full suite: **162 tests in 25.771 seconds, OK (skipped=1)**, exit code **0**. This is a test run, not model training or an accuracy evaluation. Further application changes require their own final verification.
