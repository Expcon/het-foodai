# DATA CONTRACT (v0.1) — read before downloads or training

## Ordered classes
- `configs/classes.draft.json` contains **unconfirmed candidates** for eight classes.
- The data owner proposes `configs/classes.json` and all 3 members approve before `status: frozen`.
- IDs must be contiguous `0..N-1` and are immutable once the first model is trained. Add new classes at the end for 12/16 stages.
- If the meaning of a category changes (e.g., raw vs cooked, whole vs cut), increment dataset version and repeat quality review.

## Manifest CSV fields (all required columns)

| Field | Definition |
| --- | --- |
| `image_id` | globally unique ID (stable across reorganizations) |
| `relative_path` | repo-relative path such as `data/raw/kaggle/potato/img001.jpg`; local images are gitignored |
| `label_id` | integer matching frozen class ID |
| `class_key` | must match the same class ID's key, e.g. `potato` |
| `source_id` | dataset/repository/institution ID; cite origin and rights in `docs/DATA_SOURCES.md` |
| `group_id` | original item, photoshoot session, original image ID **before augmentation**; same source+group may not cross splits |
| `split` | `train`, `val`, `test`, `external_test` |
| `license` | SPDX-style label or `unknown` pending verification; unknown sources cannot be redistributed by default |
| `sha256` | full SHA256 of image bytes (64 lowercase hex chars) when calculated; empty only during early inventory |

## Core policy

- Freeze test/external_test partition **before training** and version it; choose hyperparameters on val only.
- No identical file hash or known near-duplicate/augmentation parent may cross splits.
- When data has existing train/val/test directories, do not assume they are clean; audit before use.
- Different datasets have different backgrounds: accuracy should additionally be tested on a separate outside source if available.
- Public availability does not equal permission for enterprise redistribution; track licensing explicitly.
- Raw images, identifiable people and enterprise photos MUST NOT be pushed to a public repo.
- Before final accuracy reporting, record sample counts, class balance, source balance, excluded groups, and any label uncertainty.

## Validation

`python scripts/validate_manifest.py --manifest data/manifest.csv --classes configs/classes.json --check-files`

The validator checks schema, class mapping, file existence (opt-in), SHA duplicates across splits, and grouped split leakage. Dataset owner's work additionally includes near-duplicate detection (pHash), semantic label review and source documentation.
