# Data owner's Codex prompt — 杨熙珂

You own DATA ENGINEERING + ROBUSTNESS AUDIT for HeT FoodAI. Read AGENTS.md and docs/DATA_CONTRACT.md first. Do not edit the core trainer or demo. Operate on `feat/data-inventory`.

1. Check which of the draft eight food categories have suitable **real independent** raw food photos and licenses; propose replacements. Record each source URL, license, date, estimated originals and mismatch risks in docs/DATA_SOURCES.md. Don't invent dataset sizes or claim downloads that did not happen.
2. Implement a repeatable `scripts/audit_images.py` for unreadable photo detection, per-class/source counts, full SHA-256, near-duplicates (e.g. pHash), and human review list for doubtful labels.
3. Implement group-safe train/val/test split before any augmentation; output data/manifest.csv with fields in DATA_CONTRACT.md. Raw image files must not enter Git.
4. Run tests including one deliberately duplicated original across splits that must be rejected; present results and limitations. Ask for team sign-off before freezing eight class IDs.
5. Submit a PR with scripts, data source registry and audit log, not photos.
