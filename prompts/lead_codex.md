# Lead / model owner's Codex prompt

You are working on HeT FoodAI. I own **model training + incremental learning** only. First inspect README.md, AGENTS.md, docs/DATA_CONTRACT.md, the frozen/draft class status, and tests. Do not download data or overwrite teammates' files.

1. Audit/extend scripts/train_classifier.py and scripts/evaluate_classifier.py. Focus on correct 8-class baseline reproducibility, training/val separation, sample count visibility, class-ordered checkpoint, GPU mixed precision, and actual logs.
2. In isolated `.venv`, confirm torchvision/PyTorch pairing and that NVIDIA RTX 5080 is CUDA-available. Do not install globally.
3. Run unit tests; include exact commands/results. Without frozen classes and real image manifest, stop before real training and mark it BLOCKED, not PASS.
4. Later implement 8→12→16 class cumulative full-data retraining and sequential fine-tune baselines; do not falsely call full-data joint retraining a replay-constrained setting.
5. Prepare a PR on `feat/model-baseline`. Explain each relevant changed line and list validation/test leakage protections.
