# Demo / deployment owner's Codex prompt — 孙璞月

You own UI, PREDICTION ADAPTER, and DEPLOYMENT RESEARCH for HeT FoodAI. Read AGENTS.md, docs/INFERENCE_CONTRACT.md, and DATA_CONTRACT.md first. Use branch `feat/app-shell`.

1. Build local Gradio image uploader (JPG/PNG), show preview, Top-3 predictions, inference latency and clear invalid/not-ready states.
2. First use `src/foodai/inference_contract.py`'s `predict_unconfigured` and display its error. Never invent fake model scores or describe a mock result as actual accuracy.
3. Build an adapter that will load a real checkpoint from the model owner when available. Check ordered class IDs/names/checkpoint hash prefix, normalize the input identically to training validation transforms, output the stable inference contract.
4. Provide local launch steps, minimum one valid-input workflow, one corrupt-image test, and one missing-checkpoint test. Record exact outputs.
5. ONNX export/performance measurement is a follow-up task; do not claim RK3506 compatibility until the real board and memory are known.
6. Deliver a PR with app code and tests; leave data and training source to other owners.
