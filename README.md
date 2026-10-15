# HeT FoodAI / 和而泰食材识别

**Status: initial software scaffold, NOT a trained food recognizer.**

Course 1105, project 1 — image classification baseline first, then 8→12→16 class expansion, generalization and eventual edge deployment when hardware is confirmed.

## Team and boundaries

| Track | Owner | Initial scope |
| --- | --- | --- |
| Model & incremental learning | Group lead | PyTorch / EfficientNetV2-S training, evaluation, 8→12→16 experiments |
| Dataset & robustness | 杨熙珂 | collect & review public sources, audit images, version manifest, freeze splits |
| Application & deployment | 孙璞月 | Gradio UI, prediction wrapper, latency/memory reports, ONNX candidate |

Use **one common ordered class list** and **one frozen split**. Do not split foods among members.

## 0. First run (Windows PowerShell; no GPU packages required)

Unzip the archive and place its inner `HeT_FoodAI` folder at `D:\Projects\HeT_FoodAI`, then:

```powershell
cd D:\Projects\HeT_FoodAI
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\inspect_contract.py
```

No Conda needed. Commands do not touch CR5AF.

## 1. Before *real* training

1. Dataset owner: verify exact food categories, number of **independent** photos, availability and usage rights. Propose final eight classes.
2. All three: freeze `configs/classes.json` (copy from the draft and set `status` to `frozen` after revising if needed). Keep IDs stable when later adding classes 8–15.
3. Dataset owner: put image files into `data/raw/` (gitignored). Create `data/manifest.csv` matching the contract. Split by capture group BEFORE image augmentation and run the validator.
4. Model owner: prepare the GPU build of PyTorch in this project venv using an official Windows CUDA 12.8/13.0 wheel matched with torchvision; verify CUDA device, *then* install other requirements.
5. App owner: build UI against `docs/INFERENCE_CONTRACT.md`. Before checkpoint exists, must explicitly report not-ready.

## 2. Commands once data is ready

```powershell
.\.venv\Scripts\python.exe scripts\validate_manifest.py --manifest data\manifest.csv --classes configs\classes.json --check-files
.\.venv\Scripts\python.exe scripts\train_classifier.py --manifest data\manifest.csv --classes configs\classes.json --num-classes 8 --epochs 10 --batch-size 16 --image-size 224
```

A successful validation does not certify labels are semantically correct: manual sample review and cross-source testing are mandatory. Training downloads ImageNet weights on first use unless `--no-pretrained` is given.

For **final** test once the model has been selected without looking at test scores:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_classifier.py --checkpoint runs\STAGE_RUN_ID\best.pt --manifest data\manifest.csv --classes configs\classes.json --split test
```

Do not run this repeatedly to tune hyperparameters.

## 3. Folder map

- `configs/` tentative classes; frozen configuration will be added after team agreement.
- `src/foodai/contracts.py` shared data/schema checks; edit with coordination.
- `scripts/train_classifier.py` model owner's training CLI; `scripts/evaluate_classifier.py` final locked evaluation.
- `scripts/validate_manifest.py` split leakage checks; dataset owner extends auditing/curation.
- `app/` UI owner; `prompts/` task statements usable with Codex.
- `docs/` integration rules and task breakdown.

## Known unknowns

- The final eight food classes are NOT yet frozen.
- No dataset has been downloaded/verified into this starter.
- Actual accuracy, generalization, training time and RK3506 / alternate board performance are UNKNOWN.
- Enterprise provides no training or test data, according to the latest course discussion.

Never turn these uncertainties into claimed results.
