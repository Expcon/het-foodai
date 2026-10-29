# 模型负责人交接说明（2026-10-15）

当前仍只有类别草稿，没有真实训练数据或可交付 Demo 的真实模型。以下内容是训练程序与临时合成输入的验证记录，不能解释为真实食材准确率。开发分支为用户指定的 `feature/training`。

## 审计发现与修补范围

| 发现 | 最小处理 |
| --- | --- |
| 局部 `ManifestDataset` 无法在 Windows spawn worker 中序列化 | 移到模块顶层，保留 `get_image_dataset` 调用方式、RGB 转换与原始 label ID |
| 公共 manifest 校验允许早期盘点的空 SHA，训练可能漏掉精确重复泄漏 | 训练和评测入口要求所有行都有 SHA；继续由既有校验器检查实际文件哈希与 source/group 隔离 |
| 原种子设置缺少 NumPy、显式 loader generator 与确定性算法 | 固定这些状态，分别给 train/val generator 种子；worker 初始化在模块顶层 |
| 原训练尾批为 1 时可能丢样本 | 保留全部样本；96 像素最小尺寸下完整 EfficientNetV2-S 的单张尾批已通过合成 GPU 冒烟 |
| 控制台缺少 loss，history 只在训练结束保存 | 每 epoch 记录并关闭 CSV，输出 train/val Loss、Accuracy、实际参与样本数 |
| 缺少验证混淆矩阵及逐类输出 | 每 epoch 单独 JSON，行是真实 ID、列是预测 ID，含每类召回与 macro recall |
| 检查点类别表、架构、尺寸与参数兼容性校验不足 | 模型负责人的加载函数核对 frozen 有序前缀、哈希、模型名、输入尺寸、预处理版本，严格加载 state_dict |
| 直接覆盖 best.pt 可能丢失上一个可用检查点 | 同目录临时文件写完再替换，写失败保留旧文件 |
| 评测只记录 manifest 哈希，无法阻止训练后改 split 再评测 | 要求评测 manifest 与检查点记录的完整文件哈希一致，两个 held-out split 均适用 |
| Demo 尚无真实模型适配器 | 提供模型加载函数和下方调用约定；Demo 模块由孙璞月实现，继续保持 not-ready |

未更改 `configs/`、`README.md`、requirements、公共 contracts、数据工程脚本或 Demo 模块；公共 manifest 列、类别格式和预测 JSON 协议均保持原样。`vision.py` 只修复 Dataset 的序列化方式，现有预处理与模型工厂签名不变。

## 真实训练的前置条件与命令

必须先由三人确认并冻结正式 `configs/classes.json`，交付来源、许可、语义标签已复核的真实图片与版本化 manifest。所有行需有经文件验证的 SHA256，先按原图/拍摄组划分再增强，训练和验证覆盖所有当前阶段类别；test/external_test 在训练前冻结。近重复、标签正确性及来源使用权仍由数据负责人复核，程序不能代替人工审核。

在项目根目录执行，使用既有 `.venv`：

```powershell
.\.venv\Scripts\python.exe scripts\validate_manifest.py --manifest data\manifest.csv --classes configs\classes.json --check-files
.\.venv\Scripts\python.exe scripts\train_classifier.py --manifest data\manifest.csv --classes configs\classes.json --num-classes 8 --epochs 10 --batch-size 16 --image-size 224 --seed 42 --num-workers 0
```

训练默认用 torchvision 的 ImageNet 预训练权重，首次可能下载模型权重；本次所有冒烟均显式 `--no-pretrained`，未下载权重或数据。真实首次运行前需准备好权重缓存或确认可下载。日志记录 seed、输入尺寸、优化器默认值、学习率、batch、worker、AMP、Python/PyTorch/torchvision/CUDA/cuDNN、GPU、训练脚本及 vision 源码 SHA、类别与 manifest SHA。运行目录时间戳使用 UTC。

训练只解码 train/val 图片用于前向、loss、梯度和选择模型；公共校验器会读取所有 split 的文件字节做存在性与 SHA 完整性检查，不计算 held-out 预测或指标。`best.pt` 只按最高 val accuracy 选择，相同分数保留更早 epoch。

选定模型和超参数后，仅做约定的最终评测：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_classifier.py --checkpoint runs\STAGE_RUN_ID\best.pt --manifest data\manifest.csv --classes configs\classes.json --split test
```

评测必须恢复训练时原样冻结的 manifest，CSV 重排、换行变化或追加行都会改变哈希而被拒绝。后续 12/16 类阶段应保留各阶段的原始 manifest；8 类检查点可以配合后来追加类别的 frozen 配置，但评测仍使用其原 manifest。新增外部集需在相应实验开始前冻结；本次没有增加更换评测集的绕过选项。

## 输出与 Demo 接入

- `run_info.json`：类别有序表及哈希、逐类 train/val 数、数据版本依据、训练配置和环境。
- `history.csv`：每个已完成 epoch 的 train/val loss、accuracy、实际样本数，逐 epoch 落盘。
- `val_epoch_0001.json` 等：对应 epoch 的验证混淆矩阵、逐类 count/recall、macro recall 和整体指标。与 `best.pt` 的 `epoch` 对照即可找到被选中模型的验证报告。
- `best.pt`：保留 `model_state`、`epoch`、`val_accuracy`、`metadata`；它用于评测与推理，没有 optimizer/scaler/RNG 状态，不支持等价断点续训。
- `test_evaluation.json` / `external_test_evaluation.json`：所选 held-out split 的 loss、accuracy、逐类召回、混淆矩阵、检查点 SHA、类别 SHA、manifest SHA 与执行环境。

模型加载与预处理可由 Demo 负责人直接复用（正常项目导入环境中）：

```python
import torch
from foodai.contracts import load_classes
from foodai.checkpoint import load_classifier_checkpoint
from foodai.vision import get_transforms

config = load_classes("configs/classes.json", require_frozen=True)
model, info = load_classifier_checkpoint("runs/REAL_RUN/best.pt", config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # 加载函数已经设置 eval()
transform = get_transforms(info["input_size"], train=False)
batch = transform(decoded_image.convert("RGB")).unsqueeze(0).to(device)
with torch.inference_mode():
    scores, ids = model(batch).softmax(dim=1).topk(min(3, info["stage"]))
```

ID 对应 `info["class_list"][id]` 的 `id/key/zh`，返回 JSON 中用既有 `class_id/class_key/label_zh/score` 字段，并带 `info["model_id"]` 与 `info["class_list_sha256"]`。预处理固定为 RGB、Resize(round(input_size × 1.14))、CenterCrop、ToTensor、ImageNet mean/std；不得换成另一个默认 transform。推理使用 FP32，与验证和最终评测一致。延迟测量、图片验证、错误处理和 JSON 包装属于 Demo 负责人，本次未实现或测量。

加载仅使用 CPU `weights_only=True` 与严格 state_dict，不回退 unrestricted pickle；仍只使用项目自己产生且可信的检查点，不能把此选项视为任意文件的安全沙箱。参见 [PyTorch 序列化说明](https://docs.pytorch.org/docs/2.14/notes/serialization.html)。

## 8→12→16 扩展边界

保留 `--num-classes 8|12|16` 和 `get_model(num_classes, pretrained=...)`。活动类别始终取正式配置的有序前缀，追加 ID 不改变旧 ID、名称或含义；每阶段 train/val 要覆盖全部活动类别。加载允许 8 类权重与后来的 12/16 类配置共存，但不会把 8 类权重静默装入 12/16 类分类头。

当前基线是累计数据联合训练。未来顺序微调可在模型负责人的模块中显式迁移 backbone 和旧分类头行，再初始化新行；本次未实现迁移、蒸馏、回放或自动续训。必须单独定义旧数据访问条件与实验协议后才能称为特定增量学习算法。

## 验证与限制

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tests\smoke_training.py
```

单元测试的轻量网络只验证训练编排与错误路径；独立冒烟使用完整 EfficientNetV2-S、RTX GPU、AMP、2 epoch、96 像素临时随机图片（train=17，val/test/external_test 各 8），检查 optimizer 确实更新、所有尾批样本参与、日志和矩阵、CPU 安全加载后 GPU RGB Top-3、8 类与 16 类配置前缀兼容，以及 12/16 类分类头构造。合成 classes 只存在于临时目录，不冻结项目草稿；结束后删除全部临时图片和权重。

确定性设置以相同代码、种子、数据、worker 配置和软硬件环境为复现边界，不保证跨 PyTorch/CUDA/设备逐位一致，且可能降低速度；不支持的非确定性算子会报错。参见 [PyTorch 复现说明](https://docs.pytorch.org/docs/2.14/notes/randomness.html)。本次未验证真实数据高精度、224 像素真实训练显存、预训练迁移效果、灾难性遗忘、Demo 端到端接入或部署性能。

2026-10-15 实际验证记录（Python 3.12.10、torch 2.14.1+cu130、torchvision 0.29.1+cu130、NVIDIA GeForce RTX 5080 Laptop GPU）：

| 实际命令 | 退出码与结果 |
| --- | --- |
| `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` | 0；`Ran 17 tests in 18.164s` / `OK`（原有 8 项及新增 9 项） |
| `.\.venv\Scripts\python.exe tests\smoke_training.py` | 0；完整 GPU AMP 流程通过，train=17/val=8，2 epoch，两种 held-out CLI 均 exit=0，临时图片与模型已删除 |
| `.\.venv\Scripts\python.exe scripts\train_classifier.py --classes configs\classes.draft.json --no-pretrained` | 2；`BLOCKED: Class list is DRAFT; obtain team approval before formal training`，符合预期 |
| `git diff --check` | 0；没有空白错误（Git 提示本地 LF/CRLF 转换，不影响执行） |

完整合成冒烟控制台记录保留在本地 gitignored 的 `runs/verification-20261015/synthetic-smoke.log`。该文件中的 Loss/Accuracy 仅用于检查程序输出，不能用作真实食材性能报告。
