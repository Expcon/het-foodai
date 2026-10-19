# 数据工程交接与下一阶段

## 本次范围

接收 `data/raw/人工筛选/` 八类人工筛选图片，自动审核，按全局关联组生成目标80/10/10划分，输出可追溯清单、质量报告和可复现脚本。一般虾类沿用ID 7、key shrimp，中文名为“虾”。

本次不修改核心训练器、推理应用或共享类别文件，不推送原始图片。

## 交付位置（以运行完成后的 summary 为准）

- 图片目录：`data/processed/selected_dataset_v1/{train,val,test}/{中文类别}/`。
- 训练入口清单：`data/manifest.csv`；版本内备份为 `data/processed/selected_dataset_v1/manifest.csv`。
- 类别提案：`data/processed/selected_dataset_v1/classes.proposed.json`，状态draft。
- 数据版本及实际数量：`data/processed/selected_dataset_v1/summary.json`。
- 来源与输入追溯：版本目录 `provenance.csv`；完整输入台账及排除原因位于 `data/cache/selected_dataset_v1/`。
- 每类少量抽查图：`data/cache/selected_dataset_v1/contact_*.jpg`。

输入图片未移动或删除。预先增强图、掩膜、重复或冲突样本仅从基础划分清单中排除，仍留在人工筛选原目录。既有增强图不参与验证和测试；正式训练可由训练器对训练原图在线增强。

划分后的文件使用稳定ID命名，通过provenance.csv追溯输入原文件。不要根据中文文件夹的排序重新分配标签，模型必须读取统一的manifest与类别表。

## 复现与检查

```powershell
.\.venv\Scripts\python.exe scripts\prepare_selected_dataset.py
.\.venv\Scripts\python.exe scripts\validate_manifest.py --manifest data\manifest.csv --classes data\processed\selected_dataset_v1\classes.proposed.json --check-files
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

为保护已划分版本，准备脚本拒绝覆盖既有输出目录或 data/manifest.csv。现有版本的检查可以直接复跑；重建版本需另建明确的版本方案，不能悄悄改动测试集。

## 进入正式训练之前

仓库 DATA_CONTRACT 要求三名成员共同确认类别含义与顺序，才创建 `configs/classes.json` 并标记 frozen。用户已明确回答“尚未共同确认，先完成数据交付”；本次保留可审核提案，暂停在正式训练之前。

共同确认后由负责人同步共享配置，再执行：

```powershell
.\.venv\Scripts\python.exe scripts\validate_manifest.py --manifest data\manifest.csv --classes configs\classes.json --check-files --require-frozen
.\.venv\Scripts\python.exe scripts\train_classifier.py --manifest data\manifest.csv --classes configs\classes.json --num-classes 8 --epochs 10 --batch-size 16 --image-size 224 --seed 42
```

以上训练命令仅为交接说明，没有代表已运行训练。训练负责人还需在项目环境确认PyTorch/torchvision与GPU兼容性；数据审核环境的Python版本不代表训练环境已验证。

模型选择使用train+val；模型选择完成后只做冻结test的正式评估，不反复用test调参。应用负责人收到真实模型后接入推理，不展示虚构结果。

## 报告边界

- 当前数据偏公开网页/市场背景，不能直接承诺烤箱场景表现。
- 稀少类别、来源单一、拍摄身份未标注等限制需在结果中保留。
- 无已检测SHA/关联组跨集，不等于已证明所有拍摄对象互相独立。
- 不为追求精确比例拆开关联组；报告实际比例与偏差。
