# 最新交接：v1.1.0 八类蔬菜

当前默认manifest与图片目录已切换至selected_dataset_v1.1.0，10,662张，train/valid/test为8,527/1,068/1,067。全部10,665个输入保留，3个重复副本只在清单中排除。ID 0～4不变，5白萝卜、6黄瓜、7甜椒；不能复用v1.0.5的鸡胸肉/鲈鱼/虾ID含义。

[工作与QA报告](../DATASET_V1.1.0_REPORT.md)｜[数据库结构](DATASET_V1.1.0_STRUCTURE.md)｜[下载和恢复](PROCESSED_DATA_DOWNLOADS.md)。类别仍draft；有限语义抽查存在待复核项，不等同于全量人工验标。

杨熙珂维护来源、质量与版本；陈谦校验本版图片、SHA和manifest，组织三名成员共同确认类别后开展真实训练；孙璞月同步新ID与后续推理契约。本次没有训练模型或自动合并main。

---

## 以下为旧版交接历史

# 最新交接版本：v1.0.5

用户已批准v1.0.5数据交付和GitHub上传。当前默认`data/manifest.csv`指向`selected_dataset_v1.0.5`，10,234张，train/valid/test为8,186/1,024/1,024。磁盘使用valid、清单使用val；类别仍draft，未正式训练。

本轮ID 0=胡萝卜、ID 4=西红柿，替代旧玉米和鸡翅；其余ID保持。鲈鱼1,051张、虾1,236张全量入选，增强和重复图保留并按检测到的关联分组，不能据此宣称完全无增强泄漏。

[下载与恢复](PROCESSED_DATA_DOWNLOADS.md)｜[完整核查及审批报告](../DATASET_V1.0.5_REPORT.md)。陈谦接收本版并准备训练，孙璞月对齐类别映射，杨熙珂维护数据版本；三名成员共同确认类别后再冻结共享类别文件。

---

## 下文为v1历史交接记录，不代表当前默认版本

# 数据工程交接与下一阶段

按成员点名的工作总结、接收步骤与下载入口见 [杨熙珂工作总结与成员交接](YANG_XIKE_WORK_SUMMARY_AND_HANDOFF.md)：杨熙珂负责数据，陈谦负责模型训练，孙璞月负责应用与推理。

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
