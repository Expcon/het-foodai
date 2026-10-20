# 本地图片审核流程

## 运行

在项目目录使用项目专用环境：

```powershell
.\.venv\Scripts\python.exe scripts\audit_images.py
.\.venv\Scripts\python.exe scripts\propose_groups.py
.\.venv\Scripts\python.exe scripts\render_audit_report.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

程序默认读取 `data/raw/kaggle_vegetable/archive.zip` 和 `data/raw/roboflow_fridge_genie/Fridge Genie.v1i.coco.zip`，支持 `--kaggle`、`--roboflow` 指定其他包名。需要 Pillow、NumPy，不依赖 PyTorch。

本次新建的数据审核环境使用 Python 3.14.7、Pillow 12.3.0、NumPy 2.5.4。此环境仅验证了数据脚本，不代表已验证 CUDA 或模型训练兼容性。

## 自动处理

1. 校验 ZIP 成员路径，读取原包并记录包 SHA256。程序直接读取 ZIP，不解压来源目录；完整读取图片时检查 ZIP CRC。
2. 逐张解码全部图片，记录文件 SHA256、解码 RGB 像素哈希、宽高、灰度标准差、64 位 DCT pHash。文件字节不同而像素相同的图片也能被标记。
3. 保存三类 Kaggle 目标图片的字节副本；依据 COCO 框裁剪 Roboflow 目标，保留父图、框坐标、原标签和原 split。异常目标框记录到 issues。
4. 仅将 fish/shrimp 裁剪放入明确标记的未确认类别目录，不赋予正式鲈鱼/对虾标签。
5. 输出完全重复组，以及 pHash 汉明距离不超过 6 的候选相似对。原图间与候选样本间分别检索，包括跨来源候选。
6. 为每类生成最多 24 张、按来源覆盖的抽查图。按解码像素去重选图，索引记录每个格子的候选 ID。

不修改 ZIP，不自动删除图片，不将相似度视为标签审核通过，不生成正式训练清单或冻结类别。

## 本地输出

`data/cache/audit_v1/`：

- `summary.json`：实际数量、包哈希、许可与说明原文、重复统计、限制。
- `images.csv`：全部来源图片的逐图台账；`archive_member` 用于追溯 ZIP 内图片。
- `annotation_counts.csv`：Roboflow 各 split、各原始标签的框数与图片数。
- `human_review.csv`：目标图片和裁剪候选，`decision`、`review_note` 留空供复核。不是已通过审核的数据，也不是契约 manifest。
- `issues.csv`：解码失败、目标框异常、COCO 尺寸不符等。
- `original_exact_groups.json`：原图字节/像素完全重复组及原始划分泄漏。
- `original_near_pairs.csv`、`candidate_near_pairs.csv`：近重复复核线索；可能有误报，不自动删除或定性。
- `contact_*.jpg`、`contact_index.csv`：抽查图和追溯索引。
- `group_proposals.csv`、`group_summary.json`：跨来源保守关联组，合并父图、完全重复与相似候选边，不是正式独立组或训练划分。
- `report.html`：本地可读审核报告。

`data/processed/audit_candidates_v1/` 保存候选图副本及无损 PNG 裁剪，均尚未人工接纳。两个输出目录均已被 Git 忽略。

## 限制与后续

- 最短边小于 96 像素仅打 small 标记，不能代替清晰度与语义判断；灰度标准差也不能当成模糊度结论。
- pHash 只能提供相似线索，不能证明增强父图关系，可能漏掉不同角度、较大裁剪、翻转或相同拍摄对象。
- COCO 类别映射与框数不证明标签正确或样本独立。多个框来自同图时必须保持同组；原图重复组还需跨来源合并。
- 待人工确认标签和相似组，再选择代表样本、建立关联组、拟定分组划分。无需沿用来源 split，但必须保留审计记录。
- 共享类别文件需依照 DATA_CONTRACT 由三名成员共同确认；数据有缺口时先补齐，不用泛类强行填满。
- 类别与数据分组通过后再输出正式 `data/manifest.csv`，运行 `scripts/validate_manifest.py --check-files`，冻结测试集；训练增强由模型训练阶段执行。
