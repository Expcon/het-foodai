# 当前数据交付：v1.1.0（八类蔬菜）

当前默认版本为 **v1.1.0**，10,662张，train 8,527 / valid 1,068 / test 1,067。类别：胡萝卜、土豆、西兰花、南瓜、西红柿、白萝卜、黄瓜、甜椒；ID 5～7已替代旧鸡胸肉、鲈鱼、虾，旧类别映射不可复用。

- [发布页与附件](https://github.com/Expcon/het-foodai/releases/tag/v1.1.0)
- [完整图片ZIP](https://github.com/Expcon/het-foodai/releases/download/v1.1.0/het-foodai-selected-dataset-v1.1.0.zip)
- [SHA256SUMS](https://github.com/Expcon/het-foodai/releases/download/v1.1.0/SHA256SUMS.txt)
- [工作与质量报告](../DATASET_V1.1.0_REPORT.md)
- [数据库结构与字段](DATASET_V1.1.0_STRUCTURE.md)

本版目录为`data/processed/selected_dataset_v1.1.0/{train,valid,test}/`；默认`data/manifest.csv`已切换并与版本内清单一致。清单验证集split仍为val，磁盘路径指向valid。类别保持draft；三名成员共同确认后才正式训练。

```powershell
# 新克隆或确认恢复本版后，在仓库根目录解压；已有其他版本时先核对，避免覆盖不同默认清单
Expand-Archive -LiteralPath '下载目录\het-foodai-selected-dataset-v1.1.0.zip' -DestinationPath . -Force
.\.venv\Scripts\python.exe scripts\validate_manifest.py --manifest data\manifest.csv --classes data\processed\selected_dataset_v1.1.0\classes.proposed.json --check-files
```

ZIP包括真实训练照片、全部元数据、审核台账、语义待复核表、抽查图、来源许可和结构说明，另附同版本data/manifest.csv；GitHub自动Source code ZIP不能替代图片附件。重新打包用`python scripts/package_processed_data.py --selected-version selected_dataset_v1.1.0`（项目.venv）。

本次保留全部10,665个输入文件，3个重复副本仅从划分清单排除。235张来源许可未匹配，保留unknown；黄瓜0225.jpg存在品种/类别歧义，保留标签并列待人工复核。完整边界见本版报告。

---

## 以下为v1.0.5及更早版本历史记录，不代表当前默认数据

# 当前数据交付：v1.0.5

当前使用 **v1.0.5**：10,234张，train 8,186 / valid 1,024 / test 1,024。用户已批准鲈鱼1,051张、虾1,236张全量纳入；本版与本地v5划分一致，正式名称统一为v1.0.5。旧版本继续保留。

- [发布页](https://github.com/Expcon/het-foodai/releases/tag/v1.0.5)
- [训练图片及全部元数据ZIP](https://github.com/Expcon/het-foodai/releases/download/v1.0.5/het-foodai-selected-dataset-v1.0.5.zip)
- [SHA256SUMS](https://github.com/Expcon/het-foodai/releases/download/v1.0.5/SHA256SUMS.txt)
- [逐文件清单](https://github.com/Expcon/het-foodai/releases/download/v1.0.5/processed-file-inventory.json)
- [v1.0.5核查、数量与交接报告](../DATASET_V1.0.5_REPORT.md)

下载具名ZIP，在仓库根目录解压，保留data/...结构；不要用GitHub自动生成的Source code ZIP替代图片附件。`data/manifest.csv`已切换为本版，与版本内清单一致。图片目录为`data/processed/selected_dataset_v1.0.5/{train,valid,test}/`；manifest验证集split仍为`val`，路径指向`valid`。类别提案仍为draft，不代表三名成员已共同冻结类别。

```powershell
# 已有不同版本时先核对，不直接覆盖；确认恢复本版后在仓库根目录执行
Expand-Archive -LiteralPath '下载目录\het-foodai-selected-dataset-v1.0.5.zip' -DestinationPath . -Force
.\.venv\Scripts\python.exe scripts\validate_manifest.py --manifest data\manifest.csv --classes data\processed\selected_dataset_v1.0.5\classes.proposed.json --check-files
```

重新打包：`.\.venv\Scripts\python.exe scripts\package_processed_data.py --selected-version selected_dataset_v1.0.5`。

当前仓库经本次API核查为公开仓库；下方关于私有仓库及旧数量的说明是2026-10-19历史记录。来源unknown、海鲜增强与重复、缺少完整增强父图映射的限制见本版报告；不将增强文件数当作独立实拍数。

---

## 历史交付记录：2026-10-19，含玉米和鸡翅的旧版

# 处理后数据下载与训练交接

用户已明确要求上传 `data/processed` 全部内容，供成员获取训练图片。此前PR只上传代码和文档，没有图片，本次补齐实际数据文件。

## 下载位置

[完整数据交付 Release](https://github.com/Expcon/het-foodai/releases/tag/data-processed-v1-20261019)

| 附件 | 内容 | 使用方式 |
| --- | --- | --- |
| [het-foodai-selected-dataset-v1.zip](https://github.com/Expcon/het-foodai/releases/download/data-processed-v1-20261019/het-foodai-selected-dataset-v1.zip) | 人工筛选并自动核查后的4,816张图片、8/1/1划分及完整元数据 | 成员训练使用此包 |
| [het-foodai-audit-candidates-v1.zip](https://github.com/Expcon/het-foodai/releases/download/data-processed-v1-20261019/het-foodai-audit-candidates-v1.zip) | 旧审核候选目录全部9,641张图片 | 仅供追溯，不混入正式训练 |
| [processed-file-inventory.json](https://github.com/Expcon/het-foodai/releases/download/data-processed-v1-20261019/processed-file-inventory.json) | processed目录全部文件的路径、大小、SHA256及所属附件 | 逐文件核对 |
| [SHA256SUMS.txt](https://github.com/Expcon/het-foodai/releases/download/data-processed-v1-20261019/SHA256SUMS.txt) | 两个ZIP及说明/清单的SHA256 | 核对下载包完整性 |
| [DOWNLOAD_README.md](https://github.com/Expcon/het-foodai/releases/download/data-processed-v1-20261019/DOWNLOAD_README.md) | 面向组员的恢复路径和校验步骤 | 先阅读 |

仓库当前为私有仓库，成员需登录GitHub且具备该仓库的访问权限。图片作为Release附件完整上传，仓库Git文件树保留版本清单、类别提案及统计，不把数百MB图片塞进Git历史。GitHub自动生成的Source code ZIP不包含这些图片，不能替代具名数据附件。

## 训练成员操作

1. 获取 `feat/data-inventory` 分支的代码，下载训练ZIP。
2. 在新克隆的仓库根目录解压，保留ZIP内 `data/...` 路径。包中额外包含 `data/manifest.csv`，解压后训练入口可找到全部图片。
3. 用版本类别提案执行清单检查。
4. 类别及顺序由三名成员共同确认后，再冻结共享类别文件并由模型负责人开始训练。

```powershell
# 在仓库根目录执行。-Force用于恢复本版本的准确元数据字节；
# 若已有不同版本的数据，请先核对版本，不直接覆盖。
Expand-Archive -LiteralPath '下载目录\het-foodai-selected-dataset-v1.zip' -DestinationPath . -Force
.\.venv\Scripts\python.exe scripts\validate_manifest.py --manifest data\manifest.csv --classes data\processed\selected_dataset_v1\classes.proposed.json --check-files
```

图像应位于 `data/processed/selected_dataset_v1/train`、`val`、`test` 的八类中文目录。训练/验证/测试分别3,852/482/482张。不要按文件夹排序重新编号，使用manifest与类别提案中的既有ID。

## 完整性与授权记录

本次按用户明确授权打包上传两个processed目录的所有14,467个文件，共639,587,329字节（未压缩，含元数据）。脚本逐文件比对源SHA与ZIP成员SHA，完整覆盖目录；上传后核对GitHub附件大小、状态及服务器SHA256。

本次没有上传原始ZIP、人工筛选输入目录或模型权重。processed内的许可、作者说明和provenance随训练包保存。用户明确要求整个processed目录，因此旧候选也完整提供，但用独立附件和明显用途说明隔离。

数据划分与原文件哈希保持原样，类别仍为draft；上传数据不代表类别已共同冻结。质量限制和1张鲈鱼来源待复核事项见 [DATASET_V1_REPORT.md](DATASET_V1_REPORT.md)。

本地重新打包命令：`.\.venv\Scripts\python.exe scripts\package_processed_data.py`。输出到已被Git忽略的 `data/cache/github_upload/processed_release_v1/`。

## ZIP校验值

- `het-foodai-selected-dataset-v1.zip`：282,979,423字节；SHA256 `7b43bad0351353dfb336ffc1290d31d79eb715006671decfc04053db1629e964`。
- `het-foodai-audit-candidates-v1.zip`：375,302,453字节；SHA256 `68d78e039fb3c2c5232cff685cae901203bc220196fc4a553248e1eec2540dad`。
