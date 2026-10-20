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
