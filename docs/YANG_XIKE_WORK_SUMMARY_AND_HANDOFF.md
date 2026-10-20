# 杨熙珂工作总结与成员交接

更新日期：2026-10-19。项目：HeT FoodAI 食材种类识别；当前模型方案：EfficientNetV2-S；本阶段范围为八类单一主要食材分类。

## 一、我的职责与已完成工作

我（杨熙珂）负责数据工程，包括公开数据集收集、来源与许可记录、人工筛选、自动核查、重复与关联样本处理、训练/验证/测试划分、质量分析和版本化交付。

| 工作 | 已完成内容 | 交付证据 |
| --- | --- | --- |
| 数据收集与来源审核 | 收集并核对Kaggle Vegetable Image Dataset、Roboflow Fridge Genie、A Large Scale Fish Dataset三个公开来源，记录类别覆盖、许可、引用及增强情况 | [来源登记](DATA_SOURCES.md) |
| 人工筛选 | 按玉米、土豆、西兰花、南瓜、鸡翅、鸡胸肉、鲈鱼、虾整理6,767张候选图片；虾已明确为一般虾类 | 输入台账与版本报告 |
| 自动清洗与核查 | 检查可读性、尺寸、文件SHA256、解码像素重复、相似线索和来源；区分原图、预先增强图片与分割掩膜 | `scripts/audit_images.py`、`scripts/prepare_selected_dataset.py` |
| 去重与防泄漏 | 1张完全重复副本、1,950张已确认预先增强副本不进入基础划分；保留输入原文件及处置原因；同源关联及已检测相似候选按组处理 | 排除清单、关联组和provenance |
| 数据划分 | 固定随机种子42，按关联组拟合8:1:1，生成训练、验证、测试图片目录与统一清单 | `data/manifest.csv`、`selected_dataset_v1/` |
| 质量分析与验证 | 输出各类及各来源数量、实际比例、样本不足与场景偏差；19项测试通过，清单文件和哈希校验通过 | [版本质量报告](DATASET_V1_REPORT.md) |
| GitHub数据交付 | 两个Release ZIP覆盖整个processed目录的14,467个文件；包括正式划分与旧审核候选；提供清单、校验值、来源与许可 | [完整数据发布页](https://github.com/Expcon/het-foodai/releases/tag/data-processed-v1-20261019) |

最终基础数据共4,816张。实际比例为训练79.9834%、验证10.0083%、测试10.0083%；微小偏差来自整数和分组约束，不拆开关联组凑比例。

| 类别（ID / key） | 训练 | 验证 | 测试 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 玉米（0 / corn） | 59 | 7 | 7 | 73 |
| 土豆（1 / potato） | 1,165 | 146 | 146 | 1,457 |
| 西兰花（2 / broccoli） | 1,185 | 148 | 148 | 1,481 |
| 南瓜（3 / pumpkin） | 1,180 | 148 | 148 | 1,476 |
| 鸡翅（4 / chicken_wing） | 68 | 9 | 9 | 86 |
| 鸡胸肉（5 / chicken_breast） | 62 | 8 | 8 | 78 |
| 鲈鱼（6 / sea_bass） | 41 | 5 | 5 | 51 |
| 虾（7 / shrimp） | 92 | 11 | 11 | 114 |
| 合计 | 3,852 | 482 | 482 | 4,816 |

这些是图片数量，不是已证明独立的食材个体或拍摄场次数。本次没有训练模型，没有生成模型准确率、真实checkpoint或硬件部署结果。

## 二、点名成员交接方式

成员姓名与分工：杨熙珂（数据工程）、陈谦（组长／模型训练）、孙璞月（应用／推理）。陈谦姓名由杨熙珂在本次交接中确认。

| 交接方向 | 交接材料与方式 | 接收后的操作与验收 |
| --- | --- | --- |
| **杨熙珂 → 陈谦** | 通过本仓库Release交付训练ZIP；通过同一PR交付代码、manifest、类别提案、数量与质量报告。来源、许可和provenance随训练包提供 | 陈谦下载并解压到仓库根目录，核对SHA256、4,816张图片及3,852/482/482划分，运行清单校验。三人共同确认类别后，再冻结共享类别文件、配置GPU环境并训练EfficientNetV2-S |
| **杨熙珂 → 孙璞月** | 交付同一份类别ID/名称提案、图像输入范围、数据版本与质量限制；需要真实图片做输入流程检查时，从训练集选择，不使用test进行功能调试或调参 | 孙璞月按统一ID对齐推理与界面标签；模型未交付时显示“模型未就绪”。正式模型到位后，对齐模型、类别文件和数据版本，核对输入异常处理 |
| **陈谦 → 孙璞月**（后续） | 陈谦实际完成训练后，交付真实checkpoint、冻结类别文件及哈希、预处理参数、运行环境和参考预测结果。权重通过团队约定的文件方式交付，不直接提交Git历史 | 孙璞月加载真实模型，核对参考预测、Top-1/Top-3输出，记录实际CPU/GPU延迟与内存；硬件未确认前不承诺板端性能 |
| **陈谦、孙璞月 → 杨熙珂**（后续） | 将训练或应用中发现的疑难输入，以样本ID、来源、现象和复核结论组成问题清单交回 | 杨熙珂复核标签、重复与来源，按新数据版本补充公开样本；不依据test错误反复修改同一冻结测试集 |

交接以“GitHub代码与文档 + Release真实数据 + 校验结果”完成。本文只记录责任与接收流程，没有代替成员发送邮件、聊天或确认验收。

## 三、下载路径

仓库是私有仓库，成员需登录GitHub且具有仓库访问权限。

- **[训练数据ZIP：4,816张、已划分](https://github.com/Expcon/het-foodai/releases/download/data-processed-v1-20261019/het-foodai-selected-dataset-v1.zip)**：陈谦训练和孙璞月必要的输入流程检查使用此包。
- [旧审核候选ZIP：9,641张](https://github.com/Expcon/het-foodai/releases/download/data-processed-v1-20261019/het-foodai-audit-candidates-v1.zip)：仅供来源追溯和复核，不混入正式训练。
- [全部附件与发布说明](https://github.com/Expcon/het-foodai/releases/tag/data-processed-v1-20261019)。
- [ZIP校验值](https://github.com/Expcon/het-foodai/releases/download/data-processed-v1-20261019/SHA256SUMS.txt)。
- [processed逐文件清单](https://github.com/Expcon/het-foodai/releases/download/data-processed-v1-20261019/processed-file-inventory.json)。
- [详细下载与恢复说明](PROCESSED_DATA_DOWNLOADS.md)。

训练包大小282,979,423字节；SHA256：`7b43bad0351353dfb336ffc1290d31d79eb715006671decfc04053db1629e964`。

旧候选包大小375,302,453字节；SHA256：`68d78e039fb3c2c5232cff685cae901203bc220196fc4a553248e1eec2540dad`。

两个ZIP均逐成员校验源文件SHA与ZIP CRC，上传后再次核对GitHub端大小、状态与SHA256。不要把GitHub自动生成的Source code ZIP当作图片数据包。

## 四、陈谦接收数据的具体步骤

1. 获取包含本次PR #2内容的代码。PR尚未合入时使用`feat/data-inventory`分支；合入后可使用包含合并提交的`main`。
2. 下载训练ZIP，用`Get-FileHash`核对校验值。
3. 在新克隆或已核对同版本的项目根目录解压，保留包内`data/...`路径。不要覆盖其他版本的本地数据。
4. 确认下列路径存在，运行清单校验并记录结果。
5. 接收回执记录“数据版本、包哈希、校验结果、类别确认状态”；三人共同确认后才进入正式训练。

```text
data/manifest.csv
data/processed/selected_dataset_v1/
  train/                   # 八类中文子目录，共3,852张
  val/                     # 共482张
  test/                    # 共482张
  classes.proposed.json
  manifest.csv
  summary.json
  provenance.csv
  SOURCES_AND_LICENSES.md
  licenses/
```

在项目环境中运行：

```powershell
./.venv/Scripts/python.exe scripts/validate_manifest.py --manifest data/manifest.csv --classes data/processed/selected_dataset_v1/classes.proposed.json --check-files
```

预期文件与标签检查通过，类别状态仍提示DRAFT。不要按中文目录顺序重新编号，不再次随机划分，不将旧候选或预先增强副本加入验证/测试。正式训练增强仅用于train。

## 五、剩余事项与责任

| 事项 | 负责人 | 当前状态 |
| --- | --- | --- |
| 八类含义、顺序及生熟/包装范围共同确认，随后冻结共享`configs/classes.json` | 杨熙珂、陈谦、孙璞月共同确认，由负责人同步配置 | 用户已明确尚未共同确认；本次保持draft |
| 1张来源仅标fish的鲈鱼图片复核 | 杨熙珂，必要时与陈谦共同复核 | 已登记；现有人工标签保留，来源本身不能证明物种 |
| 少数类及跨来源／烤箱场景公开数据补充 | 杨熙珂 | 玉米、禽肉、鲈鱼、虾样本较少；尚无独立烤箱external_test |
| GPU环境、真实模型训练与验证集模型选择 | 陈谦 | 未在本次数据工作中执行 |
| 真实模型推理接入、界面与性能验证 | 孙璞月 | 等待真实checkpoint；无模型时保持未就绪 |

当前未发现已检测SHA、关联组或相似候选跨集合；拍摄个体信息不完整、相似检索存在漏报可能，不能据此保证所有样本在生物个体层面独立。鲈鱼验证与测试各仅5张，小类指标易受单张图片影响。

本次PR的文档与数据交付合并，不代表三人已经确认类别，也不会启动正式训练或修改测试集。更详细的工程说明见 [DATASET_HANDOFF.md](DATASET_HANDOFF.md)，质量证据见 [DATASET_V1_REPORT.md](DATASET_V1_REPORT.md)。
