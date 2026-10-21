# 数据工程工作与核查报告：v1.1.0

状态：用户要求自动核查、生成并上传GitHub；本版属于数据交付，类别仍draft，未训练模型。

## 类别调整与标注方式

本轮人工筛选目录已变为八类蔬菜。继承目录名作为人工提供的类别标签，自动写入label_id、class_key与split，不使用未训练模型编造语义预测。保留ID 0胡萝卜、1土豆、2西兰花、3南瓜、4西红柿；ID 5改白萝卜、6改黄瓜、7改甜椒，替代旧鸡胸肉、鲈鱼和虾。旧版类别表和检查点不能直接用于本版。

## 划分结果

| ID | 类别 | key | 输入 | train | valid | test | 入选 | 未入选（原图保留） |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 胡萝卜 | carrot | 1,249 | 999 | 125 | 125 | 1,249 | 0 |
| 1 | 土豆 | potato | 1,457 | 1,165 | 146 | 146 | 1,457 | 0 |
| 2 | 西兰花 | broccoli | 1,496 | 1,196 | 150 | 150 | 1,496 | 0 |
| 3 | 南瓜 | pumpkin | 1,477 | 1,180 | 148 | 148 | 1,476 | 1 |
| 4 | 西红柿 | tomato | 1,386 | 1,108 | 139 | 138 | 1,385 | 1 |
| 5 | 白萝卜 | white_radish | 1,200 | 960 | 120 | 120 | 1,200 | 0 |
| 6 | 黄瓜 | cucumber | 1,200 | 960 | 120 | 120 | 1,200 | 0 |
| 7 | 甜椒 | bell_pepper | 1,200 | 959 | 120 | 120 | 1,199 | 1 |
| 合计 | 八类蔬菜 | | 10,665 | 8,527 | 1,068 | 1,067 | 10,662 | 3 |

目标train:valid:test=8:1:1，实际79.9756% / 10.0169% / 10.0075%；seed42，按全局关联组先分组再分配，不拆组凑数。

## 工作内容：输入、处理、输出、实际状态

| 步骤 | 输入 | 处理 | 输出与实际状态 |
| --- | --- | --- | --- |
| 接收与登记 | 人工筛选八类，10,665个文件 | 固定输入快照，记录SHA | 原始台账与保留核对；不删图 |
| 标签与来源 | 人工目录、已有公开数据集审计索引 | 按目录赋标签；文件SHA匹配来源许可 | 类别表、provenance及unknown记录；不推断未知授权 |
| 自动质量核查 | 全部输入 | 图像解码、尺寸、像素SHA、文件SHA、旋转/翻转规范化感知哈希 | audit.csv，非图片/不可读/短边/重复/冲突状态如实记录 |
| 去重和分组 | 完全重复、原始关联、感知相似 | 其他副本只从本版划分清单排除，原图保留；关联全局合并 | 近似候选5,266对，全局组7,035个 |
| 8:1:1划分与复制 | 入选图片 | 固定种子，整组分配，副本按ID命名 | train 8,527 / valid 1,068 / test 1,067；复制后SHA核对 |
| 数据库结构和报告 | 清单、标签表、审核记录 | 描述文件目录和CSV逻辑关系 | 工作报告、结构说明、下载与交接文档 |
| 发布与交接 | 已核查版本 | ZIP逐文件CRC/SHA校验，GitHub Release服务器摘要核对 | 真实图片作为Release附件；代码/元数据提交现有PR，不自动合并main |

## 核查证据

- 输入状态：`{"candidate": 10662, "exact_duplicate": 3}`；未入选文件仍保留在原目录。
- 原图前后路径和SHA一致：10,665个，删除0、增加0、字节变化0。
- 入选10,662张副本SHA通过，实体目录数量与manifest一致；SHA、全局组、已知来源组、检测近似对跨集合均为0。
- 来源计数：`{"kaggle_vegetable_v1": 10192, "manual_source_unresolved": 235, "roboflow_fridge_genie_v1": 235}`。来源许可未核实235张，不索取新来源、不伪造许可。
- 短边告警0；自动可读性与结构通过不等于全量语义/品种人工鉴定。
- 项目22项测试通过，命令 `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`，退出码0；日志data/cache/release_v1.1.0_tests.log。新增验证自定义类别顺序与旧类别替换选项冲突保护。
- manifest SHA256：`6bef9776c278d8efac86008a11c47386cb7d95b7164073b9bd07e84cb5cf2821`。

## 抽查发现与待复核

已查看每类12张、共96张抽查图，并放大查看黄瓜目录0225.jpg。该图主体为圆形深绿色果实，与常见长条黄瓜有差异；源数据中也位于Cucumber目录，品种或类别仍需人工确认。本版保留原图和用户目录标签，登记audit/visual_review_pending.csv，不擅自替换成其他类别。

白萝卜抽查可见白色根部样本；甜椒包含红、黄、绿果实；部分蔬菜图存在枝叶、市场背景、多实例或其他配菜。以上只是有限抽查，不证明全量标签正确，也不代表实际烤箱场景质量已经验证。

## 数据库结构与使用

训练数据恢复不依赖raw目录；若要从人工筛选重新执行完整来源匹配，需保留本地data/cache/audit_v1/images.csv与original_near_pairs.csv等来源索引。原始ZIP、人工输入与本地缓存不提交Git，Release包含本版训练所需图片和全部元数据。

详见 [结构说明](docs/DATASET_V1.1.0_STRUCTURE.md)。当前是图片文件夹+CSV/JSON索引，不新建SQL服务；版本目录为`data/processed/selected_dataset_v1.1.0/`，默认data/manifest.csv已切换为本版，与版本内manifest一致。valid是磁盘目录名；val是训练契约中的清单标记。

```powershell
.\.venv\Scripts\python.exe scripts\validate_manifest.py --manifest data\manifest.csv --classes data\processed\selected_dataset_v1.1.0\classes.proposed.json --check-files
```

## GitHub与成员交接

- [v1.1.0发布页](https://github.com/Expcon/het-foodai/releases/tag/v1.1.0)
- [完整训练图片ZIP](https://github.com/Expcon/het-foodai/releases/download/v1.1.0/het-foodai-selected-dataset-v1.1.0.zip)
- [文件清单](https://github.com/Expcon/het-foodai/releases/download/v1.1.0/processed-file-inventory.json)
- [校验文件](https://github.com/Expcon/het-foodai/releases/download/v1.1.0/SHA256SUMS.txt)

ZIP在仓库根目录解压，保留data/...路径；含本版本根manifest，会切换默认清单。GitHub自动Source code ZIP不含真实图片。旧版本原图与交付继续保留。

杨熙珂负责数据登记、QA、版本及追溯；陈谦接收本版校验后组织三名成员共同确认类别并负责EfficientNetV2-S真实训练；孙璞月同步新版ID，后续接收真实模型和预处理契约。数据交付不自动冻结类别，不宣称模型训练完成。

## 限制

近似分组不证明所有未知拍摄/增强父图关系已被发现；同来源相似背景可能造成域偏差。白萝卜、甜椒等名称继承用户目录，必须结合抽查图确认颜色、品种、生熟和混合背景是否符合任务。新版重新划分且类别映射变化，不能将旧版测试指标直接当作本版同一测试集对比。本次未做模型训练、独立烤箱域测试或准确率评测。
