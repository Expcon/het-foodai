# 公开数据来源登记

审核人：杨熙珂；网页核查日期：2026-10-19。

状态更新：用户已将两个 ZIP 下载到约定目录，已开始本地逐图审核。以下网页结论保留来源证据；实际结果以本地审核报告及 `data/cache/audit_v1/summary.json` 为准。图片数不等于独立原图数或已接纳数量。

## kaggle_vegetable_v1

- 地址：https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset
- 可读数据卡证据：https://www.kaggle.com/dsv/2965251 （指向版本 1）。原始地址正文抓取为空，搜索索引中的同一发布者数据卡可读。
- 发布者 M Israk Ahmed；页面声明 CC BY-SA 4.0，并要求引用 Ahmed、Mamun、Asif 的 2021 年论文《DCNN-based vegetable image classification using transfer learning: A comparative study》。下载后保存随包许可及引用文件，核对与网页一致性。
- 数据卡声明 15 类、21,000 张、每类 1,400 张、224×224 JPG；列出 train/test/validation 为 15,000/3,000/3,000，同时文字写 70%/15%/15%，两种表述不完全一致。实际以下载后清点为准。
- 本项目可候选使用 Potato、Broccoli、Pumpkin；其公布的类别列表没有 Corn。其他类别不纳入八类训练。
- 独立原图、增强父图、近重复、各类可用量、文件完整性及原始划分泄漏：未审核。
- 不直接沿用原始划分作为本项目冻结划分。保留来源原始 split 元数据供审计。

## roboflow_fridge_genie

- 地址：https://universe.roboflow.com/ece-capstone-fridgegenie/fridge-genie
- 发布者 ECE Capstone FridgeGenie；页面声明 CC BY 4.0。
- 页面显示目标检测任务、9,855 张图片、163 类、1 个数据版本。此数量为项目页面展示值，不是已确认的导出版本数量、目标类别数量或独立原图数量。
- 页面可见候选标签 corn、potato/Potato、broccoli、pumpkin、chicken wing、chicken breast/chicken_breast，以及泛类 fish、chicken、shrimp。
- 版本详情页面读取失败；实际版本号、增强和预处理设置、类别数量及每类实例数需随下载包核查。不要推断所有项目图片都存在于可下载版本。
- 不可把多食材整图直接当作单标签图片；需保留框标注，审核后裁剪。裁剪样本必须追溯到父图，并和同一父图的其他裁剪处于相同 split。
- fish 不能自动映射 sea_bass；chicken 不能自动映射鸡翅或鸡胸肉；shrimp 不能仅凭通用名称证明为对虾。相关图片暂时排除在已接受样本之外，待来源证据或人工复核。
- 同义标签合并需检查导出时的实际类别名；官方文档说明导出可能规范化类别名，不能使用原始数字 ID 直接对应本项目 ID。
- 按来源所声明许可登记；多来源汇集项目仍需检查随包溯源及许可说明，当前不是逐图权属审计。

## 八类缺口表

| ID | 本项目类别 | Kaggle | Roboflow 页面标签 | 当前结论 |
| --- | --- | --- | --- | --- |
| 0 | corn / 玉米 | 不在公布列表 | corn | 有候选；框与数量待下载审核 |
| 1 | potato / 土豆 | Potato | potato / Potato | 有候选；两来源之间也需去重 |
| 2 | broccoli / 西兰花 | Broccoli | broccoli | 有候选；两来源之间也需去重 |
| 3 | pumpkin / 南瓜 | Pumpkin | pumpkin | 有候选；两来源之间也需去重 |
| 4 | chicken_wing / 鸡翅 | 无 | chicken wing | 有候选；部位、烹饪状态及数量待审 |
| 5 | chicken_breast / 鸡胸肉 | 无 | chicken breast / chicken_breast | 有候选；部位、烹饪状态及数量待审 |
| 6 | sea_bass / 鲈鱼 | 无 | 仅确认看到 fish 泛类 | 缺少已确认的专属来源；不能将 fish 强制标为鲈鱼 |
| 7 | shrimp / 对虾 | 无 | shrimp 泛类 | 有相关候选；对虾语义与数量未确认 |

以上八类尚未完成语义与独立性验收，不能据此声称八类训练数据已齐全。

## 本地接入更新

- `data/raw/kaggle_vegetable/archive.zip`：560,031,432 字节，ZIP 中有 21,000 个图片成员。
- `data/raw/roboflow_fridge_genie/Fridge Genie.v1i.coco.zip`：570,879,097 字节，包含 9,855 张图片、3 份 COCO JSON、2 份说明文件。
- Roboflow 随包说明确认版本 v1，2026-10-19 导出，自动方向校正后拉伸到 640×640，该导出版本未额外增强；不代表上游汇集来源没有增强或重复。
- Roboflow 原始 train/valid/test 分别为 7,046/1,642/1,167 张。COCO 每份类别表有 164 个条目，其中包含 groceries 上级类别；不能将表条目数直接当作 164 个实际食材类。
- 随包 README 和 COCO licenses 均声明 CC BY 4.0；Kaggle ZIP 仅含图片，没有随包许可文件，保留上述发布者数据卡许可证据。
- 完全重复、可读性和框异常结果见后续本地审核报告。操作流程及复现命令见 [DATA_PIPELINE.md](DATA_PIPELINE.md)。

## 接入与划分原则

### 新增补缺候选：large_scale_fish（未下载、未接纳）

- 作者仓库：https://github.com/DiclehanOguzhan/A-Large-Scale-Fish-Dataset
- 作者给出的下载入口：https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset
- 作者明确列出 sea bass 与 shrimp，故作为鲈鱼缺口的候选；不能凭此确认对虾物种，也需核对 sea bass 是否符合项目鲈鱼范围。
- 作者说明每类有1,000张经过翻转/旋转的RGB图，以及相应GT掩膜；不能把掩膜当分类食材图片，也不能把1,000张增强图当1,000张独立原图。
- 本地更新：已收到 `archive (1).zip`，随包 `license.txt` 标明 CC BY 4.0，已保留来源声明与作者README。非增强候选中Sea Bass、Shrimp各50张。本次基础划分只保留可匹配非增强候选的图片，不把1,000张增强图当独立原图。
- 推荐完整ZIP放入 `data/raw/large_scale_fish/`，保留README和所有目录；接入时追踪增强父图，无法可靠分组时不随机打散进入冻结评测。
- `audit_images.py` 保留前两个来源的首轮审核；`prepare_selected_dataset.py` 已对人工筛选结果接入三个来源并生成划分。最新数量与质量结论见 [DATASET_V1_REPORT.md](DATASET_V1_REPORT.md)。

1. 下载包、许可与原始目录保留；不覆盖、不上传原始照片。
2. 在裁剪前记录父图与来源；跨来源计算完全重复及近重复，关联组再做划分。
3. 检测图的多个框不是多个独立拍摄样本。来源不同也不自动代表相互独立。
4. 暂议 train/val/test = 80/10/10，实际比例服从类别覆盖与组完整性，记录随机种子和偏差。尚未冻结。
5. 独立外部测试只在溯源及跨来源去重后建立；不因平台不同直接认定为外部独立数据。
6. 按 DATA_CONTRACT.md 输出清单；裁剪坐标、父图、原始标签、原始 split、复核状态另存审计台账，不擅自改变共享清单接口。
7. 类别定义与全体成员共同确认后才冻结 classes.json；未通过审核的样本不进入正式 manifest。

下载方式见 [DATA_DOWNLOAD_GUIDE.md](DATA_DOWNLOAD_GUIDE.md)；类别草案见 [CLASS_DEFINITIONS_PROPOSAL.md](CLASS_DEFINITIONS_PROPOSAL.md)。下载后补充包版本、包 SHA256、实际数量、独立组数、重复统计、抽查记录及最终接纳结论。
