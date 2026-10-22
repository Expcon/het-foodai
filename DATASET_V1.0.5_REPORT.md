# 初始训练数据集核查与审批报告（2026-10-20）

版本：`v1.0.5`（目录 `selected_dataset_v1.0.5`）。状态：**用户已批准本次数据交付并要求上传GitHub；共享类别仍未共同冻结，未启动训练。**

## 本次变更与交付范围

按用户最新人工筛选目录生成八类：胡萝卜、土豆、西兰花、南瓜、西红柿、鸡胸肉、鲈鱼、虾。玉米改为胡萝卜，鸡翅改为西红柿；新增鸡胸肉和虾参与本次核查。一般虾类不限定为对虾。按用户最新明确要求，鲈鱼和虾目录全部图片入选，包括已知预增强图及重复副本。其余六类沿用基础核查和重复代表保留策略。

全部原图保持原路径、名称和字节；训练图片通过复制生成，没有删除、移动或修改原始图片。旧版本目录保留。本报告所指的是 v1.0.5；旧 v4 仍保留，但不再是本轮交付版本。

## 海鲜全量纳入专项核对

| 类别 | 原始文件 | 入选文件 | 排除 | 来源标记为预增强 | 同像素重复额外副本（已保留） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 鲈鱼 | 1,051 | 1,051 | 0 | 1,000 | 0 |
| 虾 | 1,236 | 1,236 | 0 | 854 | 103 |

预增强数量按来源索引优先角色统计；同一文件有多个来源匹配时详见 provenance.csv。文件数不是独立拍摄对象数；目前没有可靠的完整增强父图映射，无法给出海鲜独立样本总数。所有已检测到的同像素重复副本均在同一集合，检测到的近似组也不跨集合。

## 抽查与版本采用说明

v4发现的17条图像抽查待确认记录已更新为本版本split与group，保存在 audit/visual_review_pending.csv。每类12张新版抽查图保存在 review_samples/。本版按用户要求全量纳入海鲜，不将拼图、生熟或包装现象擅自视为错标。

本次正式交付名称为 v1.0.5，内容沿用已审核的本地v5划分，只修改版本目录与清单路径，不重新随机划分。v4、v5及其他旧版本均保留；根 data/manifest.csv 已统一切换为 v1.0.5。

## 数据数量与划分

| ID | 类别 / key | 输入 | train | valid | test | 入选合计 | 仅从基础划分排除 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 胡萝卜 / carrot | 1,249 | 999 | 125 | 125 | 1,249 | 0 |
| 1 | 土豆 / potato | 1,457 | 1,165 | 146 | 146 | 1,457 | 0 |
| 2 | 西兰花 / broccoli | 1,481 | 1,185 | 148 | 148 | 1,481 | 0 |
| 3 | 南瓜 / pumpkin | 1,477 | 1,180 | 148 | 148 | 1,476 | 1 |
| 4 | 西红柿 / tomato | 1,375 | 1,100 | 137 | 137 | 1,374 | 1 |
| 5 | 鸡胸肉 / chicken_breast | 910 | 728 | 91 | 91 | 910 | 0 |
| 6 | 鲈鱼 / sea_bass | 1,051 | 841 | 105 | 105 | 1,051 | 0 |
| 7 | 虾 / shrimp | 1,236 | 988 | 124 | 124 | 1,236 | 0 |
| 合计 | 八类 | 10,236 | 8,186 | 1,024 | 1,024 | 10,234 | 2 |

目标比例 train:valid:test=8:1:1，实际为 79.9883% / 10.0059% / 10.0059%。固定随机种子42，先按全局关联组划分，允许整数和整组约束带来的少量偏差，不拆分近似关联组凑数。

## 自动核查结果

- 输入前后核对全部 10,236 个文件的路径和SHA256：新增0、删除0、字节变化0。
- 状态计数：`{"candidate": 10234, "exact_duplicate": 2}`。
- 鲈鱼与虾不再因预先增强或重复而排除；重复像素和已检测近似图归在同一集合。其他类别的完全重复副本仅从划分清单排除，仍保留原图；具体处置见 audit/audit.csv 和 audit/excluded.csv。
- 入选 10,234 张图片逐张核对复制字节SHA256、路径、标签与划分；独立核查通过。
- 当前检测 10,682 对感知近似候选；归并后的全局关联组 6,362 个。SHA、全局组、已知来源原组及已检测近似对跨集合均为0。
- 来源许可未自动核实的入选图片 1,337 张：`{"carrot": 49, "tomato": 180, "chicken_breast": 832, "shrimp": 276}`。按用户选择不再索取来源，保留图片并明确记录 unknown，不伪造出处或授权。
- 图像质量短边告警 0；自动可读性不等于人工语义、部位或物种鉴定通过。
- 项目21项测试通过，退出码0；测试覆盖原图保留、根清单不覆盖、valid路径、旧版本保护，以及海鲜重复副本保留与跨类别冲突检查。日志：data/cache/release_v1.0.5_tests.log。

## 使用路径与训练接口

本地目录：`data/processed/selected_dataset_v1.0.5/`。

```text
selected_dataset_v1.0.5/
  train/八类中文目录/图片
  valid/八类中文目录/图片
  test/八类中文目录/图片
  manifest.csv
  classes.proposed.json
  summary.json
  provenance.csv
  audit/
  review_samples/
  licenses/
```

磁盘目录按用户要求使用 valid；manifest中的验证集split仍为 val，relative_path指向valid下的真实图片，与当前按清单读取数据的项目接口兼容。不应由通用ImageFolder按中文目录排序重新分配ID。

根目录data/manifest.csv与本版本manifest.csv字节一致，可直接作为默认训练入口清单。类别表仍为draft，正式训练前仍需三名成员共同确认。

```powershell
.\.venv\Scripts\python.exe scripts\validate_manifest.py --manifest data\processed\selected_dataset_v1.0.5\manifest.csv --classes data\processed\selected_dataset_v1.0.5\classes.proposed.json --check-files
```

清单SHA256：`3939dc63c59ac16d0e67a5fec08f6207b2963ace9e237798c52ea92b1a730144`。

复现命令（仅用于新的、尚不存在的版本目录；不要覆盖当前审核结果）：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_selected_dataset.py --version selected_dataset_v6 --replace-corn-with-carrot --replace-wing-with-tomato --validation-folder valid --keep-root-manifest --include-all-seafood
```

## 训练前需要了解的限制

- 类别ID 0已改为胡萝卜，ID 4已改为西红柿；旧玉米/鸡翅模型和旧类别表不可直接复用。其余ID保持既有顺序。本版本类别表仍为draft，三名成员共同确认后才冻结并正式训练。
- 新数据重新分组与划分，测试集属于这一新版本；不同版本的测试指标不能直接作为同一测试集上的提升对比。
- 来源未核实图片的拍摄组及增强家族可能不完整；近似检测无法证明没有所有语义泄漏。本版按用户要求纳入全部海鲜增强图；已按精确重复与旋转/翻转规范化感知近似合并，但缺少完整增强父图映射，不保证所有变体关系已被发现。验证与测试中增强图片不能等同于新增独立实拍样本，指标可能偏乐观。
- 鲈鱼中来自泛fish来源的图片仍需人工复核物种；肉类部位、生熟、混合配菜、背景与实际烤箱域也需人工确认。没有独立烤箱域测试，也没有模型准确率或训练效果结论。
- 类别数量不均衡，小类的测试样本少；本阶段没有新生成增强图片或额外复制来凑数；海鲜已存在的增强图片按用户要求全部保留。后续新生成的在线训练增强只能用于训练部分。来源许可unknown不表示获得公开再分发授权。

## 审批与交接

杨熙珂已审批本版数据交付并明确要求上传GitHub；八类及ID、划分、保留原图和来源unknown记录见上文。陈谦负责接收通过后的版本、与团队确认类别并开展真实训练；孙璞月对齐新版类别映射和后续推理。

用户已批准本版并要求上传GitHub，数据文件使用Release附件交付，代码、清单和说明补充到现有PR #2；不执行main合并。数据交付审批与三名成员共同冻结类别是两个步骤。

## GitHub下载与成员接收

- [v1.0.5 数据发布页](https://github.com/Expcon/het-foodai/releases/tag/v1.0.5)
- [完整训练数据ZIP](https://github.com/Expcon/het-foodai/releases/download/v1.0.5/het-foodai-selected-dataset-v1.0.5.zip)
- [文件清单与校验说明](https://github.com/Expcon/het-foodai/releases/download/v1.0.5/processed-file-inventory.json)

ZIP含 data/processed/selected_dataset_v1.0.5/ 下的全部图片及元数据，并包含同版本 data/manifest.csv。在仓库根目录解压，保留data/...路径。解压后根清单恢复为本版；若已有其他版本，请先核对版本。

陈谦接收图片、manifest和类别提案，核对SHA和数量后与团队共同确认类别再训练；孙璞月接收新版ID映射，ID 0=胡萝卜、ID 4=西红柿；杨熙珂负责后续数据问题与版本追溯。GitHub仓库在发布前核查为公开仓库。
