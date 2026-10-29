# v1.1.0 数据库结构与字段说明

本项目的数据交付采用“图片文件夹 + CSV索引 + JSON版本/类别表”，不是新增SQL数据库服务。版本为v1.1.0，共10,662张入选图片；每张图片都有image_id、标签、来源、组、划分及SHA256。

```text
het-foodai/
├─ data/raw/人工筛选/                  用户输入，仅核查，不移动或删除
│  ├─ 胡萝卜/
│  ├─ 土豆/
│  ├─ 西兰花/
│  ├─ 南瓜/
│  ├─ 西红柿/
│  ├─ 白萝卜/
│  ├─ 黄瓜/
│  └─ 甜椒/
├─ data/manifest.csv                    当前默认入口，与本版清单字节一致
├─ data/processed/selected_dataset_v1.1.0/
│  ├─ train/<八类中文目录>/             8,527张
│  ├─ valid/<八类中文目录>/             1,068张
│  ├─ test/<八类中文目录>/              1,067张
│  ├─ classes.proposed.json             有序类别，仍draft
│  ├─ manifest.csv                     10,662行图片记录
│  ├─ summary.json                     数量、版本、质量检查、限制
│  ├─ provenance.csv                   入选图片到人工输入/来源的映射
│  ├─ audit/                           全量输入台账、排除原因、相似对和验证
│  ├─ review_samples/                  每类最多12张抽查拼图
│  ├─ licenses/                        已有公开来源随包说明
│  ├─ SOURCES_AND_LICENSES.md           来源许可与未核实说明
│  └─ 交付说明.md                      本版工作与核查报告
└─ data/cache/selected_dataset_v1.1.0/               本地中间结果，不提交Git
```

train、valid、test各自含同样的八类中文子目录。图像是复制生成的副本；旧处理版本仍保留，不混入本版。

| 逻辑记录 | 文件 | 关联与用途 |
| --- | --- | --- |
| 版本 | summary.json | version=v1.1.0、固定seed42、目标比例、数量与质量边界 |
| 类别 | classes.proposed.json | id从0连续到7；key、zh与manifest一致；status=draft |
| 图片索引 | manifest.csv | image_id唯一；label_id引用类别id；relative_path定位图片 |
| 输入与来源 | provenance.csv | image_id关联图片；input_path定位人工原图；source_member记录已匹配源包路径 |
| 审核台账 | audit/audit.csv | 全部10,665条输入及候选/重复/冲突/不可读等实际状态 |
| 未入选记录 | audit/excluded.csv | 原图仍保留，只说明为何未进入基础划分 |
| 近似关联 | audit/near_pairs.csv | left、right关联image_id；distance为感知哈希汉明距离 |
| 保留核对 | audit/input_preservation.json | 前后原图文件路径、数量、字节是否有变化 |
| 独立核对 | audit/independent_verification.json | 副本SHA、实体目录计数、已检测关联跨集检查 |

## manifest.csv字段

| 字段 | 含义与约束 |
| --- | --- |
| image_id | 输入相对路径生成的稳定标识；在本版清单内唯一 |
| relative_path | 相对仓库根目录的图片路径，禁止越界 |
| label_id | 0～7，引用本版本类别表，不能按目录排序重新编号 |
| class_key | 与label_id一致的英文类别键 |
| source_id | 已匹配公开来源标识，或manual_source_unresolved |
| group_id | 已知来源关联与感知相似合并后的全局组；同组不跨集合 |
| split | train、val、test；磁盘验证目录为valid，清单标记仍为val |
| license | 已匹配来源的许可声明，未知保留unknown |
| sha256 | 交付图片的完整文件SHA256，副本字节应与原图一致 |

```mermaid
erDiagram
    DATASET_VERSION ||--o{ IMAGE_RECORD : contains
    CLASS ||--o{ IMAGE_RECORD : labels
    SOURCE ||--o{ IMAGE_RECORD : attributes
    GROUP ||--o{ IMAGE_RECORD : keeps_together
    IMAGE_RECORD ||--|| PROVENANCE : traces
```

以上是文件索引的逻辑关系，未声称运行了SQL数据库。来源unknown和未知拍摄/增强父子关系仍可能存在；group_id不是已经证明独立的真实拍摄对象编号。
