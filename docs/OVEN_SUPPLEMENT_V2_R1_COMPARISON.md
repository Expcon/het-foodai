# 四类现场补充数据：v2 / R1 冻结对照

2026-11-11；基线 `d65b00f`，工作分支 `feat/model-oven-independent-validation-audit`。实际完成只读审计、预测前资格冻结、本机 CUDA GPU（型号不公开） 成对推理、逐图CSV和联系表。**独立开发验证仍为 BLOCKED。以下数字都是目录候选标签诊断计数，不是正式准确率。** 没有训练、挑选Epoch、改标签、导出ONNX、修改现场软件或合并PR。

## 数据与资格

本机用户指定补充目录有四类各36张，共144张PNG，均为1920×1080 RGB，可完整解码，损坏0。每类三个编号子目录，各12张；合计12个**声明目录组，不等于12个独立实物**。没有JSON、CSV、登记文件、PNG文本元数据或EXIF可核实来源。文件修改时间不作为采集日期或独立性依据。

来源登记：用户提供的本地现场补充照片；实际拍摄者、原料来源及拍摄/使用授权尚未核实。许可状态unknown，**未取得公开再分发授权**；仅在本机按本次授权只读处理，原图及逐图证据不提交Git或上传云端。

| 检查 | 实测 | 解释 |
| --- | ---: | --- |
| 本批字节/解码RGB像素完全重复组 | 0 / 0 | 像素SHA包含尺寸及RGB像素 |
| 对全部1,996张旧训练图的字节/像素匹配图 | 0 / 0 | 逐一核验旧训练文件SHA；复用同字节文件的既有像素/pHash事实 |
| 对既有841张诊断审计的字节/像素匹配图 | 0 / 0 | 未匹配到不等于从未见过同一实物 |
| 对训练池pHash候选关系 | 1,495对，涉及63张新图 | 64位既有pHash，Hamming距离≤6；含70对距离0，也不能证明同一实物 |
| 本批pHash候选关系 | 293对，涉及105张 | 含12对距离0；近似关系均未人工核实，不能当作确认重复数 |
| 来源/真实实物/母体/会话/人工签署标签 | 全部缺证 | 目录只提供候选标签；视觉相容性不能替代来源及人工登记 |
| 合格开发验证 / 封存排除 / 诊断 | 0 / 0 / 144 | 没有现成已登记封存test；没有按目录1/2/3猜测dev/test |

旧训练池清单为 `runs/oven-round2-20261107/audit/combined_rows.json`，SHA256：
`d74e39a55f7e4225b4e023d7cf5762379358ac48aef95ab3b7c827378f963181`。
共1,996张有效训练图，包含1,428张第二轮图及568张第一轮有效图；第一轮候选清单569条含1条排除记录。第一轮有效SHA是该训练池子集。本批不能仅凭“新文件/新日期/没有相同SHA”证明不属于旧训练实物。旧训练池本身也缺完整实物与会话登记。

预测前冻结资格方案SHA：
`c1ae53d4e1f92cf647448166018cb6a151aec779b3ef006854e3d9ba4421ce42`。
方案绑定源文件快照、完整资格inventory及用途；程序在加载模型前重验。所有图片预定为诊断，无任何合格或封存样本。本批已经暴露两组预测，**今后不得再称为从未使用的最终测试集**。关联封存组若真实登记存在，程序会按实物/母体/会话/重复及保守近似关系连接分量排除；不会送入预测。当前未发现此类登记，不能宣称已建立最终测试。

## 冻结模型与公平输入

使用指定最终轮文件，没有按本批分数改选模型：

| 模型 | 本机项目内路径 | SHA256 |
| --- | --- | --- |
| v2 | `runs/oven-round2-20261107/yolo11n-12cls-fixed30/last.pt` | `44878014afe8381055ed2a26fffff6269982e3f5b57c2da0fbedc596127770d5` |
| R1 | `runs/oven-r1-20261111/R1/training/last.pt` | `fec9b89f29e65dc5f35976e704ebce9900a9c34a9b1c231f5adfdd93b02813cd` |

复用 `tools.oven_training.model.load_model`，`torch.load(weights_only=True)`、文件SHA、metadata、有序类别SHA及strict state_dict校验。两组分类头均为12×1280，有序类别SHA均为 `eeb5260a718f2a1bd42a4d600c8cb2e5ee290c98a5a24e38aa1822e7391827f7`。白玉米映射现有 `corn` ID9，不改变类别含义或旧八类配置。

每张原图只生成一次固定输入：EXIF转正、RGB、Pillow双线性等比缩放居中补114至224×224、ToTensor除255、NCHW、无随机增强。同一float32 CUDA tensor依次送给两组全模块eval模型，`torch.inference_mode()`，不启用AMP。144个逐图输入SHA均留存，所有批次共享输入未被修改；两组持久模型状态在预测前后逐项指纹相同。没有使用train模式或更新BN。

环境：Python3.12.10、Torch2.14.1+cu130、torchvision0.29.1+cu130、Ultralytics8.3.230、CUDA13.0、本机 CUDA GPU（型号不公开），项目既有 `.venv`；没有安装依赖或改变全局环境。

## 全十二类诊断结果

Top-1表示“预测与目录候选标签相同”；Top-3表示候选标签出现在前三项。标签没有人工签署、独立实物数全部unknown，因此不提供验证百分比或置信区间。0张的类别是**无覆盖**，不是零准确率。没有计算实物级准确率；开发验证混淆矩阵为N=0的空矩阵，诊断矩阵另存，二者不混用。

| ID / 类别 | 张数 | v2 Top-1 | R1 Top-1 | v2 Top-3 | R1 Top-3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 asparagus | 0 | — | — | — | — |
| 1 bacon | 0 | — | — | — | — |
| 2 broccoli | 0 | — | — | — | — |
| 3 chicken_wing | 0 | — | — | — | — |
| 4 croissant | 0 | — | — | — | — |
| 5 french_fries | 0 | — | — | — | — |
| 6 potato | 36 | 9 | 35 | 20 | 36 |
| 7 salmon | 36 | 28 | 29 | 36 | 35 |
| 8 toast | 0 | — | — | — | — |
| 9 corn（白玉米候选） | 36 | 0 | 15 | 11 | 27 |
| 10 sausage | 36 | 5 | 22 | 34 | 29 |
| 11 steak | 0 | — | — | — | — |
| 合计 | 144 | 42 | 101 | 101 | 127 |

逐图配对：两组均与候选相同36张，v2不一致而R1相同65张，v2相同而R1不一致6张，两组均不一致37张。R1仍有43张候选不一致，其中21张Top-1分数≥0.9；高分不是经过校准的正确概率，不能用于自动确认标签。

重点问题的直接证据：

- **土豆→可颂/三文鱼：** v2的16张可颂中R1有15张转为potato，1张仍为croissant；v2的10张salmon全部转为potato。原来9张与候选一致的没有退步。本批支持显著的诊断改善，尚未证明新实物泛化。
- **香肠→鸡翅：** v2的31张chicken_wing中18张转为sausage，13张仍为chicken_wing；同时新增1张原v2一致图转为chicken_wing。R1合计14张仍判鸡翅。Top-3候选包含数反而由34降为29，不能只报告Top-1改善。
- **三文鱼→培根：** v2的7张bacon中5张转为salmon、1张仍bacon、1张转croissant；但R1新增1张v2原判salmon的图转bacon。R1全部输出为salmon29、potato3、bacon2、corn1、croissant1。净Top-1仅+1，存在5张退步，Top-3也下降1。
- **白玉米：** v2输出toast18、french_fries9、chicken_wing9；R1输出corn15、chicken_wing16、toast5。只能说R1对本批15张输出corn，**不能确认这些属于训练从未使用的新母穗，也不能宣称已解决白玉米识别**。品种、成熟度和原料身份尚未登记。

6张退步图（5张三文鱼、1张香肠）均独立列CSV和一页联系表，保留原始路径、SHA、两组全部12类分数、Top-3及输入SHA；没有删除反例。完整预测CSV有144条、每条两组结果。错误并集108张共7页；R1不一致43张共3页。

## 背景、层位及声明目录组

读取预测前，实际视觉查看9页全图联系表，覆盖全部144张。仅记录助手观察到的背景：锡纸可见72张、黑盘可见72张，每类各18+18。观察绑定每张SHA及9页联系表SHA，另存 `scene_observations.json` 与seal。**这是AI视觉场景观察，不是人工真值、实际拍摄记录或来源证明。** 源metadata仍保留unknown，没有用观察覆盖原图/JSON。拍摄会话、烤箱来源、照明测量、层位均无法核实，层位unknown144；未从编号或图像垂直位置推断层位。

下表每格18张；独立实物数全部unknown：

| 候选类别 | 锡纸 v2→R1 Top-1一致数 | 黑盘 v2→R1 Top-1一致数 |
| --- | ---: | ---: |
| potato | 3→17 | 6→18 |
| sausage | 1→5 | 4→17 |
| salmon | 16→13 | 12→16 |
| corn | 0→11 | 0→4 |
| 总计（各72张） | 20→46 | 22→55 |

6张退步均在可见锡纸组；香肠锡纸仍有13/18不一致，白玉米黑盘仍有14/18不一致。三文鱼锡纸净退步、黑盘改善；白玉米表现方向又不同。**这些相关性不能证明“锡纸造成错误”**：真实实物/会话未核实，姿态、尺度、曝光和共同母体也可能混杂，没有做单变量场景实验。

12个声明目录组各12张，逐组计数见本地 `analysis/groups.csv`。白玉米目录1的R1一致9/12，目录2及3均3/12；视觉上段长不同，但不能据此认定来自三根不同玉米。香肠目录1/2/3分别10/12、6/12、6/12；土豆11/12、12/12、12/12；三文鱼10/12、11/12、8/12。这些是目录组描述，不是按独立实物统计。没有合格层位、会话或来源分组可供正式比较。

## 本机产物及复现

原图留在用户原始来源目录，权重仍在上述既有模型路径，不复制、不上传。逐图路径、分数、联系表和本次日志生成在Git忽略目录 `runs/oven-supplement-20261111/`。Git只纳入工具、合成测试、本报告和不含个人信息的汇总CSV。

| 路径（相对于上述本机产物目录） | 内容 |
| --- | --- |
| `audit/inventory.json`、`source_snapshot.json`、`near_training.json`、`near_peer.json` | 只读事实、候选近似关系及缺证原因 |
| `audit/pre_prediction_plan.json`、`.sha256` | 预测前用途与资格冻结 |
| `audit/missing_evidence_intake.csv` | 12个真实存在的目录组；身份、母体、会话、来源、人工签署等留空待补，不伪造ID |
| `audit/blind_contacts/`、`scene_observations.json`、`scene_observation_seal.json` | 预测前实际视觉观察与页面绑定 |
| `analysis/per_image.csv` | 完整144张原图路径/SHA、候选、资格、两个模型Top-1/Top-3/全部分数、输入SHA和独立场景观察 |
| `analysis/contacts.html` | 可打开的联系表入口；错误并集、R1不一致、6张退步 |
| `analysis/regressions.csv`、`rescues.csv`、`both_mismatch.csv` | 分别6、65、37张，不自动改标签 |
| `analysis/*_diagnostic_confusion.csv`、`*_development_confusion.csv` | 12×12候选诊断矩阵；独立开发验证为N=0，不是零准确率 |
| `analysis/groups.csv`、`class_background.csv` | 图像数量与unknown独立实物数同时记录 |
| `comparison/contracts.json`、`predictions.json`、`summary.json` | 模型/预处理/状态指纹及原始输出 |
| `comparison/summary_counts_v2.json`、`summary_schema_note.json` | 同一冻结预测重新汇总，明确分开physical_objects与independent_source_groups；原summary保留，不改预测 |
| `run_audit.py`、`summarize.py`、`verify_preservation.py`及对应日志 | 本机实际执行驱动、汇总及保全核验 |

可携带的全部十二类/背景汇总：[逐类CSV](experiments/oven_supplement_20261111_per_class.csv)、[场景CSV](experiments/oven_supplement_20261111_background.csv)。其独立实物数为unknown，不含原图或个人路径。

实际命令在项目目录、既有 `.venv` 运行：

```powershell
.venv/Scripts/python.exe runs/oven-supplement-20261111/run_audit.py
$supplementPlan = (Get-Content runs/oven-supplement-20261111/audit/pre_prediction_plan.sha256).Trim()
.venv/Scripts/python.exe -m tools.oven_training.supplement predict `
  --audit-dir runs/oven-supplement-20261111/audit `
  --out runs/oven-supplement-20261111/comparison --plan-sha $supplementPlan `
  --v2 runs/oven-round2-20261107/yolo11n-12cls-fixed30/last.pt `
  --r1 runs/oven-r1-20261111/R1/training/last.pt --device cuda:0
.venv/Scripts/python.exe runs/oven-supplement-20261111/summarize.py
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe runs/oven-supplement-20261111/verify_preservation.py
```

输出目录拒绝覆盖；重新执行请给本地驱动/CLI指定全新目录并重新做预测前资格冻结，不要删除历史证据或把事后分组冒充预注册。CLI `audit` 是基本接口；本次 `run_audit.py` 额外校验全部旧训练文件、第一轮子集、已有诊断关系和MVP封存字节SHA。MVP封存匹配只依据文件字节SHA，不能识别重新编码或同一实物的其他帧；未来封存识别仍需像素/实物/会话证据，不能单靠该检测证明无泄漏。程序不会调查现实中的身份、自动确认人工标签或自动创建封存集。完整登记可通过Python `audit(..., registrations=...)` 显式传入；目前没有简表自动转换器，需数据负责人审核交接。

## 测试及保全

- 审计、144张真实成对CUDA预测、汇总、保全命令均退出0；不以合成图代替本批预测。
- 新增10项合成单测覆盖缺证不晋级、完整声明路径、训练重合与关联封存排除、白玉米ID9、原图不变、损坏图、计划篡改、弱标签不计算验证准确率及12类完整输出；额外验证两个实物共享一个母体时实物数2、来源组数1分开记录，避免把照片/母体/实物混计。合成测试只验证软件逻辑。
- 最终 `python -m unittest discover -s tests -v`：218项，217通过、1项Windows软链接权限条件跳过、失败0，退出0（21.608秒）。日志 `unittest_release.log` 和对应 `.exitcode.txt` 留存。另有10项专项测试全部通过，退出0；早期预期失败与误用模块导入形式的失败日志也保留，没有冒充通过。
- 对比前后源字节快照：本批144、旧补充来源2,925、旧现场目录220项均相同；再次核验全部1,996训练图SHA。
- 100项保护文件（含v1/v2/A/B/R0/R1及观测checkpoint、历史ONNX、旧类别与清单）和138项基线非文档代码/软件文件的SHA全部相同。两组预测前后持久状态相同。没有触碰原模型、旧八类冻结数据/类别、现场EXE或生产UI。
- 只读代码/证据复核核对了资格绑定、逐图覆盖、全部转移计数及汇总CSV。复核发现的合格验证计数命名问题已用合成测试修正；真实推理核心不变，原144条冻结预测保留，只增加明确对象/来源组字段的新汇总。修正前后代码SHA另存schema note，未将后算汇总冒充新测量。

## 仍缺什么；下一步先补证，不训练

1. 采集者逐一确认12个编号目录究竟代表不同实物、角度还是摆放轮次；同一个土豆/香肠换方向不能增加实物数，同一母穗的切段必须关联。补真实physical_sample_id、共同parent_source_group_id和证据，不能由程序或照片序号生成身份。
2. 补实际拍摄会话、来源/采购或制作批次、烤箱/相机代号、时间、实际层位、背景及照明；不知道填unknown。由实际了解原料的人审核四类标签并记录依据、审核者和时间；AI/YOLO一致不能代替签署。
3. 与旧训练池实物/来源登记核对独立性，人工审查63张新图涉及的近训练候选关系。旧池身份缺口无法补齐时，需要数据负责人提供可复核的来源隔离证据，再协调资格规则，不能偷偷放宽门禁。
4. 本批保持已暴露诊断用途。若能补齐来源和标签，可按审核结果开展开发诊断，仍不得回称未触碰的最终测试。最终test另行预留，先定关联组和用途，再拍摄/封存，不向模型对照暴露。
5. 若本批无法补证，优先四类每类3个有来源、彼此上游独立的新实物：每类2个开发验证、1个封存测试，分开会话；每实物3层位×锡纸/黑盘共6张。计划共72张（开发48张/8实物、封存24张/4实物），不是已采集数量；共享母体/来源关系须整组分配。这只是小规模起步，不能支持宽泛泛化结论。其余8类随后补齐，不能从本批推断其性能。

当前决定：保留v2和R1两个冻结候选，**不自动选R1替换生产模型**。土豆和香肠有有用的诊断改善，但三文鱼退步反例、白玉米剩余错误及身份缺证仍然存在；独立验证资格通过前，不进行部署结论或新的训练。
