# 烤箱到货冷态检查与数据链路记录

准备日期：**2026-10-29**；预计到货：2026-10-30（计划，未观察）。关联 [EDGE_DEPLOYMENT_PREP.md](EDGE_DEPLOYMENT_PREP.md)、[DATA_CONTRACT.md](DATA_CONTRACT.md)、[INFERENCE_CONTRACT.md](INFERENCE_CONTRACT.md)。

**本页尚未执行。所有实际硬件验收项为 BLOCKED；未提供规格为 UNKNOWN。** 本次没有连接/启动板卡，没有打开摄像头，没有加热、刷固件或安装驱动。此清单不构成硬件操作授权；后续只由具备授权的人按企业手册操作。先冷态，不触及加热、风机、继电器、GPIO 或其他控制接口。

## 1. 填写规则与交付规格

状态只使用 `PASS / FAIL / BLOCKED / UNKNOWN`：PASS 必须有观察者、时间和证据；FAIL 写实际不符合项；BLOCKED 写前置条件；UNKNOWN 用于尚不清楚的规格/需求。不得把“官方支持”写为“现场通过”，不得因表已填完而自动 PASS。

| 字段 | 当前值 | 来源 / 以后填写 |
| --- | --- | --- |
| 实际到货时间、包装/整机型号 | UNKNOWN | ____ |
| 板卡厂商/型号/修订、精确 RK3506 SKU | UNKNOWN | ____ |
| 标称 RAM、实际 MemTotal/MemAvailable、存储余量 | UNKNOWN | ____ |
| 厂商手册/SDK/镜像版本、Linux/内核、libc/ABI | UNKNOWN | ____ |
| 摄像头型号、接口、供电、像素格式、实际分辨率/FPS | UNKNOWN | ____ |
| 控制与计算部分能否隔离、冷态操作步骤 | UNKNOWN | ____ |
| 允许的本地访问方式/操作范围 | UNKNOWN | ____；账户、主机地址等不入库 |
| 单主食分类还是多目标检测；单/多物体交付要求 | UNKNOWN | ____；三人签认，不从“YOLO”推断 |
| YOLO/MNN 精确版本、输入尺寸、checkpoint | UNKNOWN | ____ |
| 性能/内存/误差阈值和验收时长 | UNKNOWN | ____；测量前确认 |

## 2. 三人最小分工

| 负责人 | 到货当天最小产出 |
| --- | --- |
| 模型/训练负责人（lead） | 确认 cls/det、类别映射、选用模型/输入尺寸；只接收分组隔离后的新旧数据，提供 FP32 参考与之后的 INT8 对齐记录 |
| 杨熙珂（数据/QA） | 记录相机/冷态场景、来源与许可；少量真实图像人工标签、session/physical sample/group、SHA256、重复与质量复核 |
| 孙璞月（推理/UI/部署） | 核对板卡/摄像头接口与手册，记录可用 Linux/SDK/ABI；获准后完成只读系统信息、离线推理和测量；无模型明确 not-ready |

三人先一起确认“整图一个主食”还是“同一画面多个目标”。如需检测，由三人协调数据框标注和新的输出契约，不能把现场多食材照片强行当作单标签训练样本。

## 3. 冷态先行：按顺序更新实际验收行

证据保存到项目忽略的本地数据目录。表中只写脱敏摘要、相对记录位置和团队别名，不提交企业原图、序列号、地址、账户或令牌。以下每行须人工改状态，不能根据下一行成功倒推前一行 PASS。

| ID | 现场动作 / 判据 | 负责人 | 状态 | 观察时间 / 证据 / 阻塞或失败原因 |
| --- | --- | --- | --- | --- |
| H01 | 实际收到设备，按交付单核对可见外观和附件；不拆外壳 | 三人 | BLOCKED | 未到场观察；____ |
| H02 | 企业手册、许可及冷态操作范围已获得；操作者确认不会进入加热/控制流程 | lead | BLOCKED | 缺交付资料与现场授权；____ |
| H03 | 冷态、加热保持关闭；确认计算/相机测试可与加热及控制隔离 | 孙璞月 | BLOCKED | 未观察隔离条件；不能隔离则停止；____ |
| H04 | 确认板卡型号/SKU/RAM，摄像头接口/电压/线缆方向；与手册一致 | 孙璞月 | BLOCKED | 精确规格 UNKNOWN；不试插；____ |
| H05 | 按已确认手册由授权人员连接计算板及供电；无损伤/异常 | 孙璞月 | BLOCKED | H02–H04 未通过；____ |
| H06 | 按手册冷态连接相机到已确认接口；不改驱动、固件或设备设置 | 杨熙珂 + 孙璞月 | BLOCKED | H02–H05 未通过；____ |
| H07 | 使用企业许可入口观察板卡启动与 Linux 信息；只读核对架构、内存、存储和 SDK/ABI | 孙璞月 | BLOCKED | 无连接/访问日志；____ |
| H08 | 记录摄像头被动枚举结果，核对正确设备；枚举不等于成功取帧 | 孙璞月 | BLOCKED | 无实物枚举；____ |
| H09 | 三人签认 cls/det、单/多食材、场景范围和采集权限 | 三人 | BLOCKED | 需求 UNKNOWN；____ |
| H10 | 获准后显式选择设备/后端，小数量冷态拍照；记录实际尺寸/格式，不假定 1080p 请求已生效 | 杨熙珂 | BLOCKED | H02–H09 未通过；____ |
| H11 | 复核少量真实原图可读、食材可见、无隐私/非授权内容；保留失败与拒绝样本原因 | 杨熙珂 | BLOCKED | 没有现场图片；____ |
| H12 | 每张图有真实来源/许可、session、物理样本标识、SHA256；图像/组不跨 split | 杨熙珂 | BLOCKED | 无现场 collection/audit；____ |
| H13 | 人工复核标签；混合场景可保留研究记录，分类 export 必须拒绝多类别 | lead + 杨熙珂 | BLOCKED | 无现场标签复核；____ |
| H14 | 检查与旧数据的重复、近重复候选、同物理样本和同拍摄 session 的全局连通组，再划分 | 杨熙珂 | BLOCKED | 缺数据及审计；____ |
| H15 | 真 checkpoint、类别顺序、FP32/INT8 来源与数值对齐资料已齐；没有则 UI 显示 model not ready | lead | BLOCKED | Phase 2 模型与任务未冻结；____ |
| H16 | ARMv7 Linux 构建/依赖检查通过；不得用 x86 或 Android 库替代 | 孙璞月 | BLOCKED | Linux/SDK/目标 ABI 未确认；____ |
| H17 | 板端少量离线真实图推理，前后处理与参考一致，错误/超时清晰退出 | 孙璞月 + lead | BLOCKED | 无实板运行日志；____ |
| H18 | 分别记录 FP32/INT8 的加载、首次、稳定推理、端到端、CPU、RSS/peak RSS、完整包大小 | 孙璞月 | BLOCKED | 无实板测量；阈值 UNKNOWN；____ |
| H19 | 获准后再检查 camera-to-result 链路；冷态且无硬件控制，实际分辨率/FPS、掉帧/超时均记录 | 三人 | BLOCKED | 离线板端链路未验收；____ |
| H20 | 关闭采集程序、释放摄像头，按厂商手册结束测试；脱敏交接未决项 | 三人 | BLOCKED | 本次未开始现场测试；____ |

## 4. 最少采集步骤与本地命令模板

以下命令是到货后给授权操作者的模板，**今天未运行**。使用项目 `.venv`，每次先看当前 CLI `--help`；命令退出 0 也不能替代 H01–H20 的人工观察。原图和设备元数据保持在已忽略目录，示例中的路径须替换成本地已批准输入。

1. 不接触未知硬件。先填规格与 H01–H09；同一物理食材换角度、同一次 session 都要有稳定分组标识。少量照片覆盖单主食、空场景、反光/遮挡等实际遇到的条件，数量按现场权限记录，不补造样本。
2. 可先被动枚举；使用文件导入也能测试数据链路，不必打开相机：

   ```powershell
   python -m tools.oven_capture devices --out data/raw/oven_capture/devices.json
   python -m tools.oven_capture import --files <approved-files...> --metadata <session.json> --output data/raw/oven_capture/<collection>
   python -m tools.oven_capture audit --collection data/raw/oven_capture/<collection> --report data/raw/oven_capture/<collection>/audit.json
   ```

3. H02–H09 全部有证据、相机打开已获准后，才使用 CLI 的 `probe` / `capture`；两者都会打开选定摄像头，均必须带 `--cold-confirmed`，由操作者现场确认冷态后填写。必须显式写设备、后端、输出目录；Windows 选已确认的 `dshow` 或 `msmf`，Linux 选 `v4l2`，设备索引须按枚举与现场核对。禁止无人确认地轮询/打开多个设备。`probe` 不接收 metadata，向 `--output` 目录生成 `probe.json`；`capture` 必须额外提供操作者填写的 `--metadata <session.json>`。一次先采 1 张；`--width 1920 --height 1080` 是请求值，须检查实际图像尺寸。命令形态：

   ```text
   python -m tools.oven_capture probe --device <confirmed-index> --backend <dshow|msmf|v4l2> --output data/raw/oven_capture/<probe-directory> --cold-confirmed --width 1920 --height 1080
   python -m tools.oven_capture capture --device <confirmed-index> --backend <dshow|msmf|v4l2> --output data/raw/oven_capture/<collection> --cold-confirmed --metadata <session.json> --count 1 --width 1920 --height 1080
   ```

   `import` 与 `capture` 的 `session.json` 由操作者按实际情况填写。下列是占位模板，不是现场记录；所有占位值均须替换。`captured_at` 使用带时区的 ISO8601 时间；`camera_id`、`oven_id`、`session_id` 及每个物理样本 ID 使用全局稳定化名，同一实体或拍摄 session 跨 collection 复用，不能使用序列号、路径或 UNKNOWN。`physical_sample_ids` 列出画面内全部物理样本；`class_keys` 为 frozen classes 中不重复的 key 数组。空场景/未断言类别可明确用 `class_keys:[]`，空场景允许无物理 ID，但不能导出单标签分类 manifest。`placement`、`lighting` 为非空单行实际描述；`license` 明确填写许可，尚未核实则填 `unknown`。

   ```json
   {
     "captured_at": "YYYY-MM-DDTHH:MM:SS+08:00",
     "camera_id": "REPLACE_WITH_GLOBAL_CAMERA_ALIAS",
     "oven_id": "REPLACE_WITH_GLOBAL_OVEN_ALIAS",
     "session_id": "REPLACE_WITH_GLOBAL_SESSION_ALIAS",
     "physical_sample_ids": ["REPLACE_WITH_GLOBAL_PHYSICAL_SAMPLE_ALIAS"],
     "class_keys": ["REPLACE_WITH_FROZEN_CLASS_KEY"],
     "placement": "替换为实际摆放描述",
     "lighting": "替换为实际光照描述",
     "license": "unknown"
   }
   ```

   `import` 保留操作者提供的 `captured_at`；`capture` 最终保存主机 UTC 取帧完成时间，并覆盖该字段，不将其声称为传感器曝光时间。

4. 杨熙珂逐图复核，标签先 pending，不能把模型输出当真值。复核语法示例 `review --collection <collection> --image-id <id> --class-keys carrot [other...] --reviewer <TEAM_ALIAS>`；`carrot` 是参数示例，实际用 frozen classes 的 key。多类别照片保留记录，但不能导出当前单标签分类 manifest。
5. 全局合并相同 session、相同物理样本、重复 SHA 和近重复候选，再由数据负责人填写 `image_id -> split` assignments；augmentation 必须晚于 split。最后用 `export --collection <collection-A> <collection-B> ... --assignments <assignments.json> --out <collection-A>/candidate-manifest.csv`，必须包含本地采集目录下全部带 images 或 records 的 collection；部分目录独立导出被拒绝。新增数据后联合重新导出，不拼接旧 manifest；历史/外部数据另行跨版本审计。保存 audit/export 原始日志与退出码。冻结 test/external_test 不能进入重训、校准或方案选择。
6. 三人记录“哪些实物步骤通过、哪些仍阻塞”和对应证据。新旧数据只有在来源/许可、标签、组隔离审核通过后才能联合重训。链路烟测不是正式精度、校准集或性能验收。

## 5. 立即停止条件

- 发现损伤、受潮、异常气味/温升、接头/供电/线缆方向不确定：停止连接，由企业人员处理。
- 无法确认冷态、无法隔离加热/控制部分，或系统要求控制硬件才能继续：停止本次测试。
- 下一步要求刷固件、安装驱动/系统、修改内核/设备树、提权、接入账户/云服务：保留阻塞，交企业及负责人确认范围；本清单不授权这些动作。
- 设备身份/权限与预期不符、摄像头可能采到人员或非授权画面：不打开或立即结束采集，不提交原始画面。
- task/类别/预处理不明、缺真模型、输出不匹配、内存不足/崩溃/超时：停止推理并保留日志，不伪造预测、不用冻结测试集调参。

## 6. 现场交接填写区

现场日期/团队别名：____；已观察 PASS：____；FAIL 与原因：____；仍 BLOCKED：____；仍 UNKNOWN：____。

本地证据相对位置与 SHA256 索引：____；原图来源/许可/分组记录：____；实际执行命令、退出码和日志：____。

模型负责人签认：____；杨熙珂签认：____；孙璞月签认：____。只有证据具备时更新对应状态，不提前填写性能或准确率数字。
