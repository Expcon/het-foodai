# Phase 2 烤箱候选数据采集与审计规范

日期：2026-10-29；预计 2026-10-30 到货。**今天没有打开相机或连接烤箱/目标板。** 工具先服务于少量冷态试采集和离线审核，不自动创建正式训练集，也不修改 Dataset v1.1.0 的类别、manifest、划分或权重。

## 已知与未知

用户提供的企业路线：YOLO、INT8、Linux、MNN、RK3506 系列、1080p 相机，扩类时保留旧类数据联合重训。任务是整图分类还是检测、YOLO 版本、相机接口/供电/驱动、确切板卡/SKU/RAM、SDK 和访问方式均 **UNKNOWN**。实际相机取帧、1080p 可用性和板端推理均 **BLOCKED**。禁止为消除 UNKNOWN 而轮询打开相机、试插未知接口、安装驱动或控制加热。

## 工具和环境

入口 `python -m tools.oven_capture --help`，仅新增工具，不改公共协议。离线部分复用 `scripts/audit_images.py` 的 SHA/pixel SHA/DCT pHash、BKTree 和 `scripts/propose_groups.py` 的 UnionFind，最终复用 `foodai.contracts.validate_manifest`。不调用任何识别模型。

已有项目环境若包含 Pillow/NumPy，即可运行离线 import/audit/review/export。可选 UVC/OpenCV 只在明确选择相机时加载。要单独准备环境，在仓库根目录执行（所有安装仅进入新环境）：

```powershell
py -3.12 -m venv runs/oven-capture-env/.venv
$py = 'runs/oven-capture-env/.venv/Scripts/python.exe'
& $py -m pip install -r tools/oven_capture/requirements.txt
& $py -m tools.oven_capture --help
```

依赖为 Pillow12.0.0、NumPy2.2.6、opencv-python4.12.0.88；本次已在独立 Phase2 环境的同版本依赖中运行离线测试。没有在实际相机上验证任何 backend。已有 Linux/厂商 SDK 环境应使用其核准项目 Python；不要把 Windows wheel 当 ARMv7 wheel。

## 本地目录和记录

CLI 强制将输出留在 `data/raw/oven_capture/`，该目录已由项目 Git 忽略；必须显式给输出路径，不写根 `data/manifest.csv`、旧 processed 或 configs。禁止自动上传或直接 `git add -f`。设备序列号、地址、原图、metadata、审核记录都只保存在本地；对外只提交脱敏摘要。

```text
data/raw/oven_capture/
  host-inventory-YYYYMMDD.json    # 被动设备信息，含敏感设备 ID，不入 Git
  cold-collection-A/
    images/<image_id>.jpg|png    # 保留文件或解码帧，不覆盖现有图片
    records/<image_id>.json      # 扩展元数据、SHA、尺寸、标签复核
    audit-before.json            # 显式命名的审计输出，不覆盖旧报告
    assignments.json            # 人工给每个 image_id 指定 split
    candidate-manifest.csv       # 独立候选交接，绝不自动冻结/拼入 MVP
```

每张记录包含 `captured_at`（带时区）、`imported_at`（工具 UTC）、camera/oven 化名、物理食材 ID 数组、session ID、人工提供的 `class_keys` 候选、摆放位置、照明、许可、原文件 basename、图像尺寸/编码、SHA256、pixel SHA、pHash、表示形式及 review 状态。

同一个物理食材换角度/位置/日期时沿用其物理 ID；同次拍摄沿用 session ID；不同实物/新会话不要复用 ID。ID 需全局唯一、稳定并脱敏，不能填 UNKNOWN。驱动未核实时明确 `UNKNOWN_NOT_PROBED`，不能假定设备索引0等于枚举的某个 PnP ID。许可未确认写 `unknown`，不意味着可以对外再分发。

## 先做离线导入（不需要相机）

复制 `tools/oven_capture/session.example.json` 到本地 collection 之外的忽略目录，**人工填写后再用**。模板中的 UNKNOWN ID 和时间占位会被拒绝。以下是格式示例，日期/标签也不是实际采集证据：

```json
{
  "captured_at": "2026-10-30T09:00:00+08:00",
  "camera_id": "camera-A",
  "camera_driver": "UNKNOWN_NOT_PROBED",
  "oven_id": "oven-A",
  "physical_sample_ids": ["food-20261030-A"],
  "session_id": "session-20261030-A",
  "class_keys": ["carrot"],
  "placement": "middle-rack/front",
  "lighting": "room-on/oven-light-off",
  "license": "unknown"
}
```

```powershell
$collection = 'data/raw/oven_capture/cold-collection-A'
& $py -m tools.oven_capture import --files '<本地已获准图片.jpg>' '<另一张.png>' --metadata '<填写的session.json>' --output $collection
& $py -m tools.oven_capture audit --collection $collection --report "$collection/audit-before.json"
```

离线导入保留 JPG/PNG **原始文件字节和编码矩阵尺寸**，不重新编码、不缩放、不从 EXIF 或文件修改时间猜拍摄时间。导入时间不能冒充拍摄时间；操作者必须给真实、有时区的拍摄时间。每文件上限64MiB，拒绝损坏/不支持编码和像素炸弹。文件批次或相机中途失败时，已完整写入的记录会保留；先 audit 检查实际数量/孤立图片，不删除源文件、不把部分完成说成整批成功。

单主食候选如 `class_keys:["carrot"]`；混合场景可记 `class_keys:["carrot","potato"]` 和所有物理 ID；空场景/尚未断言类别的候选可显式记 `class_keys:[]`、`physical_sample_ids:[]`。默认全部为 `pending`。现有八类 key 以冻结配置为准；新类别需要单独提案与批准，不修改旧配置来让导入通过。没有边界框，更不能把混合整图标签变为检测框。

## 审核、标签与防泄漏

audit 会检查损坏/不可读、哈希/尺寸/pixel SHA/pHash 与记录不符、孤立图片/坏 JSON/缺字段、字节重复、像素重复及近重复候选。相同像素的已确认标签若相互冲突，拒绝交接。不会删除、自动纠正图片或用模型补标签。

分组是在**显式提供的全部 collection 内**计算传递闭包：同物理样本 OR 同 session OR 同 SHA OR 同像素 OR pHash 候选即相连；跨相机/跨目录也不拆开。例如 A/B 同 session、B/C 同物理样本，则 A/B/C 必须同组。默认 pHash Hamming 半径6，候选需人工看；可能过度合并，也可能漏掉未知关联，不等于已证明独立实拍数。audit 可用 `--collection A B` 联合审核、用 `--near-threshold 0..16` 辅助查看；export 始终以默认6重新审计，不能用更低审计阈值绕过导出隔离。

操作者打开原图确认后，才运行：

```powershell
& $py -m tools.oven_capture review --collection $collection --image-id '<实际image_id>' --class-keys carrot --reviewer '<团队化名>'
# 明确确认空场景可传空列表：--class-keys --reviewer '<团队化名>'；仍不能导出单标签分类 manifest
```

review 记录操作者/UTC时间/人工标签来源，保留最初候选标签。空候选若要改为食材标签，必须先由操作者补齐真实 physical_sample_ids 并审核元数据，否则拒绝写入，保留原记录。缺许可信息一律归为 unknown。review 不是类别或数据集冻结，工具也不能证明键盘后的操作者真的完成了现场语义核查。审计仍需结合原图和记录签认。

首次到货 **不要建立正式训练集**。之后如团队批准独立候选交接，手工创建 `assignments.json`，覆盖所有 image_id，例如 `{"<image-id>":"train"}`，合法值只有 train/val/test/external_test。不要照抄示例来划分真实数据；先审核整组关系，保持所有关联图片在同 split。导出需要所有标签已确认且每图恰好一个类别；多类和空场景会被拒绝。

```powershell
& $py -m tools.oven_capture export --collection $collection --assignments "$collection/assignments.json" --out "$collection/candidate-manifest.csv"
& $py scripts/validate_manifest.py --manifest "$collection/candidate-manifest.csv" --classes configs/classes.json --check-files
# 存在多个 collection 时，必须联合导出；assignments 覆盖所有目录的 image_id：
& $py -m tools.oven_capture export --collection $collection 'data/raw/oven_capture/cold-collection-B' --assignments '<联合assignments.json>' --out "$collection/joint-candidate-manifest.csv"
```

export 要求显式列出当前 `data/raw/oven_capture/` 下所有带 images 或 records 目录的 collection，拒绝部分目录单独导出；任一目录的孤立图片或坏元数据也会阻止交接。此保守约束防止跨目录关联消失。空场景、多食材和未确认记录留在此 catalog 时，整个单标签交接会被阻止；本轮优先留全冷态诊断证据，后续正式训练需另行设计经人工签认的筛选/排除流程，不能为导出成功而删除现场证据。group_id 是该快照的联合组件标识；后续新增采集后应联合重新导出为新文件，**禁止拼接旧 manifest**。未放入此目录的外部/历史数据不在自动检查范围。

导出严格使用九字段：image_id、relative_path、label_id、class_key、source_id、group_id、split、license、sha256。路径为 repo-relative，source_id固定 `oven_capture`。扩展上下文仍在 sidecar；公共 schema 未改变。程序不会选择 split、不会覆盖已存在的 manifest、不会修改旧数据/原权重，也不会开始训练。与历史数据的跨版本近重复复核、来源权利核实、完整语义复核及正式冻结仍是独立人工放行步骤。

## 相机到货后的显式操作（今天未执行）

先按 [到货清单](OVEN_ARRIVAL_CHECKLIST.md)核对手册、线缆、电压和冷态隔离。不得操作加热/风机/继电器/GPIO/固件/驱动。先关闭其他占用相机的应用。被动枚举不取帧：

```powershell
& $py -m tools.oven_capture devices --out data/raw/oven_capture/device-inventory-20261030.json
```

Windows 读取已安装 camera/image 驱动信息；Linux 读取 sysfs video 设备。枚举到设备仍为 UNKNOWN：可能不是烤箱相机，不能由索引推断。没有设备/查询失败为 BLOCKED。不会自动遍历并打开设备。实际驱动与索引映射由操作者核对；厂商 SDK 接口尚待适配，使用 `FrameSource.read()->(PIL.Image, JSON-compatible details)` / `close()` 的替换接口，不能伪装成 UVC。

只有操作者已确认设备、输出与冷态后，才运行一次 probe；Windows 明确选已核准的 dshow 或 msmf，Linux选 v4l2。下面索引0只是语法示例，**不得盲试**：

```powershell
& $py -m tools.oven_capture probe --device 0 --backend dshow --output "$collection/probe-A" --cold-confirmed --width 1920 --height 1080
& $py -m tools.oven_capture capture --device 0 --backend dshow --output $collection --metadata '<填写的session.json>' --cold-confirmed --count 1 --width 1920 --height 1080
# 有限帧示例：--count 3 --interval 1；count限制1..100，不存在无限录制
```

probe 写 `probe.json`，包括请求尺寸、驱动上报尺寸/FOURCC/FPS、**真正读取帧的宽高**。不保证请求1080p被驱动接受；不支持的属性为 UNKNOWN/null，FPS仅是上报值、不是实测帧率。失败会清楚返回未就绪，不偷偷扫描其他索引或访问网络摄像头。capture 记录 host UTC 取帧完成时间，覆盖 metadata 中的候选拍摄时间；它不是传感器曝光时钟。

三种“原始”必须分开：

1. 相机输出的 MJPEG/YUY2 等流/数据包：OpenCV后端可能已解码，本工具**不保留其原始压缩字节**；FOURCC只是驱动报告。
2. 解码帧：本适配器接受 uint8 三通道 BGR，转为 RGB 并无缩放保存 PNG，记录实际解码矩阵尺寸；相机/驱动是否已裁剪或缩放仍未知。
3. 传感器 RAW：Bayer、位深、黑电平、厂商元数据等需要专门接口。当前 `sensor_raw_claimed=false`，绝不将 PNG/JPEG 称为传感器 RAW；未知/非支持布局需厂商适配，不猜测解码。

正常结束或取帧失败都会释放已选句柄。出现不明控制页面、热态、异味、异常供电、线缆/接口不符、权限/SDK不明或相机读帧失败，立即停止软件操作并按厂商安全流程处理，不通过刷机/装驱动继续试。

## 今天的验证范围与退出码

`tests/test_oven_capture.py` 用带 SYNTHETIC 标记的临时文件/假 FrameSource 验证软件；`python tests/smoke_oven_capture.py` 实际运行 import→audit→review→export→共享validator CLI，并删除自己的合成输入。它们不是相机、食物精度、1080p能力或真实数据验收证据。

正常软件命令 exit0；审计结构/损坏/元数据错误、导出前置条件未满足为 exit1；相机/可选依赖未就绪为 exit2；argparse缺参数也为exit2。audit exit0但 `REVIEW_REQUIRED` 只表示审计成功执行，不代表标签/数据已验收。硬件表只使用 PASS/FAIL/BLOCKED/UNKNOWN，所有现场行今天仍为 BLOCKED。
