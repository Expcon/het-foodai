# Phase 2 工程交付与证据（2026-10-29）

已交付可运行的离线导入、原尺寸图像/元数据存储、重复与分组审核、人工确认标签、九字段候选 manifest 交接，以及可替换的显式设备 OpenCV 适配器。桌面 CPU 上完成两种官方轻量 YOLO 试件的 ONNX→MNN 转换和真实推理。**没有打开摄像头、连接烤箱/板卡、开展正式 YOLO 训练或完成 RK3506 部署。**

分支 `feature/phase2-oven-edge-foundation` 从 PR #6 的 `integration/mvp-v1.0-delivery`（`f77ced3`）建立，使用独立 worktree；没有合并 main，没有重写成员历史，没有发布 Release。原 main 和 MVP worktree 检查均无未提交修改。旧 Gradio 服务未被本轮停止。

## 改动及入口

| 文件 | 用途 |
|---|---|
| `tools/__init__.py` | 新工具命名空间 |
| `tools/oven_capture/{__init__,__main__,core,camera}.py` | 离线 import/audit/review/export、完整本地 catalog 联合防泄漏、被动设备清单、显式冷态有限采集 |
| `tools/oven_capture/requirements.txt`、`session.example.json` | 独立依赖、必须人工填写的 metadata 模板 |
| `tools/edge_export/{__init__,offline_guard,smoke}.py`、`setup.ps1`、`requirements-cpu.lock.txt` | 隔离环境、固定官方来源及哈希、真实导出/比较、阻断 SDK 自动联网/安装 |
| `tools/edge_runtime/{__init__,mnn_cpu}.py` | 不导入 Torch/Ultralytics 的最小 MNN CPU 张量推理 |
| `tests/test_oven_capture.py`、`smoke_oven_capture.py` | 临时 SYNTHETIC 输入、人工审核的模拟操作、真实 CLI 链路；不构成食材/相机精度证据 |
| `tests/test_edge_export.py`、`test_edge_runtime.py` | 输出路径、离线 guard、任务/张量/概率契约边界测试 |
| `.gitignore` | 忽略 `.mnn`、外置 `.mnn.weight`、Ultralytics 根目录意外缓存；原有 raw/runs/权重忽略继续有效 |
| [采集规范](PHASE2_OVEN_CAPTURE_PROTOCOL.md) | 完整操作、元数据、RAW 边界、退出码和审核限制 |
| [YOLO/MNN 实测](YOLO_MNN_EXPORT_REPORT.md) | 分类/检测选择、真实导出日志、误差、大小、许可及版本 |
| [部署准备](EDGE_DEPLOYMENT_PREP.md) | 官方 RK3506/RAM、MNN/Linux ARMv7、量化和资源测量方法 |
| [到货清单](OVEN_ARRIVAL_CHECKLIST.md) | 三人角色及20个 BLOCKED 现场验收项 |
| `docs/phase2-evidence/20261029/`、`docs/superpowers/plans/2026-10-29-phase2-foundation.md` | 脱敏实际日志、哈希、环境意外及修复记录、执行台账 |

未改 README、共享 requirements、公共 contracts、MVP 训练/推理/UI、冻结类别及数据清单。没有提交原图、模型、张量、虚拟环境、真实设备标识或个人路径。

## 实际执行及结果

命令均在独立 Phase2 worktree 根目录执行。`$projectPython` 表示已有项目 `.venv` 的绝对解释器；`$edgePython` 表示新建 `runs/phase2-edge/.venv/Scripts/python.exe` 的绝对解释器，不替换全局 Python 或原项目依赖。

| 命令 / 检查 | 实际结果 | 退出码 |
|---|---|---:|
| `$projectPython -m unittest discover -s tests -v`，修改前 | 68/68 PASS，37.797s | 0 |
| 同命令，最终修改后 | **99 项：98 PASS，1 SKIP，0 FAIL，16.727s**；跳过原因是 Windows 不允许创建测试目录符号链接 | 0 |
| `$projectPython tests/smoke_oven_capture.py` | 18次真实子命令调用；4张临时合成候选；损坏/孤立/未审核/跨 split/部分 catalog 拒绝，跨目录联合交接及共享 validator 通过；文件随后删除 | 0 |
| `$projectPython -m tools.oven_capture devices --out <ignored-inventory.json>` | 被动枚举2个本机候选；state UNKNOWN，opened_any_device=false；不是烤箱相机验收 | 0 |
| `tools/edge_export/setup.ps1`、`$edgePython -m pip check` | 独立锁定环境已建立并复查，无依赖冲突 | 0 |
| `$edgePython -m tools.edge_export.smoke --allow-downloads`，随后无下载复跑 | 29阶段：27个 exit0，两个直接 TORCH converter 阶段 exit1；16次数值比较完整保留 | 0（流程退出） |
| `$edgePython -m tools.edge_runtime.mnn_cpu ...`，父任务再独立复跑分类/检测 | CPU/high precision，float32 `[1,1000]` / `[1,84,2100]` | 各0 |
| 受保护文件/历史 Baseline 权重 SHA256 前后比较 | 11个对象字节哈希全部不变 | 0 |

完整单元日志：[unit-tests.log](phase2-evidence/20261029/unit-tests.log)；结构化核验：[foundation_verification.json](phase2-evidence/20261029/foundation_verification.json)。导出各阶段原始 stdout/stderr、实际命令与退出码见 [export_stage_logs.txt](phase2-evidence/20261029/export_stage_logs.txt) 及 [export_verification.json](phase2-evidence/20261029/export_verification.json)。`smoke` 的 exit0 表示流程完整记录，**不是全部路径成功或数值等价验收**。

修复前先实际运行回归，分别观察到缺实物ID确认被错误允许、缺license导出KeyError、跨目录部分导出漏关联、images-only中断集合被遗漏；修复后全部相应用例通过。review 写前重验完整 metadata；缺许可归 unknown；audit 支持联合目录；export 要求包含所有 images/records collection 并检查传递组，不能用独立导出绕过隔离。所有空/混合/未确认诊断记录仍会阻止单标签交接，不自动排除或删除现场证据。

## MVP 保全与实验边界

| 对象 | 前后相同的 SHA256 |
|---|---|
| 两份 v1.1.0 manifest | `6bef9776c278d8efac86008a11c47386cb7d95b7164073b9bd07e84cb5cf2821` |
| `configs/classes.json` 文件字节 | `27d5337826c26a3b743ec0928ea8da050f4ae7bf0ef31e7efb051e4e907e9614` |
| 历史真实 Baseline `best.pt` | `41056b3ea90d5fa718a18f48cf5182373d1010c5185cb426ba5c949b1ec1f4c5` |

本轮没有重新评测冻结1067张 test、没有利用它校准或选择新模型，没有修改八类顺序，也没有新增食物准确率结论。导出只使用官方 YOLO11 ImageNet/COCO 权重及官方 bus/zidane 图片，均只留在忽略目录；没有把它们当企业模型或独立食材评估。

FP32 MNN 能实际运行：分类11,275,392字节，检测10,550,860字节。weight8 分别2,933,892和2,820,888字节，已实际运行但有输出漂移；FP32检测及全部weight8未通过预设严格allclose。当前仅证明工具链和记录能力，尚未放行模型数值等价。weight8不是已校准INT8运算，更不是ARMv7加速证明。

## 环境意外记录

首次执行官方 MNN3.2.0 Python wrapper 的 `--help` 时，上游 logging import 意外启动裸 pip，写入全局16个新增包。已中断并按实际安装日志/metadata时间逐个移除；独立只读复查这16包及SDK均不存在，已有4项依赖保留。无完整事前全局快照，不能宣称全系统字节级还原。此后改用 native converter 并加已实测的网络/进程/shell/telemetry-import guard。

完整依据与限制：[事件报告](phase2-evidence/20261029/export_environment_incident.json)、[脱敏实际输出](phase2-evidence/20261029/export_incident_transcript.txt)、[独立清理复查](phase2-evidence/20261029/environment_cleanup_recheck.json)。没有提供凭据或运行日志上报。本轮没有安装新的系统Python/CUDA、WSL、Docker、驱动或交叉工具链。

## 明天最少人工步骤与观察结果

1. **核对再接线。** 三人按企业手册确认整机/板卡SKU、RAM、相机接口和供电、冷态隔离及访问权限，再由授权人员接线；应观察接口/标识与手册一致、没有异常。不明则保持 BLOCKED，不试插或加热。
2. **确认相机并单帧 probe。** 被动枚举后人工核对正确设备及backend，显式 `--device ... --cold-confirmed`；应得到可读取帧的实际宽高、驱动上报格式/FPS，不能以请求1920×1080代替结果。无可用UVC或需厂商SDK则 BLOCKED。
3. **填真实metadata，冷态拍少量图。** 先1张，再覆盖不同位置/照明、单/多食材及空场景；同实物/同session沿用化名。应产生原尺寸PNG/JPEG、sidecar、SHA/格式/时间，默认pending；没有自动标签或传感器RAW声明。
4. **人工看图并 audit。** 应看到损坏/孤立/重复候选和分组报告；先保留原始证据、逐张确认标签。多食材记录不能导出单标签manifest。首日只验质量，不冻结训练集。
5. **确认任务与板端入口。** 与工程师确认cls/det、YOLO版本、板卡Linux/SDK/ABI及允许访问方式；应取得可追溯规格/只读日志。随后才能安排FP32、正式校准INT8、板端RSS/时延验收；今天这些全部BLOCKED。

详细可复制命令见 [采集规范](PHASE2_OVEN_CAPTURE_PROTOCOL.md)；现场逐行填 [到货清单](OVEN_ARRIVAL_CHECKLIST.md)。未解决：真实相机/板卡验收、原生Ubuntu与厂商SDK、任务/版本/资源阈值确认、正式数据筛选与许可、代表性校准及板端部署。现有许可未核实图片和待复核黄瓜记录仍以原MVP材料为准，本轮没有改变其状态。
