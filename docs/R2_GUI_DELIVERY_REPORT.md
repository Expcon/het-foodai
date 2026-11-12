# R2 Windows 实验交付验收（2026-11-12）

结论：已实际构建并打开 **HeT OvenCapture Desk v1.2.2-r2-experimental**，原生窗口完成 v2 → R1 → R2 → v2 的真实模型加载及文件夹预测。R2 是可切换实验选项，未替换默认模型或历史软件。本轮不训练、不改变现场相机逻辑、不合并 main、不发布 Release。

## 分支和最小修改

工作开始时 Git 干净；从 `feat/model-oven-r2-whole-corn` 的 `bceb927` 建立 `feat/model-oven-r2-experimental-demo`。确认 R1 模型列表提交 `b3215ae` 已是该提交祖先，不需要再次合并或覆盖成员分支，没有合并冲突。

| 文件 | 修改原因 |
|---|---|
| `tools/oven_training/export_desk.py` | 原有安全导出链增加显式 R2 来源与已暴露诊断来源；绑定完整原清单，拒绝将封存测试追加为诊断；完整覆盖计数、原图和模型状态不变校验 |
| `tools/oven_desk/tests/round2_alignment_check.py` | Desk 从相同原图解析 R2 与诊断路径，验证逐图输入张量及输出 |
| `tools/oven_desk/model_catalog.py` | 保留旧目录格式兼容；新条目同时钉住 ONNX、JSON、checkpoint 身份及输入输出规格，任何不匹配拒绝加载 |
| `tools/oven_desk/oven_capture_gui.py` | 仅更新独立实验版版本号；复用已有 R1 模型列表及 ready/busy 保护 |
| `tools/oven_desk/build_windows.ps1` | 独立版本名称，不覆盖历史 EXE |
| `tools/oven_desk/tests/gui_catalog_smoke.py` | 显式四段模型切换序列与真实预测验收 |
| `tests/test_oven_export.py`、`tools/oven_desk/tests/test_model_catalog.py` | 新来源、防路径逃逸、封存测试拒绝、目录哈希和规格不匹配回归测试 |
| 本报告、`docs/R2_ONNX_ALIGNMENT_REPORT.md` | 实际命令、输出、范围和未验收项目 |

没有修改共享八类 classes、公共契约、原始清单、README、依赖文件、相机兼容、防覆盖、拍照、录像或加热相关实现。新权重/ONNX/EXE/图片及完整日志均只留在 Git 忽略的 `runs/`。

## 交付目录

`runs/oven-r2-demo-20261112/delivery/` 包含：

```text
HeT-OvenCapture-v1.2.2-r2-experimental.exe
experimental_models.json
models/v2/oven12.onnx
models/v2/classes.json
models/r1/oven12.onnx
models/r1/classes.json
models/r2/oven12.onnx
models/r2/classes.json
使用说明.txt
SHA256SUMS.txt
```

目录中的三组模型分别绑定历史 v2、R1 和新 R2 checkpoint；原 v2/R1 ONNX 与 JSON 按字节复制，没有重新导出。`experimental_models.json` 使用相对路径，无个人设置或绝对用户路径。每条记录绑定模型、JSON SHA、checkpoint SHA、`oven_round2`、FP32 `[1,3,224,224] → [1,12]`。

| 产物 | SHA256 |
|---|---|
| EXE | `6d8d2ead99e5ad9337fb2d8c9dd98e66cbbf301811d171de1f32f40b6883a5bc` |
| R2 ONNX | `c3c00944c7765b74d02b4f890caf77b2fe77bbadb87dacdb436f99a541febaa6` |
| R2 JSON | `fe0484729f3ec5619dbdf4372072f42fef1aefae11c7bd207d69d70bf266fc77` |
| 完整 ZIP | `d148ace8bc74e550502fe3af557e01c1d5cb0086cad240a2d5f211b9f6e5bb9a` |

ZIP：`runs/oven-r2-demo-20261112/HeT-OvenCapture-v1.2.2-R2-experimental-v2-R1-R2.zip`，99,762,199 bytes，内含单一顶层文件夹和 10 个文件。实际 CRC 检查通过、逐文件 SHA 一致、`SHA256SUMS.txt` 的 9 个负载文件全部核对通过。另解压到独立 `zip-relocation-check/`，从新路径运行其中 EXE，真实加载 R2 并预测 PNG，退出码 0、ready=true。未把 PT checkpoint、原始图片或用户设置打入 ZIP。

## 构建与测试

复用既有隔离 Windows Python 3.12.10 / PyInstaller 6.22.3 环境；没有改全局 Python 或安装系统依赖。实际 PyInstaller 构建命令如下，`$deskPython` 为既有独立构建环境，所有输出指向本次新目录：

```powershell
& $deskPython -m PyInstaller --noconfirm --clean --onefile --windowed `
  --name HeT-OvenCapture-v1.2.2-r2-experimental `
  --distpath runs/oven-r2-demo-20261112/exe/dist `
  --workpath runs/oven-r2-demo-20261112/exe/work `
  --specpath runs/oven-r2-demo-20261112/exe/spec --paths tools/oven_desk `
  --hidden-import serial.tools.list_ports_windows --hidden-import onnxruntime `
  --collect-binaries onnxruntime --collect-all tkinterdnd2 `
  --collect-all cv2_enumerate_cameras --exclude-module torch `
  --exclude-module ultralytics --exclude-module MNN --exclude-module MNN.tools `
  tools/oven_desk/oven_capture_gui.py

.venv/Scripts/python.exe -m unittest discover -s tests -v

& $deskPython tools/oven_desk/tests/gui_catalog_smoke.py `
  --catalog runs/oven-r2-demo-20261112/delivery/experimental_models.json `
  --image "<whole-corn-root>"/corn_white_0001.png --sequence v2 R1 R2 v2 `
  --out runs/oven-r2-demo-20261112/gui_catalog.json

& $exe --healthcheck $newHealthReport
& $exe --model-selfcheck --mode oven_round2 --model $onnx `
  --classes $matchingClasses --image $realImage --out $newModelReport
```

| 实际检查 | 结果与退出码 |
|---|---|
| PyInstaller 打包 | PASS，0；日志 `build.log` |
| 全项目单元测试 | 236 个：235 PASS、1 Windows symlink SKIP、0 FAIL，0；`unittest-final.log` |
| 新导出测试 / Desk 测试 | 导出 7 个 PASS；Desk 51 个 PASS，均 0；先 RED 后 GREEN 的日志保留 |
| 全量 PyTorch → ORT → Desk | 2168/2168 张量、Top-1、有序 Top-3 对齐；最大误差 3.6656856537e-6；两阶段均 0 |
| 真 Tk 模型列表四段切换 | PASS，0；`gui_catalog.json`，每次身份核验、重新载入、预测；busy 拒绝、无画面拒绝、失败载入清除 ready |
| 真 Tk R2 完整交互 | PASS，0；`gui_R2.json`：文件注入当前帧、PNG/JPEG、损坏图、空目录、失败重载、忙时保护；没有物理相机 |
| 打包 EXE healthcheck | PASS，0；Tk/ORT 可用，MNN 不包含在此包中 |
| 打包 EXE 正例 5 项 | v2 PNG、R1 PNG、R2 PNG、R2 JPEG、R2 严重欠曝图，全部 0、ready=true |
| 打包 EXE 负例 7 项 | 损坏图片、缺失模型、错配 JSON、未批准契约、截断模型、缺失图片、错误补边参数，全部如期 1、ready=false |
| 原生 EXE 窗口点击 | PASS：显式选模型/载入/选真实图集、v2 → R1 → R2 → v2；四次各 51/51 文件执行成功；无真值不显示准确率 |
| ZIP 完整性与异目录运行 | PASS；CRC/逐文件 SHA 无异常，解压后的 R2 EXE 自检 0 |

原生窗口没有仅凭 Python 脚本验收：实际点击独立打包 EXE 的“模型 / 图集”页，观察到三个模型均载入 12 类、224 输入；仅选择配置时回到“未就绪”；点击“识别当前摄像头画面”时日志“当前没有真实帧，无法预测”。相机一直显示“未连接”。窗口截图在工具对话记录中，未冒称已经生成磁盘截图文件。

原生图集 CSV 保留在 `native-ui-output/`：17:50:24(v2)、17:50:52(R1)、17:51:13(R2)、17:51:40(v2)。四份各 51 条成功记录并有完整 Top-3/概率；首尾 v2 的全部概率逐图完全相同，R2 原生图集概率与 PyTorch 参考最大误差 2.3841857910e-7。详见 `native_ui_report.json`，这只证明执行和模型身份切换，不是准确率。

如实保留两项验收操作错误：初次错误预处理负例误把报告输出设为同一个输入 JSON，防覆盖保护拒绝；已用独立 `wrong-preprocess-result.json` 重跑并确认契约拒绝。原生文件夹输入首次转义错误被 Windows 对话框拒绝，修正后完成 51 张测试。它们不是模型通过证据，也没有覆盖输入。

本次运行证据根目录 `runs/oven-r2-demo-20261112/`：`exe_test_summary.json`（5 正例/7 负例）、`exe-checks/`、`gui_catalog.json`、`gui_R2.json`、`native_ui_report.json`、`native-ui-output/`、`zip_verification.json`、`preservation_after.json`。历史日志未回填或改写。

## 现场使用

1. 完整解压 ZIP，不要只移动 EXE。保留原 v1.2.1 和历史模型。
2. 运行新 EXE，进入“模型 / 图集”，在实验模型列表选 **R2（整根玉米·实验12类）**。
3. 点击“载入真实模型”，等待显示已加载 12 类、224×224；失败时不得继续宣称 ready。R1/v2 的选择与加载方法相同。
4. “图集测试”选择直接包含 JPG/PNG 的文件夹；不递归。CSV 写到“相机 / 采集”的保存目录。没有人工真值时只显示预测、Top-3 和分数。
5. 实时画面需要用户现场明确选择并打开相机，看到真实预览后再点击识别；本轮没有替用户打开摄像头。

新目录默认仍为手动选择，不自动加载 R2，不覆盖已有默认设置。所有自动测试及原生点击均使用隔离 APPDATA；验收后用户原设置 SHA 保持不变。相机兼容及照片/录像防覆盖代码未修改。

## 未验收范围

- **BLOCKED：**物理摄像头当前画面预测、真实拍照/录像和现场设备兼容回归，按本轮禁止打开硬件要求未执行，需用户现场操作。
- **BLOCKED：**真正独立实物泛化评估；本次训练图和已暴露诊断数据均不具备独立测试资格。
- **未开展：**MNN、INT8、RK3506、生产部署、正式 Release。没有将这些项目标为 PASS。
- 当前实验软件及文件推理没有未解决失败项；上述负例均是预期拒绝，不能计作成功识别的食材样本。
