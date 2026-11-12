# HeT OvenCapture v1.3.0 展示实验版回归报告

日期：2026-11-12。基线：已交付 v1.2.2 / `c7d7e62`。分支：`feat/app-oven-presentation-v1.3`。

本轮改善操作流程和结果关联可靠性，没有训练、替换模型或改变预处理。它不是通用推理引擎改造，也不构成新的食材准确率评估。

## 实际修改

| 文件 | 用途 |
|---|---|
| `tools/oven_desk/presentation.py` | 默认识别首页，分离采集、模型管理和高级设置；等比画面、大字结果、模型分数、实测时间、快照与可见错误 |
| `tools/oven_desk/recognition_session.py` | 请求与来源/模型代次绑定，复制实际推理帧，拒绝过期结果，独占保存 PNG/JSON |
| `tools/oven_desk/oven_capture_gui.py` | 导入图片无需相机；显式加载和身份哈希；可靠完成队列；状态失效与页面接线 |
| `tools/oven_desk/build_windows.ps1` | 独立新输出目录、复用隔离环境、三模型完整打包、EXE 依赖自检，拒绝覆盖已有目录 |
| `tools/oven_desk/展示版使用说明.txt` | 中文操作与计时/分数/硬件边界 |
| `tools/oven_desk/tests/test_recognition_session.py`、`test_presentation.py` | 新状态、帧绑定、错误、异步、缩放与长名称回归；合成输入明确标注 |
| `tools/oven_desk/tests/presentation_smoke.py` | 真实模型与 JPG/PNG 的旧/新输出对照、保存身份和六种软件布局检查 |
| 三个现有 GUI 测试文件 | 将“文件注入相机”改为真实图片导入；适配带身份的完成事件并保留断言 |
| 本报告及设计/执行记录 | 可复核的交付边界与复现方法 |

`inference_engine.py`、`capture_core.py`、`camera_devices.py`、`model_catalog.py` 和原 `CaptureWorker` 保持不变；共享配置、旧八类清单、历史权重及原图未修改。990 个事前保护文件全部通过 SHA256 回查，用户原设置 SHA256 也保持一致。

## 已运行的测试

所有日志和独立产物位于 `runs/oven-desk-presentation-20261112/`；未提交权重、原图、截图或二进制到 Git。

| 命令/入口 | 退出码与结果 | 证据 |
|---|---|---|
| `.venv/Scripts/python.exe -m unittest discover -s tests -v` | 0；261 项，260 PASS、1 SKIP、0 FAIL，38.443s | `unittest-reviewed.log`；Windows 不允许创建测试符号链接而跳过一项 |
| 构建环境 `python -m unittest discover -s tests -v`（目录 `tools/oven_desk`） | 0；75/75 PASS，8.692s | `build-final.log` |
| `tests/presentation_smoke.py --catalog ... --baseline ... --out ...` | 0；v2→R1→R2→v2 × 原始 PNG/JPG 共 8 次，原始输出逐项完全一致 | `presentation-reviewed.json` |
| `tests/gui_catalog_smoke.py --sequence v2 R1 R2 v2 ...` | 0；真实载入/预测、忙时拒切换、无帧拒绝及缺模型失效 | `catalog-real-fixed.json` |
| `tests/gui_model_smoke.py --mode oven_round2 ...` | 0；真实加载、文件预测、图集 PNG/JPEG 成功 2 张、损坏文件失败 1 张，空目录拒绝 | `model-real.json` |
| `tests/gui_fake_camera_smoke.py` | 0；合成设备的两次带元数据拍照、5 张编号连拍、AVI 和安全停止 | `synthetic-capture.log`，不是物理硬件证据 |
| `tests/gui_named_camera_smoke.py` | 0；合成设备列表的名称/后端映射与回退保护 | `synthetic-named.log`，不是设备连接证据 |
| 新 EXE `--healthcheck <新报告>` | 0；ORT、Tk、相机枚举组件导入通过 | `build-final/healthcheck.json`；本构建明确 `-SkipMNN` |
| 新 EXE `--model-selfcheck --mode oven_round2 ...` | 六个 v2/R1/R2 × 原始 PNG/JPG 返回 0，输出与旧基线完全一致 | `packaged-verification.json`、`packaged-checks/` |
| 新 EXE 错模型/错 JSON/坏图片 `--model-selfcheck` | 各返回 1；均为预期拒绝，没有就绪或假预测 | 同上，三条负例原始 JSON |

首次误用全局 `python` 时因缺 Pillow 失败，记录保留在 `unittest-final.log`。随后使用项目 `.venv` 完成上述全套测试，没有安装全局依赖。布局和两个独立复核问题均保留 RED→GREEN 日志，没有删除失败证据。

## 原生 EXE 点击与截图

实际运行 `HeT-OvenCapture-v1.3.0-presentation-experimental.exe`，不是源码替代验收，也没有使用模拟模型：

- 手动选择 v2→R1→R2→v2，每次必须点击“载入真实模型”。选择配置后旧结果/就绪状态失效，实际加载校验后才显示就绪。
- 没有连接相机，使用 Windows 文件框导入真实 PNG 和 JPG 后识别成功。相同玉米 PNG 的 v2 Top-3 为 sausage / french_fries / chicken_wing，R1 为 corn / broccoli / chicken_wing，R2 为 corn / broccoli / toast；回切 v2 恢复原输出。
- R1 连续点击没有串图或错误成功卡；后台任务期间重复提交及 A→B→A 的精确拒绝由单元/真实 Tk 工作流测试另外验证。
- 换 JPG 后先清空原玉米结果，再显示该 JPG 的真实结果和缩略图。损坏图片清空画面与成功卡，显示解码错误；错 JSON 和缺模型均显示加载失败。
- “摄像头画面”只改变来源；未连接时显示空状态，识别与保存结果按钮禁用，没有打开设备。
- 原生点击“保存本次结果”后，实际 PNG、JSON 与源图/推理帧/模型 SHA、Top-3 完全一致。该次实测 12.067599993315525 ms，界面显示 12.1 ms；不是性能基准或中位数。

包内 `screenshots/` 的截图都直接来自真实程序：

| 状态 | 文件 |
|---|---|
| 未就绪 | `exe-unready.png` |
| 模型就绪、尚未识别 | `exe-ready.png` |
| 真实 R2 PNG 识别结果 | `exe-result.png` |
| 损坏图片错误、旧结果已清空 | `exe-error.png` |
| 额外证据 | v2/R1/回切结果、真实 JPG、错 JSON、缺模型、无帧、两种窗口尺寸 |

最初用户误按 Escape，中止了前一次点击验收；没有把中止标为成功。用户恢复授权后重新实际完成上述步骤。四状态截图没有使用设计图、生成图或旧测量值。

## 显示、时延和结果契约

验收使用本机 Windows/Python 环境；实际硬件和显示设备配置保留本地，不公开。EXE 分别以 `--window-size 1366x768` 和 `1920x1080` 原生运行、加载和识别，关键按钮、来源/模型/时间及错误区域可见。截图工具按自身坐标缩放输出，PNG 像素尺寸不等于应用客户区尺寸。另有两个客户区尺寸 × Tk 字体 DPI 96/120/144 的六组软件布局检查；它们不能冒充三种物理 Windows 缩放环境。其他显示器及系统 100%/125% 缩放尚未实际验收。

类别名称从当前模型契约读取，corn 的 ID 9 不变。Top-3 显示为“模型分数”，没有准确率、校准概率、低分阈值或未知类承诺。长名称首页省略显示，完整名称保留在日志和结果 JSON，模型页保留完整身份。

首页计时覆盖模型锁等待、预处理、CPU 推理和输出检查；不含图片解码、模型加载、文件哈希或界面绘制。EXE 命令行自检另有包含解码的墙钟计时，二者口径明确区分。

`result.json` 保存请求编号、来源类型/原路径/原文件 SHA、开始与完成时间、模型/类别/来源 checkpoint 身份、完整 Top-3、实测耗时及范围。`frame.png` 是该次推理的解码帧，不是随后到来的摄像头帧，也不宣称传感器 RAW。解码帧 SHA 与保存 PNG 文件 SHA 分别记录。结果保存使用新目录与独占写入。

## 构建与迁移

复用已有隔离 Python 构建环境中的 PyInstaller 6.22.3、ORT 1.23.2、Pillow 12.0.0、NumPy 2.2.6、OpenCV 4.12.0；没有修改 Windows 全局 Python/CUDA。示例：

```powershell
# PythonPath 指向已有的独立构建环境；OutputRoot 必须是仓库 runs/ 下尚不存在的目录。
./tools/oven_desk/build_windows.ps1 -SkipMNN `
  -PythonPath <隔离环境>/Scripts/python.exe `
  -OutputRoot <本仓库>/runs/<新的构建目录> `
  -ModelBundle <本仓库>/runs/oven-r2-demo-20261112/delivery
```

完整交付目录包含 EXE、原 catalog、v2/r1/r2 三组 ONNX/JSON、使用说明、截图与回归报告。必须完整解压后运行，模型列表使用相对路径。`SHA256SUMS.txt` 校验包内文件；ZIP 的 CRC、逐项 SHA 和解压迁移自检记录在包旁 `zip-integrity.json`，ZIP 自身 SHA 另存同名 `.sha256`，避免自引用校验。

独立复核发现并修复：长合法标签/模型名挤掉结果身份；过期成功任务覆盖新损坏图片错误。两项都有先失败后通过的测试。无延期的代码复核问题。为保留用户要求的完整证据，忽略目录中的失败日志、截图和临时构建没有删除。

## 仍需现场确认

- 物理摄像头连接、原尺寸拍照、连拍、录像、串口和设备断开；本轮没有打开设备或控制烤箱，合成设备测试不能替代这些项目。
- 不同物理显示器、系统 100%/125% 缩放及其他同学电脑。
- MNN/INT8/RK3506、通用模型适配、独立实物泛化评估均不属于本版已验证能力。

保留原交付包和全部模型，没有合并 main、发布 Release 或改变现场默认模型策略。下次启动仍需显式选择并载入模型。
