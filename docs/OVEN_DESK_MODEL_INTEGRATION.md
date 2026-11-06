# OvenCapture Desk v1.1 九类模型接入

本轮继续 PR #8，使用已训练的 `last.pt`，没有重新训练、选择 checkpoint、改动原始图片或旧八类 MVP。用途是本地探索性分类和采集辅助，**不是已完成泛化评估或端侧验收的可部署版本**。无人工真值的图集只保存预测，不计算准确率。

## 原程序与修改范围

已找到用户本机 `HeT_OvenCapture_Windows_v1.1.0` 完整源码，复制到 `tools/oven_desk/` 纳入本分支；没有重写 GUI，也没有修改桌面的原程序。`camera_devices.py`、`capture_core.py` 与原版逐字节一致，GUI 的相机打开/兼容切换、拍照/录像、防覆盖逻辑不改。原始文件哈希保存在本轮证据目录的 `source_provenance.json`。

原 `yolo_classify` 模式是中心裁剪，会丢失本轮训练使用的全图内容。因此增加独立 `oven_round1` 模式；其他旧模型继续使用原选项。加载本轮契约时必须选新模式，错误模式、错误类别、错误 SHA、未验收的模型格式及失败的真实前向均不能变为 ready。GUI 工作线程持有加载时的模型快照，推理期间禁止替换，重新加载前清除旧 ready 状态。

预处理严格复现 `tools/oven_training/model.py`：Pillow EXIF 方向校正、RGB、`ImageOps.contain(...,(224,224),BILINEAR)`、居中补 RGB `(114,114,114)`、float32 /255、NCHW。尺寸取整及居中偏移也使用 Pillow 规则。相机传入的是完整 BGR 解码帧，先转 RGB；它不是传感器 RAW。新模式文件读取使用 Pillow，避免 JPEG 解码器差异；没有改变照片保存路径。

## 类别及模型文件

按 ID 0–8：`asparagus, bacon, broccoli, chicken_wing, croissant, french_fries, potato, salmon, toast`。它们属于本轮实验命名空间，**不覆盖 `configs/classes.json` 的八类冻结表**。

| 文件/有序列表 | SHA256 |
|---|---|
| 原 `last.pt` | `72367305b65b31920c59865758dcdd4a1ee506751bde57db113f017c5c314b25` |
| 本轮有序类别 JSON 紧凑序列 | `42bd80f04446cd87f8c719c78034bf274bd447fe7a60205713f118cd0b584b52` |
| ONNX | `e1ea69368fa420f3df5a13ef1df8bea9837fdfd3e6f459adb64e3c040742ffb7` |
| EXE 读取的 `classes.json` 文件 | `bee97ce33c0d8122eb428ace6e5879b7b8fcc095ba8f5df7853cb64222f391dc` |
| 未通过验收的 MNN 候选 | `7d20e2923ba87190d339e51769376240190667072d8dd13098bf08acd2ba8b5c` |

`classes.json` 绑定模型 SHA、原 checkpoint SHA、类别顺序、精确预处理、输入输出维度与实验状态。文件位于忽略的 `runs/oven-desk-20261106/`，只批准 ONNX。MNN 候选不能作为可用模型交付，未加入格式批准列表。SHA 校验用于发现错配/字节变更，不能替代文件来源可信度或代码签名。

导出入口使用已有安全加载器 `torch.load(weights_only=True)`，检查 metadata、严格 state_dict、前向，不把 `last.pt` 当任意 Ultralytics pickle 对象加载。静态 FP32 ONNX：opset17，输入 `images [1,3,224,224]`，输出 `probabilities [1,9]`；输出已含 softmax，不再做二次 softmax。

## 实际数值验证

全部569张真实烤箱 PNG逐张检查原图SHA。每张记录9维概率、Top1、排序Top3、绝对误差及图片SHA；重复的1张仍包含在格式检查中。它们是已见训练图，不提供独立准确率证据。验收门槛预先设为所有图排序Top3一致、最大绝对概率误差≤`1e-4`，且概率合法。

| 验证 | 结果 |
|---|---|
| PyTorch CPU FP32 → ONNX Runtime CPU | PASS，569/569 Top1及排序Top3一致；最大误差 `1.7881393432617188e-6` |
| EXE依赖环境的Desk完整文件解码/预处理 | PASS，569张输入张量与训练端逐元素完全相同；Top1/Top3全一致，最大误差同上 |
| ONNX → MNN3.2.0原生转换器 | 文件生成成功，6,185,832字节；没有使用会自动安装云日志依赖的 `MNN.tools` |
| MNN CPU high与PyTorch | **FAIL**，569张Top1/Top3一致但最大误差 `0.0005148649215698242`，468个输出向量不满足概率合法性检查；例如分数大于1。没有裁剪/重归一化掩盖错误 |
| INT8 / RK3506 / 真实相机 | 未验收；本次没有量化、板卡推理或相机操作 |

MNN首次检查在第一个非法概率抛出异常（exit1）；随后保留同一MNN字节，以 `check-mnn` 完成全部569张失败记录（exit2）。转换文件成功不等于运行通过。重新检查MNN会先撤销旧授权，失败或异常不能遗留先前的 ready 授权。

依赖：导出环境 Torch2.14.1+cu130 / torchvision0.29.1+cu130 / Ultralytics8.3.230 / ONNX1.19.1 / ORT1.23.2 / NumPy2.5.2 / Pillow12.3.0；EXE环境 NumPy2.2.6 / Pillow12.0.0 / OpenCV4.12.0.88 / ORT1.23.2。均为既有隔离虚拟环境，仅当前项目 `.venv` 新增导出依赖；旧GPU环境及桌面构建环境没有安装或覆盖包。

## 重现导出与检查

在仓库根目录执行。模型和图片需从本地备份单独恢复；不要把权重或照片加入 Git。下面导出必须使用全新的输出目录，已有权重不覆盖。

```powershell
$py = '.venv/Scripts/python.exe'
& $py -m pip install -r tools/oven_training/requirements-export.txt
& $py -m tools.oven_training.export_desk export --checkpoint runs/oven-first-round-20261106/yolo11n-fixed30/last.pt --expect-sha256 72367305b65b31920c59865758dcdd4a1ee506751bde57db113f017c5c314b25 --rows runs/oven-first-round-20261106/candidate_rows.json --out runs/oven-desk-NEW
# ONNX全部通过后才能转换，使用已有MNN隔离环境；失败保持FAIL。
& runs/phase2-edge/.venv/Scripts/python.exe -m tools.oven_training.export_desk mnn --out runs/oven-desk-NEW
# 对原MNN字节复查，不重复转换/覆盖：
& runs/phase2-edge/.venv/Scripts/python.exe -m tools.oven_training.export_desk check-mnn --out runs/oven-desk-NEW
& $py -m unittest discover -s tests -v
```

独立Desk环境执行 `tests/smoke_oven_desk_alignment.py --bundle runs/oven-desk-NEW --rows runs/oven-first-round-20261106/candidate_rows.json` 可重现569张预处理与预测对齐。此脚本须在该环境安装训练项目源码可引用的标准依赖；不需要Torch或Ultralytics运行时。

## Windows 打包与使用

源码和标准打包脚本位于 `tools/oven_desk/`。使用 Python3.12 x64，脚本建立自己的 `.build-venv`，不修改全局 Python/CUDA；构建时执行单元测试与无相机健康检查。EXE不内置权重或真实照片。单独交付 `oven9.onnx` 和匹配 `classes.json`，不要选旧八类表或未经验证的MNN候选。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/oven_desk/build_windows.ps1
```

打开「模型 / 图集」，选择本轮九类模式、网络尺寸224，选择上述ONNX与类别JSON，点击载入。只有哈希、契约和真实前向通过才显示已加载。图集选择只读JPG/PNG目录，结果CSV另存于用户选择的输出目录；每张保存Top3与分数，损坏图片记录失败，继续其他图片。没有真值不计算准确率。当前帧识别要等用户自行打开相机并确认真实画面后使用；此次文件注入测试不能替代真实相机验收。

当前帧墙钟时间包括BGR→RGB、缩放补边、张量转换、CPU推理及Top3解析，不含相机取帧、模型加载或UI刷新；命令行模型自检计时还包括文件解码，排除SHA计算。图集CSV不提供逐图时延。CPU实现不涉及CUDA异步计时。分数未经校准，未知食材、多食材及空炉仍可能被强制分到九类之一，不用于自动烹饪控制。

## 本机实际验收记录

| 命令/操作 | 退出码与结果 |
|---|---|
| `.venv/Scripts/python.exe -m unittest discover -s tests -v` | exit0；151项，150 PASS、1 SKIP、0 FAIL，20.486s；SKIP为Windows目录符号链接权限 |
| 独立Desk环境 `python -m unittest discover -s tools/oven_desk/tests -v`（需将工具目录加入PYTHONPATH，或在工具目录运行 `-s tests`） | 31项PASS，exit0 |
| `python -m tools.oven_training.export_desk export ...` | exit0；真实安全权重→ONNX及569图对齐 |
| `python tests/smoke_oven_desk_alignment.py --bundle runs/oven-desk-20261106 --rows runs/oven-first-round-20261106/candidate_rows.json` | exit0；EXE依赖环境569图张量及概率对齐 |
| `python tools/oven_desk/tests/gui_model_smoke.py --model .../oven9.onnx --classes .../classes.json --image <真实PNG> --out <新JSON>` | exit0；8条实际Tk/工作线程路径通过，包括文件注入当前帧、PNG/JPG图集、坏图、空目录、失败重载、忙时禁止切换 |
| `python -m PyInstaller --onefile --windowed ... tools/oven_desk/oven_capture_gui.py` | exit0；沿用桌面已有隔离构建环境，只读使用依赖；参数与标准脚本一致，dist/work/spec改到本地runs目录 |
| EXE `--healthcheck <新JSON>` | exit0；图像保存/编号/中文路径、Tk、ORT、MNN运行库导入通过；MNN库存在不等于本轮MNN模型获批 |
| EXE `--model-selfcheck --model .../oven9.onnx --classes .../classes.json --image <真实PNG或派生JPG> --out <新JSON>` | 两次exit0，model_ready=true，真实Top3；JPG由真实PNG转码，不是新增独立采样 |
| 同EXE自检，损坏图片/缺模型/未批准MNN | 各exit1，model_ready=false，属于预期拒绝，不伪造预测 |
| 实际EXE窗口点击模型页、载入、选择文件夹 | 已观察九类224已加载及“实验预测/无独立验证”；3条图集记录中2张有效图成功、1张损坏图失败；CSV含Top3、不含accuracy；测试窗口已关闭 |

EXE使用 `Start-Process -Wait -PassThru -WindowStyle Hidden` 获取自检退出码，避免Windows GUI程序异步返回造成误判。原生界面验收使用隔离APPDATA和本地CSV输出；没有修改用户原设置，没有打开相机或串口。初始化沿用原版只读设备名称枚举，实际摄像头状态保持“未连接”。

实际EXE：`runs/oven-desk-20261106/dist/HeT-OvenCapture-v1.1.1-experimental.exe`，84,916,640字节，SHA256 `f77da2c62387512eefe20e69d6bcba9552ada2f5d10a17d2bfc05d993e68615b`。本地交付ZIP在同级 `runs/oven-desk-20261106/HeT-OvenCapture-v1.1.1-experimental-ONNX.zip`，仅含EXE、批准的ONNX、classes.json、说明及哈希，不含原图、原训练权重或未批准MNN。没有上传模型或发布Release。

数值摘要、脱敏单测日志、保全SHA和EXE验收记录见 [本轮证据](phase2-evidence/20261106-desk/acceptance_summary.json)。完整逐图概率与错误记录只留在本地 `runs/oven-desk-20261106/{onnx_alignment,mnn_alignment,desk_alignment}.json`。当前模型保留探索性质；独立跨场景评估、真实相机识别和ARM端侧验收仍待完成。

实现依据：[PyTorch ONNX导出](https://docs.pytorch.org/docs/2.14/onnx.html)、[ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)；实际锁定版本与源码为复现基准。
