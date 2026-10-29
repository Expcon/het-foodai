# YOLO → MNN 离线工具链实测（2026-10-29）

本轮已在 **Windows AMD64、Python 3.12.10、CPU** 实际完成官方 YOLO11 nano 分类与检测模型的 ONNX → MNN 导出及推理。两个模型的 TorchScript → MNN 直接路径均被所装 Windows converter 拒绝。分类和检测的 FP32、8 bit 权值压缩版本均实际运行了两张官方样例；数值误差已逐项保存，不能据此宣布企业部署模型、食物识别精度或目标板性能达标。

HeT Phase2 任务类型及企业 YOLO 版本仍为 **UNKNOWN**。这里的 `yolo11n-cls.pt`（ImageNet 1000 类）和 `yolo11n.pt`（COCO 80 类）只是可重复的工具链试件。未训练模型、未读取冻结 MVP 数据或权重、未访问硬件。

## 任务选择，等待现场确认

| 比较项 | Classification 整图分类 | Detection 多目标检测 |
|---|---|---|
| 业务问题 | 一张图的一个主类别；需约定空炉/混合食材处理 | 一张图内各目标的类别与位置；需约定遮挡/小目标/空炉 |
| 标注要求 | 人工确认每图一个标签；现有八类数据可作为分类候选来源 | 每个目标的真实类别与框；现有整图标签没有框，不能直接用于检测训练 |
| 输出/后处理 | 类别概率及 Top-k | 本次模型原始框+类别张量，需阈值、NMS和逆 letterbox 坐标变换；其他版本可能不同 |
| 模型规模 | 本次 nano FP32 MNN 11,275,392 字节 | 本次 nano FP32 MNN 10,550,860 字节；不能由此推断运行内存更小 |
| 训练成本 | 标注工作通常较少，224px 试件；实际训练时间/显存未测 | 框标注和逐目标审核工作更多，320px 试件；训练时间/显存未测，受尺寸/目标数/版本影响 |
| RK3506 风险 | 裁剪可能丢掉边缘食材，空炉/多食材不能强行判单类 | 输入、检测张量与NMS增加内存/后处理负担；目标板支持和速度均待实测 |

依据 [Ultralytics 分类](https://docs.ultralytics.com/tasks/classify/)、[检测](https://docs.ultralytics.com/tasks/detect/)、[分类数据格式](https://docs.ultralytics.com/datasets/classify/)、[检测数据格式](https://docs.ultralytics.com/datasets/detect/)。先用明天的单食材/多食材真实视角，与工程师确认是否需要位置框，再选任务和版本；本轮没有根据旧分类准确率替团队决定任务，没有开展正式 YOLO 训练。

## 固定环境和真实转换路径

- 新隔离环境：`runs/phase2-edge/.venv`，不复用项目 `.venv`；完整依赖见 `tools/edge_export/requirements-cpu.lock.txt`。
- 已安装并核对：Ultralytics `8.3.230`、PyTorch `2.8.0+cpu`、torchvision `0.23.0+cpu`、MNN `3.2.0`（CPython 3.12 Windows x64 wheel）、ONNX `1.19.1`、ONNX Runtime `1.23.2`、NumPy `2.2.6`。`pip check` 返回 `No broken requirements found.`，exit 0。
- 先执行 PyTorch → TorchScript → native MNN converter `-f TORCH`：分类和检测均实际失败，子进程 exit 1。native converter 打印 `Framework Input ERROR or Not Support This Model Type Now!`，未生成 MNN；脚本额外核验非空输出，避免 converter 返回但无模型被误判成功。本 wheel 的 `--help` 也未列出 TORCH。此结论只针对已测 wheel，不外推所有 MNN 构建。
- 然后执行 PyTorch → ONNX → native MNN converter `-f ONNX`：分类/检测的 FP32 与权值 8 bit 版本均转换、加载及推理成功。固定 `batch=1, device=cpu, half=False, dynamic=False, nms=False, opset=17, simplify=False`；MNN FP32 不加压缩开关，weight8 只增加 `--weightQuantBits 8`。图导出没有内嵌 NMS。

之所以明确区分两条路径，是因为[已锁定版本的 `export_mnn` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.230/ultralytics/engine/exporter.py)内部先调用 `export_onnx()`，不是直接 PyTorch 转换。当前[官方在线文档](https://docs.ultralytics.com/integrations/mnn/)已经展示新 API 的 `quantize` 参数；本轮没有把新 API 写进旧版本脚本。

## 实际 I/O 与文件大小

| 官方工具链试件 | 实际输入（float32 RGB NCHW） | 实际输出 | FP32 MNN 字节 | weight8 MNN 字节 |
|---|---|---|---:|---:|
| `yolo11n-cls` | `[1,3,224,224]` | `[1,1000]` 概率 | 11,275,392 | 2,933,892 |
| `yolo11n` | `[1,3,320,320]` | `[1,84,2100]` 原始检测张量 | 10,550,860 | 2,820,888 |

分类使用锁定版本官方 `classify_transforms(224)`：RGB、保持比例缩放短边、中心裁剪、ToTensor，mean=0/std=1。检测使用官方 `LetterBox((320,320), auto=False)`，居中补边 114、BGR 转 RGB、NCHW、除以 255。它们不是可互换的预处理。

本次检测输出的 84 代表 `4 个 cx/cy/w/h + 80 类概率`，该契约只适用于本次 YOLO11 原始检测头。比较使用**同一锁定版本的官方** `non_max_suppression`，`conf=0.25, iou=0.45, class-aware, max_det=100`。保存的框仍在 320×320 输入坐标系内，未冒充原图坐标。其他 YOLO 版本、NMS-free 或已内嵌 NMS 的输出需要另写并验证契约。

## 与 PyTorch 的实际比较

参考值是同一官方权重在 CPU 上运行的真实 PyTorch 输出；TorchScript、ONNX Runtime、MNN 都使用同一份已保存 `.npy` 输入。MNN 固定 CPU、1 线程、`precision=high`。以下为原始输出全张量的最大绝对误差；检测坐标与概率在同一张量中，因此不能把该数值解释成准确率。

| 任务/样例 | FP32 MNN 最大绝对误差 | weight8 MNN 最大绝对误差 |
|---|---:|---:|
| 分类 / bus | 0.000154376 | 0.066438556 |
| 分类 / zidane | 0.000066668 | 0.026433855 |
| 检测 / bus | 0.093322754 | 22.056763 |
| 检测 / zidane | 0.068832397 | 50.202484 |

FP32 分类两样例保留相同 top5，weight8 的 bus 第五名由 class 829 变成 408。检测两样例在相同 NMS 后均保留与 PyTorch 相同的有序类别数列和数量（bus：bus + 4 person；zidane：3 person），置信度及坐标有变化。所有逐项概率、原始误差均值、NMS 后框、`allclose(atol=1e-4,rtol=1e-4)` 布尔值均在 [export_results.json](phase2-evidence/20261029/export_results.json)。FP32 检测和全部 weight8 比较未通过该严格 allclose 检查；脚本 exit 0 仅表示预定的导出/运行/记录流程完成，**不表示数值等价或模型验收通过**。

`--weightQuantBits 8` 仅压缩 conv/matmul/LSTM 的浮点权值；[MNN 官方转换文档](https://mnn-docs.readthedocs.io/en/latest/tools/convert.html)和实际 wheel help 均将其说明为模型大小优化。这里没有激活标定、没有声明全整数计算、没有随机/合成标定，更没有 INT8 加速结论。真实 INT8 计算需要明确目标构建、代表性标定集及单独数值/精度/性能验收。

## 来源、许可、文件追溯

权重来自 [Ultralytics 官方 assets v8.3.0](https://github.com/ultralytics/assets/releases/tag/v8.3.0)，输入来自锁定版本官方仓库的 [bus.jpg](https://github.com/ultralytics/ultralytics/blob/v8.3.230/ultralytics/assets/bus.jpg) 和 [zidane.jpg](https://github.com/ultralytics/ultralytics/blob/v8.3.230/ultralytics/assets/zidane.jpg)。模型、图片和原始输出只放在忽略目录 `runs/phase2-edge/`，不提交 git。

Ultralytics 的[官方许可说明](https://www.ultralytics.com/license)涵盖 AGPL-3.0 / Enterprise 两种方式；本轮公开来源不等于取得企业闭源部署许可。样例照片的原始摄影版权未独立审计，因此也未再分发照片。MNN [上游 README](https://github.com/alibaba/MNN#license)标为 Apache 2.0，但本次安装的 MNN 3.2.0 wheel `METADATA` 标为 `BSD-3`；已保留此不一致，正式再分发前需核对具体构建的许可包。每个下载文件、导出模型的来源 URL、字节数和 SHA-256 已保存到 evidence；脚本固定四个官方 URL 并核验本轮观察到的 SHA-256，文件变化时拒绝直接加载。

## 复现与最小 CPU runtime

在仓库根目录 PowerShell 执行（这些命令已实际运行；setup 复核、smoke 主流程、独立 runtime 均 exit 0）：

```powershell
& .\tools\edge_export\setup.ps1
$edgePython = (Resolve-Path runs\phase2-edge\.venv\Scripts\python.exe).Path
& $edgePython -m tools.edge_export.smoke --allow-downloads
# 再次运行不需要网络；已实际复跑：
& $edgePython -m tools.edge_export.smoke

& $edgePython -m tools.edge_runtime.mnn_cpu `
  --model runs\phase2-edge\export-smoke\yolo11n-cls-onnx-fp32.mnn `
  --tensor runs\phase2-edge\export-smoke\yolo11n-cls-bus-input.npy `
  --contract runs\phase2-edge\export-smoke\yolo11n-cls.contract.json `
  --output runs\phase2-edge\export-smoke\standalone-cls-output.npy
```

独立示例只需要 MNN、NumPy 和仓库内的标准库 guard，不导入 PyTorch/Ultralytics。它要求显式任务、layout、dtype、shape、范围及输出头契约；拒绝 NHWC、错误 dtype、NaN/Inf、logits 冒充分类概率等。输出是原始 `.npy`；分类不捏造食品标签，检测不猜测通用 YOLO 后处理。检测的实际独立运行同样返回 float32 `[1,84,2100]`，完整命令及 stdout 在 [export_verification.json](phase2-evidence/20261029/export_verification.json)。

```powershell
& $edgePython -m unittest discover -s tests -p 'test_edge_*.py' -v
```

本模块 8 个有意义的边界测试通过，exit 0，输出 `Ran 8 tests ... OK`。完整项目测试及 PR 前验证由 Phase2 总体验证记录另行给出。原始阶段日志（已脱敏）见 [export_stage_logs.txt](phase2-evidence/20261029/export_stage_logs.txt)。阶段耗时包含解释器启动/模型加载，不能作为推理性能指标。

## 已纠正的环境意外与尚未验证事项

首次查询官方 `MNN.tools.mnnconvert --help` 时，其 Python wrapper 的 logging import 意外执行裸 `pip install -U aliyun-log-python-sdk`，解析到全局 Python。已立即中断，并按实际 stdout 的 Collecting/Installing 清单及新建 metadata 时间精确移除 16 个本次新增包；没有既有包卸载记录，SDK 本体未装完。复查这些包均不存在且无遗留子进程。没有转换/日志上报执行、没有提供凭据。此前未保存完整 global 基线，故不能宣称做过全系统前后快照比对。完整影响、修复依据及限制见 [export_environment_incident.json](phase2-evidence/20261029/export_environment_incident.json)。

完整安装、清理及复核输出已从实际工具返回保存在[脱敏 transcript](phase2-evidence/20261029/export_incident_transcript.txt)，没有重新执行该 wrapper。此后所有转换通过 `_tools.mnnconvert` 内核绑定；SDK worker 和独立 runtime 安装 Python audit guard，实际阻断裸 pip、shell、网络和 `MNN.tools` import，独立实测 exit 0。Ultralytics settings 限定在忽略目录，`sync=false`；首次产生的两个根目录临时配置文件也已移入忽略目录。

没有验证原生 Ubuntu、ARMv7/RK3506 构建、目标板 MNN 算子/内存/延迟/功耗、正式校准或真实食物集精度；没有为此安装系统 Python/CUDA、WSL、Docker、编译器或驱动。正式部署仍需确认企业任务/模型版本、板卡 SKU/RAM/系统 ABI、输入输出及许可证，再在真实目标平台完成验收。
