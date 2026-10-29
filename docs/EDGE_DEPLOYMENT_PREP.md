# Phase 2 — RK3506 / Linux / MNN 部署准备

核查日期：**2026-10-29（Asia/Shanghai）**。本文是准备材料，不是板端验收报告；本次未连接板卡、未打开摄像头、未加热、未刷机、未安装驱动或系统环境。官方来源索引见 [platform-sources.json](phase2-evidence/20261029/platform-sources.json)，到货操作记录见 [OVEN_ARRIVAL_CHECKLIST.md](OVEN_ARRIVAL_CHECKLIST.md)。

## 1. 已确认路线与待确认规格

2026-10-22 企业沟通路线（用户提供，非官网硬件证明）：YOLO、INT8、Linux、MNN、RK3506 系列、1080p 摄像头；新数据与旧数据联合重训，暂不引入复杂增量算法。预计到货 2026-10-30，实际到货与可用性尚未观察。

| 项目 | 当前记录 | 放行证据 / 负责人 |
| --- | --- | --- |
| Phase 2 实际任务 | **UNKNOWN**：单主食分类，还是多目标检测 | 三人共同确认企业需求；模型负责人记录决策 |
| YOLO 版本、模型大小、输入尺寸 | **UNKNOWN** | 模型负责人给出任务、版本、checkpoint SHA256、类别顺序 |
| 精确 SoC / 板卡型号 / PCB 修订 | **UNKNOWN** | 孙璞月核对交付单、板卡手册、可见标识 |
| 安装 RAM、可用 RAM、存储余量 | **UNKNOWN** | 厂商规格 + 获准访问后实际系统日志 |
| SDK、Linux 发行版、内核、libc、浮点 ABI | **UNKNOWN** | 企业交付 SDK/镜像说明 + 系统只读信息 |
| 摄像头接口、供电、像素格式、驱动 | **UNKNOWN**；1080p 只是路线目标 | 杨熙珂与孙璞月核对摄像头及板卡手册 |
| 板卡访问方式 / 操作授权 | **UNKNOWN** | 企业说明串口/显示器/其他入口与允许操作范围；不提交账户或连接标识 |
| 板端 CPU、延迟、RSS、模型加载能力 | **BLOCKED** | 实板运行日志；桌面结果不能替代 |

现有 [INFERENCE_CONTRACT.md](INFERENCE_CONTRACT.md) 仍规定单主食分类，输入为解码 RGB、输出 Top-3 类别。未经共同决策，不能把检测输出塞进分类接口，也不能把“YOLO”自动解释为检测。类别与数据划分继续遵守 [DATA_CONTRACT.md](DATA_CONTRACT.md)。

## 2. RK3506 的可核实边界

Rockchip 官方产品页列出三核 Cortex-A7，以及以下系列差异；这些是芯片规格，不是已收到板卡的配置。[Rockchip RK3506](https://www.rock-chips.com/a/en/products/RK35_Series/2025/1208/2126.html)

| 系列型号 | 官方内存相关条目 | 对本项目能得出的结论 |
| --- | --- | --- |
| RK3506B / RK3506J | 16-bit DDR2 / DDR3 / DDR3L 接口 | 容量取决于交付板设计，不能填成固定 64/128/256/512 MB |
| RK3506G1 | 64 MB DDR2 | 只有确认具体 SKU 后才适用；还要测 Linux 可用余量 |
| RK3506G2 | 128 MB DDR3L | 同上，不能把芯片内存数字当作模型可用 RSS 预算 |

Arm 官方将 Cortex-A7 归入 Armv7-A。因此部署准备以 **32 位 ARMv7 Linux CPU** 为起点，仍须确认交付系统 ABI 和实际 CPU 特性。[Arm 架构与内核表](https://support.arm.com/tools-and-software/arm-toolchain-for-embedded)

Rockchip 此页列出 USB2.0 OTG 等外设，但没有证明交付板可直接使用某一摄像头、ISP、GPU 或 NPU 后端。**项目判断**：先以 MNN CPU 后端核验；没有厂商对应 SKU/SDK 文档及运行日志时，不承诺 RKNN、GPU/NPU 加速、1080p 帧率或实时性。不能套用 RK3566/RK3588 等型号的指标。[Rockchip RK3506](https://www.rock-chips.com/a/en/products/RK35_Series/2025/1208/2126.html)

1080p 相机输入不等于网络输入 1920×1080。**容量估算，非测量**：一张 1920×1080 RGB8 图约 5.93 MiB，RGB float32 约 23.73 MiB；解码、缩放、输入拷贝、激活和线程工作区还会叠加。应缩短大图缓冲生命周期，先测 batch=1，拒绝仅凭 `.mnn` 文件大小判断能否装入 RAM。

## 3. 分类与检测需要两套契约

| 环节 | 单主食分类 | 多目标检测（待批准） |
| --- | --- | --- |
| 数据 / 标签 | 整图一个主类别，混合食材样本不能直接导出分类训练 manifest | 需每个目标框与类别；现有整图标签不能自动变成框标签 |
| 预处理 | 固定 RGB/BGR、resize/crop、插值、归一化、NCHW/NHWC；复用训练验证变换 | 固定 resize/letterbox、pad 值、stride、缩放比例、布局；保留映射回原图的信息 |
| 网络原始输出 | 检查实际导出形状及 logits/概率；softmax 只能按真实导出语义应用一次 | 输出形状与 YOLO 代际、检测头、导出参数有关；不得硬编码某一版本的通道数或转置 |
| 后处理 | 类别顺序匹配 frozen classes，排序 Top-k；分数不是正确率 | 解码框、置信度筛选、坐标还原、裁边；是否需要 NMS 由实际版本/导出图决定 |
| 验证 | 同一输入张量的输出误差、Top-1/Top-3 一致性，之后才做正式分类评估 | 同一原图的类别/框/分数及匹配误差，之后按冻结检测集评估 |
| RK3506 风险（项目判断） | crop 可能漏掉偏边食材；大输入和多次拷贝增加内存 | 大中间张量、候选框与后处理增加内存/时间；小目标、遮挡和反光使量化回归更敏感 |

官方 Ultralytics 分类文档说明默认验证/推理使用 CenterCrop，`result.probs` 表示类别向量；检测文档的 `result.boxes` 表示框、类别和分数。这些高层 Python 结果不等于裸 MNN tensor 的固定格式。[分类](https://docs.ultralytics.com/tasks/classify/)、[检测](https://docs.ultralytics.com/tasks/detect/)

官方 predict 文档区分固定输入与最小矩形 padding，并说明新旧模型可采用不同 NMS 路径。应保存所选版本的完整前后处理清单及参考张量；不能照搬当前网页示例模型名作为本项目选型。[Ultralytics predict](https://docs.ultralytics.com/modes/predict/)

## 4. MNN 版本、转换与“INT8”证据

2026-10-29 实时读取时，GitHub Releases 将 **3.6.1** 标为 Latest；Read the Docs 的版本菜单同时列出 3.6.2。记录此差异，不据此自动升级或认定本地版本。项目实际采用版本仍须从工具日志确定，并锁定 converter、Python binding、C++ headers/runtime、构建 commit 与模型来源。[官方 Releases](https://github.com/alibaba/MNN/releases)、[当前转换文档](https://mnn-docs.readthedocs.io/en/latest/tools/convert.html)

| 产物 / 设置 | 它能证明什么 | 它不能单独证明什么 |
| --- | --- | --- |
| 普通 FP32 `.mnn` | ONNX 等格式经转换器生成模型 | 实板能运行、预处理一致、速度或精度达标 |
| `--fp16` | 指定权重的存储压缩 | Cortex-A7 获得 ARMv8.2 FP16 运算能力 |
| `--weightQuantBits 8` | 权值压缩；默认运行会还原浮点 | 已校准的全 INT8 网络、峰值 RAM 减为 1/4、板端加速 |
| 权值量化 + `MNN_LOW_MEMORY=ON` + `Memory_Low` | 官方提供动态量化计算路径 | 每个算子都 INT8、所选 ARMv7 构建必定更快 |
| `quantized.out` / `mnnquant` + 校准集 | 离线量化，估计激活量化参数 | 没有浮点算子、量化精度或实板吞吐已验收 |

以上行为来自官方转换/压缩文档；是否覆盖本模型算子、是否发生回退及实际收益，须查看转换日志、图结构和目标端运行，不能靠文件名含 `int8` 判断。[转换参数](https://mnn-docs.readthedocs.io/en/latest/tools/convert.html)、[模型压缩与离线量化](https://mnn-docs.readthedocs.io/en/latest/tools/compress.html)

离线量化准备顺序：

1. 保留未压缩 FP32 MNN 基线、checkpoint/ONNX/MNN SHA256、导出参数、输入输出名称/形状/类型，先完成同输入数值对齐。
2. 从 **train** 划出有来源、许可、图像 SHA256、物理样本/拍摄 session/group 的代表性校准子集。不能用 frozen test/external_test；val 只用于选择方案和比较。新旧数据联合重训时也保持全局分组隔离。
3. 校准必须接收与真实运行相同的张量：颜色、crop/letterbox/pad、归一化、形状和布局均一致。官方示例中的 RGB/mean/normal/224 不可当成 YOLO 参数；若图像模式不能复现实际变换，先生成同处理的校准输入并验证工具输入模式，不能直接压缩。
4. 保存实际使用的 `quant.json`、校准清单与数量、工具版本/命令/退出码/日志。官方给 KL 100–1000 张的一般参考，**不是本项目采集数量或精度保证**。少量到货图片只用于链路烟测，不能冒充代表性校准集。
5. 对 FP32 和 INT8 运行同一批准的 val 样本，记录原始输出误差与任务指标；量化调参后不得借 frozen test 反复选方案。保留跳过量化算子和浮点边界的说明。

官方离线量化使用 `quantized.out origin.mnn quantized.mnn quant.json` 或 Python `mnnquant`；旧版工具文档明确量化模型输入输出仍可为浮点。因此浮点 I/O 也不等于网络内部未量化。[离线工具](https://mnn-docs.readthedocs.io/en/latest/tools/quant.html)、[官方量化 README](https://github.com/alibaba/MNN/blob/master/tools/quantization/README_EN.md)

## 5. Linux ARMv7 构建准备（未执行）

**本机只读观察，2026-10-29**：分别执行 `Get-Command wsl.exe -ErrorAction SilentlyContinue | Select-Object Name,Source` 与 `Get-Command bash.exe,ubuntu.exe,docker.exe -ErrorAction SilentlyContinue | Select-Object Name,Source`，仅观察到 `wsl.exe`、`bash.exe` launcher。没有执行这些程序，没有启动/安装 WSL、Docker、发行版或未知主机。原生 Ubuntu/Linux 可执行环境 **BLOCKED（未确认）**；launcher 存在不证明发行版已装好或可用。

官方 Linux 编译页给出 CMake ≥3.10、GCC ≥4.9 或 Clang 的基础要求，并提供跨平台 CMake toolchain 流程。它不是每个 SDK 的完整适配说明；以锁定版本的源码和厂商工具链为准。Android `build_32.sh` / NDK 产物不能直接当 Linux glibc/musl 产物。[MNN 主库编译](https://mnn-docs.readthedocs.io/en/latest/compile/engine.html)

构建前需补齐：Linux 构建入口；所选 MNN tag/commit；厂商 **ARMv7 Linux** C/C++ compiler、sysroot、CMake toolchain；hard/soft-float ABI 与 NEON 特性；目标 libc/loader、libstdc++/libgcc、线程库；图像解码实现依赖及许可。先检查 SDK 已含组件，不安装系统 Python/CUDA，不改全局环境。Python 只用项目 `.venv`；不能假设 x86 wheel 有 ARMv7 对应 wheel。

主机端 converter/quant 工具和板端 runtime 分开构建。官方开关分别为 `MNN_BUILD_CONVERTER`、`MNN_BUILD_QUANTOOLS`、`MNN_BUILD_BENCHMARK`；不需要板上放训练/转换依赖。[其他工具编译](https://mnn-docs.readthedocs.io/en/latest/compile/other.html)

下列是待 SDK 确认后填写的 **Linux shell 模板，未执行、非现成板卡命令**。三个变量须指向已核准、独立的源码/构建目录与厂商工具链，不得指向现有环境或其他项目：

```sh
cmake -S "$MNN_SOURCE" -B "$MNN_BUILD_ARMV7" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$BOARD_SDK_TOOLCHAIN_CMAKE" \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=armv7 \
  -DMNN_BUILD_SHARED_LIBS=ON \
  -DMNN_ARM82=OFF -DMNN_SUPPORT_FP16_ARMV7=OFF \
  -DMNN_OPENCL=OFF -DMNN_VULKAN=OFF -DMNN_CUDA=OFF \
  -DMNN_BUILD_CONVERTER=OFF -DMNN_BUILD_QUANTOOLS=OFF \
  -DMNN_BUILD_BENCHMARK=ON
cmake --build "$MNN_BUILD_ARMV7" --parallel 2
```

这是项目的 CPU 基线建议；线程、动态量化、算子裁剪在实板基线以后再比较。硬浮点设置必须与 SDK 相符，不能盲目设置 `MNN_BUILD_HARD` 或手写 `-mfloat-abi`。3.6.1 源码明确区分 ARMv7 与 ARMv8.2 FP16，并警告 `MNN_SUPPORT_FP16_ARMV7` 可能使库不适用于 32 位 CPU。[3.6.1 CMakeLists](https://github.com/alibaba/MNN/blob/3.6.1/CMakeLists.txt)

产物检查先在主机读 `file`、`readelf -h/-A/-l/-d`：应与板卡 ISA、ELF 位数、浮点 ABI、动态加载器、NEEDED 库一致；再核对依赖版本。`CMAKE_SYSTEM_PROCESSOR=armv7` 只是一项配置，不能修正错误工具链。不要在主机执行未知目标二进制，不用系统安装或刷固件来绕过缺依赖。

## 6. Python 与 C++ runtime 对齐点

Python Session 路径为 `MNN.Interpreter` → `createSession` → `getSessionInput` → `copyFrom` → `runSession` → `getSessionOutput` → `copyToHostTensor`。显式指定 CPU/线程；依据模型读取名称、形状、类型、布局，不能默认首个输出就是分类概率。此路径可用于桌面对齐，不代表 ARMv7 Python wheel 已就绪。[MNN Python 推理](https://mnn-docs.readthedocs.io/en/latest/inference/python.html)

C++ 对应 `Interpreter::createFromFile`、`ScheduleConfig`、`createSession`、`getSessionInput`、`runSession`、`getSessionOutput`；对 host/device tensor 做正确拷贝，并检查空指针和返回码。错误、超时、OOM 或模型不存在应明确报错/`model not ready`，不得生成预测。Python/C++ 必须消费同一参考输入并使用相同类别映射。[3.6.1 Interpreter.hpp](https://github.com/alibaba/MNN/blob/3.6.1/include/MNN/Interpreter.hpp)

## 7. 测量口径与验收证据

所有阈值（延迟、峰值 RSS、CPU、模型包大小、量化可接受误差）当前 **UNKNOWN，待三人及企业确认**。不从其他平台 benchmark 推算 RK3506 FPS。记录板卡/SKU/OS、模型 SHA256、MNN commit/构建开关、输入形状、线程数、后台负载、采样数、预热数、是否含 I/O、原始日志与退出码。

| 指标 | 记录方法（待实板运行） | 口径限制 |
| --- | --- | --- |
| 模型体积 | 对 `.mnn` 及外置权重分别记录字节数和 SHA256，再汇总 runtime/依赖总包大小 | 文件大小不等于驻留内存；不得漏外置权重 |
| load time | monotonic/steady clock 计时模型文件加载、Interpreter 创建、Session 创建/resize 分段；另记首次推理 | 进程冷启动与 OS 文件缓存冷状态分开命名；不清系统缓存、不为测量重启设备 |
| inference-only | 完成输入准备后计时 `runSession`，输出取回/同步边界明确；预热和样本循环独立记录 | 同一 tensor 重复运行是 kernel benchmark，不是业务图像吞吐 |
| 端到端 | 从输入图像字节读取/解码起至最终结构化结果，分记 decode/preprocess/copy/run/copy/postprocess | 相机采集等待另作 camera-to-result 指标；尚未开相机就标 N/A/BLOCKED |
| CPU 时间/占用 | GNU `/usr/bin/time` 的 user/system/elapsed 与 `%P`；可用时采样进程 CPU | `(user+system)/elapsed×100%` 可超过 100%，不混成整板 CPU%；load average 也不是 CPU% |
| RSS / peak RSS | `/proc/<pid>/status` 的 VmRSS/VmHWM；可用时 `smaps_rollup` 核对；GNU time `-v` 的 Maximum RSS | 进程整体包含 runtime/图像缓冲/模型；Python 内存对象统计不能代替 RSS |
| 数值与任务指标 | 同张量桌面 PyTorch/ONNX→FP32 MNN→INT8 MNN；实板再与桌面 MNN 比较 | smoke、数值对齐、正式评估分别报告；test 仅最终冻结评估 |

GNU time 的 wall/user/system/CPU/Maximum RSS 定义见 [官方手册](https://www.gnu.org/software/time/manual/time.html)。目标系统若仅有 BusyBox 或没有 GNU time，则记录工具缺失，不冒用 `-v` 输出；替代方法须说明。Linux 文档指出 VmRSS 的异步统计可能不精确，`smaps`/`smaps_rollup` 更精确但开销更高；支持性与权限以目标内核为准。[Linux proc 文档](https://www.kernel.org/doc/html/latest/filesystems/proc.html)

可用时，GNU time 包住实际批准的离线 runner，例如 `/usr/bin/time -v -o resource.txt <runner-and-approved-arguments>`；这是未执行的占位写法，不是已存在可运行程序。每个 FP32/INT8 方案使用独立进程，先量 load/first-run 峰值，再量稳定循环，原始日志本地保留并脱敏。

MNN `benchmark.out` 支持 loop/warmup/backend/thread/precision；使用时显式选 CPU、线程、Normal precision 并核对所锁版本帮助，不能照抄默认 4 线程/Low precision。算子定位可参考 `timeProfile.out`，其 callback 会引入额外测量开销，不能替代业务端到端延迟。[Benchmark 文档](https://mnn-docs.readthedocs.io/en/latest/tools/benchmark.html)、[3.6.1 timeProfile 源码](https://github.com/alibaba/MNN/blob/3.6.1/tools/cpp/timeProfile.cpp)

**放行顺序**：规格与任务确认 → Linux/SDK 可用 → 同输入 FP32 对齐 → 真实校准与 INT8 对齐 → ARMv7 构建/依赖检查 → 实板离线少量图 → 实板性能/内存 → 获准后的相机链路。任一步缺证据均保留 BLOCKED；桌面成功不能跨过实板步骤。
