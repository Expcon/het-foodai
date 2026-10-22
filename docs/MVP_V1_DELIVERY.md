# HeT FoodAI — MVP v1.0 首版交付说明

集成开始：2026-10-21；交付候选验收：2026-10-22。目标为单张、主体明确的八类蔬菜分类，供课程展示和程序复现。当前不包含混合食材检测或真实烤箱部署。交付分支为 `integration/mvp-v1.0-delivery`，评审目标为 main；不自动合并或创建正式 Release。

## 分工和已完成范围

| 负责人 | 已完成 | 后续工作 |
|---|---|---|
| 陈谦：模型/训练 | EfficientNetV2-S 真实训练、验证集选择、冻结 test 评估、安全 checkpoint 与共享预处理 | 独立场景评估、增量学习 |
| 杨熙珂：数据/QA | Dataset v1.1.0、manifest、分组去重、来源与许可台账、版本包和质量报告 | 未知许可和黄瓜语义复核 |
| 孙璞月：应用/部署 | 真实权重适配、CPU/GPU Gradio、上传/Top-3/时延和错误提示 | ONNX 与真实端侧验收 |
| 最终集成 | 保留双方 Git 历史，统一冻结入口，修正验收统计并重新实测；详见本次验证记录 | 教师/三人评审后决定合并与发布 |

## Dataset v1.1.0 与冻结契约

总数 10,662；train **8,527**、val **1,068**、test **1,067**，external_test **0**。磁盘验证目录名为 `valid`，manifest 中的 split 为 `val`。不要按目录排序重新编号或拼接旧版本清单。

| ID | key | 中文 | train | val | test |
|---:|---|---|---:|---:|---:|
| 0 | carrot | 胡萝卜 | 999 | 125 | 125 |
| 1 | potato | 土豆 | 1165 | 146 | 146 |
| 2 | broccoli | 西兰花 | 1196 | 150 | 150 |
| 3 | pumpkin | 南瓜 | 1180 | 148 | 148 |
| 4 | tomato | 西红柿 | 1108 | 139 | 138 |
| 5 | white_radish | 白萝卜 | 960 | 120 | 120 |
| 6 | cucumber | 黄瓜 | 960 | 120 | 120 |
| 7 | bell_pepper | 甜椒 | 959 | 120 | 120 |

正式入口 `configs/classes.json` 已为 `status: frozen`、`version: v1.1.0`；数据包的 `classes.proposed.json` 保持原历史 draft，不用作正式推理入口。旧鸡胸肉/鲈鱼/虾映射和旧 checkpoint 不兼容本版。

| 文件/含义 | SHA256 |
|---|---|
| `data/manifest.csv`，与 `data/processed/selected_dataset_v1.1.0/manifest.csv` **字节一致** | `6bef9776c278d8efac86008a11c47386cb7d95b7164073b9bd07e84cb5cf2821` |
| 正式 `configs/classes.json` 文件字节 | `27d5337826c26a3b743ec0928ea8da050f4ae7bf0ef31e7efb051e4e907e9614` |
| 模型内嵌有序类别 `class_list_sha256` | `afdc69e898ef6067b469b9f90eea3dfed84fbb76e7a5910ee1826a62f02326af` |
| Baseline `best.pt` | `41056b3ea90d5fa718a18f48cf5182373d1010c5185cb426ba5c949b1ec1f4c5` |
| v1.1.0 图片 Release ZIP | `22c0162b686d13b45c88982f1615d70030c8376fba4d9e25091995fb83478a02` |

`.gitattributes` 保留 Yang 的所有数据元数据字节规则，并保护正式 classes.json，防止 Windows Git 换行转换改变训练记录中的文件哈希。

来源记录：Kaggle vegetable 10,192 张（CC-BY-SA-4.0）、Roboflow fridge genie 235 张（CC-BY-4.0）、未匹配来源 **235 张（unknown）**。许可字段是交付记录，不是独立法律鉴定；unknown 不代表可自由再分发，课程使用也不消除核实责任。不要未经核实将图片纳入面向企业或公开再分发的交付包。详见 [数据报告](../DATASET_V1.1.0_REPORT.md)与随图片包提供的 provenance/audit 台账。

黄瓜 `0225.jpg`（image_id `02cbd0207e34cc88be1b6f90`，val）仍登记在 `audit/visual_review_pending.csv`：圆形深绿色主体的品种或类别待确认。本次不改原标签或冻结划分；若后续修正，发布新数据版本和新哈希。近重复分组没有证明未知拍摄/增强关系全部被发现。

## 模型、训练与效果边界

结构为 torchvision EfficientNetV2-S，ImageNet 预训练初始化，分类头输出 8 类，输入 **224×224 RGB**。训练与推理复用 `foodai.vision.get_transforms/v1`；验证/推理先 Resize(255)，CenterCrop(224)，ToTensor，再按 ImageNet mean/std 归一化。训练增加随机裁剪、水平翻转和轻微 ColorJitter。

Baseline 模型 ID：`stage8-20261020-164500-40a26d`。原始训练记录：Python 3.12.3、torch 2.8.0+cu128 / torchvision 0.23.0+cu128、RTX PRO 6000 Blackwell Server Edition；AdamW，lr 0.0003，weight_decay 0.0001，batch 16，10 epoch，seed 42，CUDA AMP float16、确定性算法开启。每轮实际 train 8,527、val 1,068；按 val accuracy 选最好，平局保留最早 epoch，选中第 7 轮；test 不参与选择。

恢复的原始 `test_evaluation.json` 为同源冻结 test **1067/1067，Top-1=100%，macro recall=1.0**。该值只对应上述 manifest、权重和八类映射；同源背景相似、来源与未知增强关系都可能使评估偏乐观。不能表述为所有食材、网络图片、真实烤箱或独立跨场景准确率，也不能把 softmax 当校准后的正确概率。本次固定该权重做部署核验，不用 test 调参或选择其他实验。

## Windows Python 3.12 从零部署

先准备 Git 与 Python 3.12。全部 Python 包安装在新的项目 `.venv`，不用系统级 CUDA/torch 安装，也不要覆盖现有实验环境。以下 CPU 路径无需独显：

```powershell
git clone --branch integration/mvp-v1.0-delivery https://github.com/Expcon/het-foodai.git HeT_FoodAI_Delivery
cd HeT_FoodAI_Delivery
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt gradio==6.30.0 pillow==12.3.0 numpy==2.5.2
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

torch 2.8.0 / torchvision 0.23.0 的对应关系和 CPU wheel 命令来自 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v280)。本次集成机器另用现有 torch 2.14.1+cu130 / torchvision 0.29.1+cu130、RTX 5080 Laptop 验证 CUDA；不要为展示覆盖已经可用的 GPU 环境。GPU 安装应另建项目环境并按官方驱动/版本选择器匹配，不要求老师安装本机 GPU 组合。

## 恢复单独交付的真实模型

Git 不含权重。取得 `foodai-mvp8-baseline-v1.1.0.tar.gz` 后，先查看成员，仅恢复已知 Baseline 目录中的四个文件；不从模型包覆盖代码或默认 manifest。

```powershell
$bundle = Read-Host '请输入模型 tar.gz 的本地路径'
$run = 'runs/food-v1.1.0-baseline/stage8-20261020-164500-40a26d'
tar -tzf $bundle
tar -xzf $bundle "$run/best.pt" "$run/run_info.json" "$run/history.csv" "$run/test_evaluation.json"
$checkpoint = "$run/best.pt"
$expected = '41056b3ea90d5fa718a18f48cf5182373d1010c5185cb426ba5c949b1ec1f4c5'
if ((Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLower() -ne $expected) { throw 'checkpoint SHA256 mismatch' }
if ((Get-FileHash configs/classes.json -Algorithm SHA256).Hash.ToLower() -ne '27d5337826c26a3b743ec0928ea8da050f4ae7bf0ef31e7efb051e4e907e9614') { throw 'classes SHA256 mismatch' }
```

`configs/classes.json` 已随代码交付；模型包内同名文件仅用于核对。`run_info.json` 是追溯材料，ready 必须来自 checkpoint 内嵌 metadata、`model_state` 的严格安全加载及目标设备完整前向；不以 sidecar 的文字状态代替模型验证。

## 恢复 Dataset 与校验

仅演示自己提供的 JPG/PNG 不需要下载全量图片。需要复现数据检查或冻结 test 时，从 [官方项目 Dataset v1.1.0 附件](https://github.com/Expcon/het-foodai/releases/tag/v1.1.0)取得具名 ZIP 和 SHA256SUMS；GitHub 自动 Source code ZIP 不含真实图片。先复用本地缓存并校验，不重复下载；确认版本后在仓库根目录解压。不要用图片包中的不同版本文件覆盖现有清单。

```powershell
$datasetZip = Read-Host '请输入 v1.1.0 数据 ZIP 路径'
if ((Get-FileHash -LiteralPath $datasetZip -Algorithm SHA256).Hash.ToLower() -ne '22c0162b686d13b45c88982f1615d70030c8376fba4d9e25091995fb83478a02') { throw 'dataset SHA256 mismatch' }
# 新克隆：先解到 Git 忽略的临时目录，只复制图片和 audit，避免覆盖代码/清单
$staging = 'data/cache/restore-v1.1.0'
if (Test-Path $staging) { throw '请先使用新的空 staging 目录；已有图片可直接运行校验' }
Expand-Archive -LiteralPath $datasetZip -DestinationPath $staging
if ((Get-FileHash "$staging/data/manifest.csv" -Algorithm SHA256).Hash.ToLower() -ne '6bef9776c278d8efac86008a11c47386cb7d95b7164073b9bd07e84cb5cf2821') { throw 'manifest mismatch' }
foreach ($part in @('train', 'valid', 'test', 'audit')) {
    $target = "data/processed/selected_dataset_v1.1.0/$part"
    if (Test-Path $target) { throw "已有 $part；先校验现有数据，不覆盖" }
    Copy-Item -LiteralPath "$staging/data/processed/selected_dataset_v1.1.0/$part" -Destination $target -Recurse
}
.\.venv\Scripts\python.exe scripts/validate_manifest.py --manifest data/manifest.csv --classes configs/classes.json --check-files
```

上述流程保留 Git 中的 manifest、类别与 summary，全部运行路径仍指向版本目录。已恢复过图片时先直接运行 validator，不重复解压覆盖。校验会报告 235 张未核实许可的 warning，这项尚未解决。

## ready、自检和 Gradio

```powershell
.\.venv\Scripts\python.exe app/app.py --check --checkpoint $checkpoint --classes configs/classes.json --device cpu --expect-checkpoint-sha256 $expected
# 必须 exit 0、state=ready、8 类、224px，且两个模型/类别哈希匹配
.\.venv\Scripts\python.exe app/app.py --checkpoint $checkpoint --classes configs/classes.json --device cpu --host 127.0.0.1 --port 7860
```

在本机浏览器打开 `http://127.0.0.1:7860`，上传真实 JPG/PNG，查看图片、Top-3 和时延。已验证 GPU 环境可将 `--device cpu` 改为 `--device cuda`；加载失败不会自动伪造结果或回退成功。未上传应提示“请先上传图片”；损坏文件应拒绝且不显示分数；不传 checkpoint 时按钮禁用并显示未就绪。换图会清空旧结果，预测期间上传和预测控件锁定。

`--check` 缺模型时预期 exit 1；冻结配置/权重不一致也不能显示 ready。关闭终端的本地服务可用 Ctrl+C。

## 修正后的时延验收与证据

对真实图片目录执行新脚本；输出写到新的文件，不覆盖历史记录：

```powershell
$images = Read-Host '请输入真实 JPG/PNG 目录'
.\.venv\Scripts\python.exe scripts/acceptance_run.py --checkpoint $checkpoint --classes configs/classes.json --images-dir $images --device cpu --repeats 5 --out runs/acceptance-corrected/cpu.json
# 已确认 CUDA 可用时另测：--device cuda --repeats 10 --out runs/acceptance-corrected/cuda.json
```

每张图先预热 **1 次**，再重复 N 次；每次收集 `PredictionResult.latency_ms` 和外部 `perf_counter`。内部中位数是真实 N 次内部时延的 median，偶数 N 取中间两项均值。墙钟时延包围整个 `adapter.predict`，含文件解码/校验、预处理、数据传输、前向、softmax、Top-3 与 Python 结果组装；内部计时不含解码/校验，但含预处理、H2D/前向/Top-3/D2H，CUDA 起止同步。两者都排除首次模型加载、预热、上传、页面渲染。

P95 使用 **nearest rank**：排序后第 `ceil(0.95*N)` 项，不外推；N<20 时等于最大样本。新 JSON 保存原始两组样本、方法和重复次数，汇总显示保留两位小数。5/10 次只是功能验收的小样本描述，不能当稳定延迟基准或端侧性能。空/不存在的图片目录或非正整数 repeats 返回 exit 2；错误路径检查失败仍保存报告，但返回 exit 1，不能当验收通过。

Sun 的 [历史验收说明](ACCEPTANCE_REAL_MODEL_2026-10-21.md)与 `docs/acceptance-evidence/20261021/*.json` 原样保留。旧 JSON 名为 internal_median 的字段实际为最后一次内部时延，旧 P95 为 exclusive 分位外推，旧脚本仅对首图预热一次；不能将这些数值改名为本次修正后重测结果。

本次独立证据、命令退出码、截图和重新克隆部署结果统一见 [交付验证记录](delivery-evidence/20261021/verification.json)。本次功能样本来自 v1.1.0 的已知许可 train 图片，每类一张，另将真实胡萝卜 JPG 编码为 PNG 测试格式；没有合成图，也不是独立泛化评估。原图不进 Git，来源相对路径及 SHA 单独留档。

## 本次集成改动与实测结果

从 Sun 的 `8d6fda0` 建立交付分支，合入 Yang 的 `524879a`，合并提交 `8f613fa`：**零冲突**，保留双方父提交。数据清单/工具/报告、`.gitignore` 与已有 `.gitattributes` 采用 Yang 最新版；Git 对比确认训练、评测、安全 loader、预测适配器及 Gradio 均保持 Sun/PR #4 的实现。旧版 manifest 仅作历史，不进入本版运行入口。

本次必要修改文件：

- `.gitattributes`：为正式 classes.json 加入字节保护；类别语义与顺序没有变化。
- `scripts/acceptance_run.py`、`tests/test_acceptance.py`：真实内部中位数、nearest-rank P95、逐图预热、原始样本、参数/空目录校验，以及错误路径失败的非零退出码；新增 5 项回归测试。
- `README.md`、本说明、`docs/ACCEPTANCE_REAL_MODEL_2026-10-21.md`：更新真实模型状态、可执行部署/恢复步骤和历史时延勘误；原历史验收 JSON Git 内容不变。
- `docs/delivery-evidence/20261021/`：新增真实验收、训练追溯选摘、原始 test 与本机复核、来源记录、两张网页截图、独立环境版本快照及验证索引；不提交权重/原图/虚拟环境。
- `docs/superpowers/plans/2026-10-21-mvp-delivery.md`：集成计划与验证记录。

下表 `python` 表示所述环境的项目 `.venv` 解释器；完整 stdout/stderr 保留在本地 ignored 日志，机读摘要见 [verification.json](delivery-evidence/20261021/verification.json)。

| 实际命令/操作 | 结果 |
|---|---|
| `python -m unittest discover -s tests -v`，主 GPU 环境 | exit 0；68/68，0 失败、0 跳过，19.319s |
| `python scripts/validate_manifest.py --manifest data/manifest.csv --classes configs/classes.json --check-files` | exit 0；10,662 行/全部文件 SHA 通过；235 张未核实许可 warning 保留 |
| `python app/app.py --check --checkpoint runs/food-v1.1.0-baseline/stage8-20261020-164500-40a26d/best.pt --classes configs/classes.json --device cpu`，另测 `cuda` | 各 exit 0、ready、8 类、224px；实测还传入完整 `--expect-checkpoint-sha256` |
| `python scripts/acceptance_run.py --checkpoint runs/food-v1.1.0-baseline/stage8-20261020-164500-40a26d/best.pt --classes configs/classes.json --images-dir runs/delivery-20261021/real-images --device cpu --repeats 5 --out runs/delivery-20261021/acceptance-cpu-final.json`，另测 CUDA/repeats 10/独立输出 | 各 exit 0；9 个真实图片文件产生 Top-3，损坏输入与缺模型检查通过；不是准确率或延迟基准 |
| `python scripts/evaluate_classifier.py --checkpoint runs/delivery-20261021/held-out/best.pt --manifest data/manifest.csv --classes configs/classes.json --split test --batch-size 16` | exit 0；固定同一 SHA 权重的 1067/1067 复核；权重副本只为将新报告与历史文件隔离，未调参 |
| 本机 Gradio CPU/CUDA 实际 HTTP 路由 | JPG/PNG 上传、Top-3、无输入、损坏文件、缺模型和禁用按钮断言通过；组合脚本后续等浏览器完成时曾超时 exit 1，不能称整个组合脚本 exit 0 |
| 本机浏览器 | 真实 CPU JPG 上传、预览、Top-3 和未上传提示有本次截图；之后 origin 权限被拒绝，清除/更多页面复测保持 BLOCKED/NOT_REPEATED；未绕过 |
| GitHub 新克隆 + 新 `.venv`，torch 2.8.0+cpu / torchvision 0.23.0+cpu | 安装 exit 0；恢复前自检 exit 1/not_ready，恢复后 exit 0/ready；9 个真实文件/每图 3 次 CPU 验收 exit 0；Gradio 实际启动并 GET config HTTP 200/ready |
| 新克隆 `python -m unittest discover -s tests -v` | exit 0；68 项中 67 通过、1 项 CUDA 专用计时测试跳过、0 失败，12.487s；对应 CUDA 路径已在主环境通过 |

独立克隆完整测试的代码提交为 `6ffe216`；后续只有交付文档与证据更新，模型/应用/验收代码的规范化 SHA 与该副本完全一致。主工作区 main 没有变化；没有自动合并 PR 或创建正式 Release。

新 CPU 环境完整依赖快照为 [environment-cpu.txt](delivery-evidence/20261021/environment-cpu.txt)。需要固定全部解析后的依赖时，可在新 `.venv` 使用 `python -m pip install -r docs/delivery-evidence/20261021/environment-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu`；日常演示沿用上面的已实测安装命令即可。


## 尚未完成与交付判定

- 独立跨来源/真实烤箱场景正式评估与完整语义复核；external_test 当前为空。
- 235 张 unknown 许可的核实与再分发授权，黄瓜待复核样本的团队决定。
- 8→12→16 增量学习、遗忘度量；当前仅预留有序前缀与分类头阶段接口。
- ONNX 导出验证、真实板端运行、延迟/内存/功耗验收。

本次代码、固定权重、数据版本、真实 CPU/GPU 推理与 Demo、干净克隆部署验收已满足课程展示范围，可作为 **mvp-v1.0.0 课程展示候选**。未知许可素材的再分发与真实场景性能不能随此候选自动获准；正式合并和 Release 仍由团队决定。
