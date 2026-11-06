# 第二轮真实烤箱实验：结果与现场使用

执行日期：2026-11-07。分支 `feat/model-oven-second-round` 基于 PR #8 的 `20e71f3`，保留第一轮分支、九类模型和八类 MVP。没有合并 main，没有操作烤箱、摄像头或串口。

**结论：十二类模型已实际训练、导出并接入 Desk 实验模式；独立泛化验收仍为 BLOCKED。** 本轮有 1,428 张新图、1,996 张去重联合训练候选。数据缺少可证明独立的实物身份，绝大多数标签尚未逐张人工确认，不能发布正式准确率。v2 对新增训练图的薯条、可颂目录标签全部一致，但土豆和香肠仍有明显错误，不能宣布全面提升。

## 数据与标签

完整审计见 [OVEN_ROUND2_DATA_AUDIT.md](OVEN_ROUND2_DATA_AUDIT.md)。新图为 1,154 JPEG + 274 PNG，全部 1920×1080、可解码；新图内部及新旧之间均无字节或像素完全重复。第一轮 569 张中排除 1 张重复，保留 568 张。pHash 找到 1,560 对新—新、1,803 对新—旧近似候选；跨类误报很多，不能据此认定重复、删除图片或认定同一实物。

1,427 份逐图 JSON 全部仍为 `pending_manual_review`，其中 644 个旧候选类别为错误的 `pumpkin`，没有使用这些字段作标签。整理者说明中明确逐张看过的 23 张吐司单独记录为可信诊断标签；其余使用目录弱标签进行探索训练。来源许可和对外传播权限未知，照片与逐图记录不入 Git。

四个拍摄会话、86 个实物编号字符串均不能证明独立实物。十张薯条的当前文件名与历史 JSON 层位冲突，但哈希一致，保留图片并将层位设为 unknown。所有烤箱 ID 和拍摄角度 unknown。`have/no` 不自动解释成锡纸；光照字段仅为来源的 `oven-light`，没有照度测量。

按会话、身份线索、精确重复、近似候选及无法证明独立的同类旧/新样本建立保守关联图，形成一个不确定性组。这不是“所有图片属于一个实物”的结论。仅分配 train / excluded；不生成假独立 val/test，不使用旧八类冻结测试集。

## v1：先诊断、后训练的实际结果

冻结的 v1 `last.pt` 在 CUDA 上完成 1,428 张预测，读取失败 0。保留完整概率、Top-3、可信标签错误和目录不一致记录。九类共有图片 1,000 张；另外 428 张属于玉米、香肠、牛排，不进入九类准确率分母。

只有 23 张吐司具有整理者逐张人工检查的记录，v1 为 **23/23**。这是小规模同源诊断，不是最终独立测试，也没有其他类别的可靠准确率。本轮全部组均没有经过验证的独立实物数量。

下表为**与目录标签一致的张数**，不是经核验准确率。v2 列的这些图片全部参与过 v2 训练，属于训练后回看，不能与 v1 作泛化提升结论。

| ID | 类别 | 新图数 | v1 九类目录一致数 | v2 十二类训练后目录一致数 |
| ---: | --- | ---: | ---: | ---: |
| 0 | 芦笋 asparagus | 112 | 54 | 112 |
| 1 | 培根 bacon | 97 | 44 | 97 |
| 2 | 西兰花 broccoli | 119 | 46 | 119 |
| 3 | 鸡翅 chicken_wing | 129 | 126 | 129 |
| 4 | 可颂 croissant | 110 | 59 | 110 |
| 5 | 薯条 french_fries | 122 | 26 | 122 |
| 6 | 土豆 potato | 114 | 59 | 50 |
| 7 | 三文鱼 salmon | 93 | 45 | 90 |
| 8 | 吐司 toast | 104 | 85 | 104 |
| 9 | 玉米 corn | 252 | 不支持 | 251 |
| 10 | 香肠 sausage | 97 | 不支持 | 71 |
| 11 | 牛排 steak | 79 | 不支持 | 79 |

共有九类：v1 544/1,000；v2 933/1,000。v2 十二类总体 1,334/1,428。这些仍是目录一致数，不能写成独立泛化准确率。23 张吐司在 v2 仍为 23/23，但已进入训练，不再是未见诊断样本。

### 场景分组（v1，共有九类）

| 来源场景标记 | 目录一致数 / 图片数 |
| --- | ---: |
| 明确 without_foil | 303 / 317 |
| 明确 with_foil | 4 / 328 |
| 锡纸 unknown | 237 / 355 |
| high | 181 / 337 |
| mid | 192 / 331 |
| low | 160 / 299 |
| up（不擅自与 high 合并） | 11 / 11 |
| 层位 unknown | 0 / 22 |

按会话匿名顺序的共有类计数分别为 152/251、307/644、85/104、0/0，另有会话缺失 0/1。第四会话只有新增类别，不构造九类准确率。会话完整标识和逐图关联仅保存在本地。来源前缀对应共有类结果为 `HeT-OvenCapture(1)` 152/251、`food_test` 85/104、`烤箱数据图库` 307/645；牛排和香肠来源不含共有类。来源文件夹不是烤箱身份，不能比较“不同烤箱准确率”。

这些分组混合了类别、实物、数量、层位、摆放、光照和背景变化；样本相关性高。11 张 up 和其他小组尤其不能作可靠结论。全部逐图结果与完整类别混淆矩阵保存在 `v1-diagnostic/summary.json` 和 `predictions.json`。

### 薯条和可颂的错误证据

- 薯条 122 张：26 预测薯条，其余为鸡翅 38、吐司 24、三文鱼 14、芦笋 11、可颂 8、培根 1。本批确实重现部分“薯条→可颂”，但不是主要唯一错误。按层位 high 0/23、mid 19/47、low 7/30、unknown 0/22。所有薯条的锡纸字段均为 unknown，因此不发布薯条锡纸分组准确率。
- 可颂 110 张：59 预测可颂、50 鸡翅、1 土豆；**本批没有重现可颂→三文鱼**。明确锡纸组 4/54，明确非锡纸组 55/56。high 22/37、mid 18/36、low 19/37，与锡纸背景有关的关联比简单层位差异更明显，但不构成因果证明。
- 实际查看了 19 张按预测类别及场景选择的失败/一致样例。单根小薯条在暗背景中会被判为芦笋/培根/三文鱼；锡纸占据大面积且反光时可颂和少量薯条多判为鸡翅；也有带锡纸的大可颂被识别正确。另有手持可颂判土豆，提示手部遮挡/摆放干扰。部分无锡纸、较大薯条仍误判，因此不能归结为单一背景因素。
- 第一轮每类各取一张旧样例，均为黑盘背景；这仅是九张代表性图片的观察，不代表完整旧数据的锡纸标注统计。它支持“背景分布变化值得进一步验证”的假设。
- 未提供食材框/分割真值，**没有计算或编造食材像素占比**。代码的整图亮度、暗区和亮中性像素比例只是描述性特征，不代表锡纸材质、反光测量或食材区域。224 等比缩放后 1920×1080 内容约为 224×126，小目标细节损失是待检验假设。

证据图与索引：本地 `v1-diagnostic/focus_review.jpg`、`focus_review_index.json`、`round1_background_examples.jpg`。抽查不将候选标签自动提升为人工真值。

## v2 训练配置与结果

在 RTX 5080 Laptop GPU 上实际执行，固定 30 Epoch；无独立 val/test，无 early stopping，选末轮 `last.pt`。从与 v1 相同、SHA 固定的官方 ImageNet YOLO11n-cls 初始化，重建十二类分类头，不覆盖或改写 v1。IDs 0–8 完全保留；9 corn、10 sausage、11 steak 只属于本实验，不改八类 `configs/classes.json`。

| 配置 | 实际值 |
| --- | --- |
| 输入 | EXIF 校正、RGB、PIL bilinear 等比 contain、居中补边 114、224×224、NCHW、/255 |
| 训练候选 | 568 旧图 + 1,428 新图 = 1,996 |
| 优化器 | AdamW，lr 0.001，weight_decay 0.0005 |
| batch / seed / workers | 32 / 42 / 0 |
| 随机性 | Python/NumPy/Torch 固定种子，deterministic algorithms，cuDNN benchmark false |
| AMP | 关闭，FP32，与 v1 一致 |
| 增强 | 水平翻转 p=0.5；亮度/对比度/饱和度 ColorJitter 0.1 |
| Epoch 30 Loss | 0.0828900344 |
| Epoch 30 增强训练目录标签一致率 | 0.9769539078（不是验证指标） |
| 训练循环时间 | 93.15 秒，不含数据预缓存、模型建立、导出；非设备性能基准 |

`history.csv` 保留每轮 loss、增强训练一致率、样本数与时间；`fixed_config.json`、`run_info.json` 绑定审计 rows、官方初始模型和诊断报告 SHA。命令退出码 0，日志记录 `EXPLORATORY_TRAINING_COMPLETED`。没有根据上述训练后回看结果再次调参或选择另一个 epoch。

A 为 v1 224 原模型；B 为新旧联合、十二类 224 固定配方。本轮同时必要地扩展类别头，且没有合格独立验证组，A/B 只能作描述性回看。C（320/448 或 ROI）不执行，避免在不可靠验证集上进行多项调参。训练/推理预处理保持一致。

v2 剩余目录不一致：土豆→可颂 46、土豆→三文鱼 18；香肠→鸡翅 24、香肠→培根 2；三文鱼→培根 3；玉米→薯条 1。土豆 50/114 低于 v1 的 59/114，**不隐去该退化**。这些图均已参与 v2 训练，需要先复核标签/场景，再收集新实物评估；不能用这批图宣布新场景泛化提升。

## 导出、Desk 与保护检查

安全读取实际 `last.pt`：`torch.load(..., weights_only=True)`，校验 metadata、SDK、类别顺序、模型 SHA 与真实前向。只有精确固定 SHA 的官方预训练文件使用原 SDK 加载方式。没有不安全回退读取本轮 checkpoint。

ONNX opset 17、FP32、静态 `[1,3,224,224] -> [1,12]`，输出为概率，不重复 softmax。全部 1,997 原始记录（包含审计保留的一张重复记录）逐张比较，**Top‑1 与有序 Top‑3 均 1,997/1,997，最大绝对概率误差 4.6789646e-6，阈值 1e-4，非法概率向量 0**。这是格式转换验收，不是食材识别准确率。

使用 Desk 的实际解码/预处理/ModelEngine，在现有桌面构建环境再次检查全部 1,997 张：预处理张量 SHA **1,997/1,997 完全一致**，Top‑1、Top‑3 同上，概率误差同上。训练环境 Pillow 12.3.0 / NumPy 2.5.2 与桌面环境 Pillow 12.0.0 / NumPy 2.2.6 的本批预处理一致性已实际验证。

| 验收 | 结果 |
| --- | --- |
| 全项目 unittest | 最终 176 项，175 通过、1 跳过，失败 0；退出码 0，21.457 秒 |
| 跳过项 | Windows 环境不允许创建目录符号链接测试夹具 |
| ONNX 全量对齐 | PASS，退出码 0 |
| Desk 全量对齐 | PASS，退出码 0 |
| Tk 十二类文件驱动界面测试 | 8 项检查 PASS，退出码 0 |
| Tk 九类兼容界面测试 | 8 项检查 PASS，退出码 0 |
| PyInstaller 新 EXE 构建 | 退出码 0，未安装或修改构建环境依赖 |
| EXE runtime healthcheck | PASS，退出码 0；ONNX Runtime 1.23.2 |
| EXE 真实新 PNG / JPG / v1 兼容 | 三项均 PASS、ready=true，退出码各 0 |
| EXE 错类别 / pending 契约 / 缺失模型 / 损坏图片 | 四项正确拒绝，ready=false，退出码各 1；这是预期负向结果 |
| 原图与元数据保全 | 1,997 张图片 + 1,431 个非图片文件哈希无变化，退出码 0 |
| MVP / v1 / 采集逻辑保全 | 八类 classes/draft、两份 manifest、MVP best.pt、v1 last.pt/ONNX/classes、camera_devices.py、capture_core.py 全部哈希不变 |
| 物理摄像头、其他同学电脑、独立实物测试 | BLOCKED / 本轮未连接或验收 |
| MNN、INT8、RK3506 | 未执行；本轮 ONNX-only EXE 不包含 MNN，不构成端侧部署 |

Tk 测试通过实际 worker/event 路径：无画面拒绝、加载真实模型、文件馈入“当前帧”、忙时拒绝切模型、图集 PNG/JPEG 与损坏文件、空图集拒绝、失败加载清除旧模型、硬件未打开。图集中的格式转换样例是已知真实图片的派生文件，不是新增独立样本。没有真值时只输出预测 CSV，不显示准确率。

导出审查修复了一个批准流程问题：`contract.pending.json` 现在没有可加载的模型 SHA 授权，GUI 必须拒绝；只有全部对齐 PASS 才写 `classes.json`。既有九类契约仍兼容。摄像头兼容逻辑及采集防覆盖逻辑未修改。

### 产物 SHA256

| 文件/含义 | SHA256 |
| --- | --- |
| v2 last.pt | `44878014afe8381055ed2a26fffff6269982e3f5b57c2da0fbedc596127770d5` |
| oven12.onnx（6,202,899 字节） | `8c696ce5dc63a25053198f1781504b4a53cb801f2d971e3bceeba52eed789689` |
| v2 classes.json 文件字节 | `8cd2db1f828f45520f51c8660329254104d0bd4738cb6b957efb8afe7b69a1b5` |
| 十二类有序 key 列表 | `eeb5260a718f2a1bd42a4d600c8cb2e5ee290c98a5a24e38aa1822e7391827f7` |
| EXE v1.2.0-experimental | `0223c7346b87dc90487df30e6c298708f49b7d0bc658d7205ae453f61d135817` |
| 未改动 v1 last.pt | `72367305b65b31920c59865758dcdd4a1ee506751bde57db113f017c5c314b25` |
| 联合审计 rows | `d74e39a55f7e4225b4e023d7cf5762379358ac48aef95ab3b7c827378f963181` |

## 复现命令与本地产物

在仓库根目录设置 `$approvedSourceRoot` 为本次用户授权的数据目录；`$deskPython` 为已经存在的隔离桌面构建环境 Python。以下 `runs` 路径均被 Git 忽略。再次执行请使用新的输出目录，训练和导出拒绝覆盖旧 run。审计版本需与 checkpoint 绑定的 rows 完全匹配。

```powershell
$run = 'runs/oven-round2-20261107'
$rows = "$run/audit/combined_rows.json"
$v1sha = '72367305b65b31920c59865758dcdd4a1ee506751bde57db113f017c5c314b25'
$v2sha = '44878014afe8381055ed2a26fffff6269982e3f5b57c2da0fbedc596127770d5'

.venv/Scripts/python.exe -m tools.oven_training.round2_audit --input-root $approvedSourceRoot --contacts
.venv/Scripts/python.exe -u -m tools.oven_training.round2 diagnose --rows $rows --source-root $approvedSourceRoot --checkpoint runs/oven-first-round-20261106/yolo11n-fixed30/last.pt --expect-sha256 $v1sha --out "$run/v1-diagnostic" --device cuda:0
.venv/Scripts/python.exe -u -m tools.oven_training.round2 train --rows $rows --source-root $approvedSourceRoot --diagnostic "$run/v1-diagnostic/summary.json" --pretrained runs/phase2-edge/export-smoke/yolo11n-cls.pt --out "$run/yolo11n-12cls-fixed30" --device cuda:0
.venv/Scripts/python.exe -u -m tools.oven_training.round2 diagnose --rows $rows --source-root $approvedSourceRoot --checkpoint "$run/yolo11n-12cls-fixed30/last.pt" --expect-sha256 $v2sha --out "$run/v2-resubstitution" --device cuda:0
.venv/Scripts/python.exe -u -m tools.oven_training.export_desk export --checkpoint "$run/yolo11n-12cls-fixed30/last.pt" --expect-sha256 $v2sha --rows $rows --source-root $approvedSourceRoot --out "$run/onnx"
& $deskPython -B tools/oven_desk/tests/round2_alignment_check.py --model "$run/onnx/oven12.onnx" --classes "$run/onnx/classes.json" --alignment "$run/onnx/onnx_alignment.json" --rows $rows --source-root $approvedSourceRoot --out "$run/onnx/desk_alignment.json"
.venv/Scripts/python.exe -m unittest discover -s tests -v
& $deskPython -B tools/oven_desk/tests/gui_model_smoke.py --mode oven_round2 --model "$run/onnx/oven12.onnx" --classes "$run/onnx/classes.json" --image $realPng --out "$run/gui-round2.json"
& $deskPython -B tools/oven_desk/tests/gui_model_smoke.py --mode oven_round1 --model runs/oven-desk-20261106/oven9.onnx --classes runs/oven-desk-20261106/classes.json --image $realJpg --out "$run/gui-round1-compat.json"
```

上述实际审计、诊断、训练、导出、Desk 对齐、单元测试、两项 GUI 测试命令均退出 0。记录分别位于 `audit/audit_run.log`、`v1-diagnostic.log`、`train.log`、`v2-resubstitution.log`、`onnx-export.log`、`desk-alignment.log`、`unittest-final-verified.log`、`gui-round2.log`、`gui-round1-compat.log`。最后补充了“诊断图片数完整、类别/哈希一致、明确零失败才可训练”的回归测试；已保存的真实 v1 诊断也通过最终检查，未重新训练或修改权重。一次临时统计命令因 Windows 默认 GBK 读取 UTF-8 JSON 失败（退出 1），明确指定 UTF-8 后重跑成功；没有将该失败当作模型失败或丢弃模型记录。

程序级数据→训练→安全权重→ONNX→现有 GUI 链路已实际运行。正式场景识别验收尚未完成。

## 现场使用与打包

本地交付包为 `runs/oven-round2-20261107/HeT-OvenCapture-v1.2.0-experimental-v1-v2.zip`，只含新版 EXE、两个版本的 ONNX/类别文件和使用/哈希说明，不含原图。训练 checkpoint 与训练日志仍在 `yolo11n-12cls-fixed30/`，不会提交 Git。

ZIP SHA256：`9a6be0707b6bca0c5baba76a504e34fb83bdcc16d3b1fa967263a6dc5fe15790`，93,992,525 字节。ZIP CRC 检查通过；从交付文件夹使用相对模型路径实际运行 EXE 自检，PASS / ready=true / 退出码 0。包内 `SHA256SUMS.txt` 提供各文件校验值。

1. 解压到可写目录，运行 `HeT-OvenCapture-v1.2.0-experimental.exe`。
2. 在现有“模型 / 图集”页选择 `models/v2/oven12.onnx`、同目录 `classes.json`，选择“二轮烤箱分类（实验·12类）”，输入 224，点击加载。应显示已加载、12 类、224×224、实验预测/无独立验证。
3. 先选真实 JPG/PNG 文件夹测试，应输出 Top-3 和预测 CSV；没有真值不显示准确率。损坏图单独报错，不伪造预测。
4. 对照 v1 时停止当前批次，选择 `models/v1/oven9.onnx` 及其类别文件和“首轮烤箱分类（实验·9类）”。不能只换类别文件或使用泛用 YOLO 分类模式。
5. 本轮没有打开硬件。后续由现场人员沿既有相机流程明确选择设备；确认预览有效后再识别当前帧。没有当前帧应提示未就绪，不代表摄像头故障。

无 GUI 的实际模型自检：

```powershell
$exe = (Resolve-Path './HeT-OvenCapture-v1.2.0-experimental.exe').Path
# 将模型、类别、图片和输出路径替换为已解压包及真实图片的绝对路径。
$argsText = '--model-selfcheck --mode oven_round2 --model "<oven12.onnx>" --classes "<classes.json>" --image "<real.png>" --out "<check.json>"'
$p = Start-Process -FilePath $exe -ArgumentList $argsText -WindowStyle Hidden -Wait -PassThru
$p.ExitCode
```

实际打包复用了现有隔离构建环境，没有运行 pip。PyInstaller 使用 `--onefile --windowed`、收集 ONNX Runtime、tkinterdnd2、cv2_enumerate_cameras，排除 torch、ultralytics、MNN 和 MNN.tools，输出到新的 `runs/oven-round2-20261107/exe/`。完整命令在同目录 `build-command.ps1`，日志在 `exe-build.log`。其他电脑从零构建可用仓库已有 `tools/oven_desk/build_windows.ps1 -SkipMNN`，该脚本创建自己的 `.build-venv`；本轮没有重新创建或修改现有环境。重新构建的 EXE 哈希可能不同，必须重新运行 healthcheck 和实际模型自检。

## 修改文件

- `tools/oven_training/round2_audit.py`、`round2.py`：只读外部审计、证据分层、分组、v1 全量诊断门槛与十二类探索入口。
- `tools/oven_training/model.py`、`train.py`：复用固定训练配方，将本轮 SDK 设置与训练输出放入新的忽略目录；原 CLI 保留。
- `tools/oven_training/export_desk.py`：十二类外部来源导出、逐图张量指纹、取消 pending 文件的加载授权；v1 转换入口保留。
- `tools/oven_desk/inference_engine.py`、`oven_capture_gui.py`：严格十二类模式、自检与现有页面接入，保留九类。
- `tools/oven_desk/tests/gui_model_smoke.py`、`round2_alignment_check.py`、`test_oven_round2.py`：双模式 GUI、真实全量对齐和契约回归。
- `tools/oven_desk/build_windows.ps1`、`一键打包EXE.bat`、`组员使用说明.txt`：实验版本号和既有构建/使用说明。
- `tests/test_oven_export.py`、`test_oven_round2_audit.py`、`test_oven_round2_pipeline.py`：pending 拒绝、审计、标签分母、路径与完整诊断门槛测试。
- `docs/OVEN_ROUND2_DATA_AUDIT.md`、`OVEN_ROUND2_RESULTS.md`：审计、实际命令结果、使用方法和限制。

共 18 个源代码/测试/文档文件。根 README、共享 configs/contracts/requirements、历史训练及测试清单、摄像头兼容和采集防覆盖模块未修改。大文件、图片、权重、逐图预测、设备标识和个人目录路径仅保留本地忽略目录。

## 下一轮最少补采与确认

- 先逐张确认目录标签，修复独立标注副本中的 stale pumpkin、十张薯条层位冲突和缺失元数据；原文件保持只读。
- 采购/挑选**本轮从未使用的新实物**，给每个实物持久唯一 ID，记录烤箱 ID、日期会话、食材状态/数量。换层位、背景或光照不能换成“新实物”。先分组，再冻结独立验证/最终测试用途。
- 优先补薯条（单根/少量/一堆）、可颂（土豆混淆/手遮挡）、土豆、香肠。每个新实物采同位置的锡纸/黑盘成对图，再单独改变上中下层、摆放与光照；记录真实 foil、层位、视角，避免同时改变全部因素。
- 每类需要多个独立实物与多个独立会话；连续拍几十帧不能替代独立样本。具体正式样本量和冻结协议由团队确认；到达前所有正式泛化指标保持 BLOCKED。
- 如要检验目标尺寸/ROI，补人工食材框或可核实 ROI 标注；有独立验证组后再只改变一个变量训练 320/448 或固定 ROI，训练和推理同步，最终测试不参与选择。
- 补跨烤箱/相机与现场用户机验收，以及来源和对外分享许可。MNN/INT8/RK3506 留待后续，不把本机 ONNX 程序通过当端侧部署完成。
