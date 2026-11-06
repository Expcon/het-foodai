# 真实烤箱首轮探索训练与现场文件验证

日期：2026-11-06。已经实际完成 ZIP 审计、旧模型诊断、YOLO11n-cls 训练、CPU/CUDA 批量推理和文件输入异常验证。**本轮没有独立验证集或测试集，不能宣称正式准确率。没有连接相机或控制烤箱。**

## 数据与分组结论

`food_classification_dataset.zip` 共569张1920×1080 PNG，9个来源标签。原划分454/58/57，只有部分类别恰为48/6/6。逐类统计见 [数据审计](OVEN_ROUND1_DATA_AUDIT.md)。来源ZIP SHA256为 `5a7d0ef8a72d2d85d19ed5b54e2aa0633370a28455c5c8cf9ecd2a2cf7be7d16`。

文件序号没有实物身份依据；压缩包无实物ID、session、EXIF或其他元数据。1个完全重复副本排除后有568张唯一像素图。pHash半径6产生576对跨原划分相似候选，其中252对同类；相同暗背景也带来跨类误匹配，不能将候选对当作已证明同一实物。抽看45张联系表发现共同的暗色炉腔背景、同类多角度及摆放变化，但不能据此确认实物数量。

本轮将“同类但无法证明独立”的图片保守合组，再并入重复/相似连接，得到1个**不确定性组**，不代表只有1件真实食材。原新ZIP的train/val/test目录划分停止用于评估，568张全部进入探索训练；没有从这些关联图片中另造验证集。旧MVP冻结test及配置完全不动。此批新数据以后也不能再当本模型的独立test。

9个实验类别按以下顺序：asparagus、bacon、broccoli、chicken_wing、croissant、french_fries、potato、salmon、toast。它们仅写入新实验metadata，不修改公共 `configs/classes.json`，不把9类模型直接接入旧8类Demo。旧新数据联合重训需要后续确认类别策略，本轮实验不能替代MVP。来源标签未逐图由团队确认、许可unknown，原图不得再分发。

## 真实训练与指标

初始化使用已有、精确SHA核验的官方 `yolo11n-cls.pt`（ImageNet预训练）；固定Ultralytics8.3.230。使用其真实分类网络、替换为9维输出，在PyTorch中运行显式train-only循环，避免SDK自动把test当val、执行early stopping或按val挑选best。官方架构与任务来源：[固定版本实现](https://github.com/ultralytics/ultralytics/blob/v8.3.230/ultralytics/nn/tasks.py)、[分类任务](https://docs.ultralytics.com/tasks/classify/)。

训练前写出固定配置：30 epochs、batch32、seed42、AdamW、lr0.001、weight_decay0.0005、workers0、FP32（AMP关闭）、确定性算法。全图按比例缩放并居中补114到224×224，EXIF方向/RGB、bilinear、NCHW、除255；训练再加水平翻转p0.5与亮度/对比度/饱和度0.1抖动。推理复用相同基础预处理。它不同于官方默认中心裁剪，必须使用本轮脚本复现。

执行环境为Windows、Python3.12、PyTorch2.14.1+cu130、torchvision0.29.1+cu130、RTX5080Laptop。使用当前worktree新建 `.venv`，通过本地 `.pth` **只读引用**既有项目CUDA依赖，缺少的16项SDK依赖仅安装在新环境，`pip check` exit0。没有更改Windows全局包或旧虚拟环境。SDK联网/自动pip受既有offline guard阻断，settings保存在runs且sync=false。版本清单在本地 `environment-freeze.txt`；新增依赖锁在 `tools/oven_training/requirements-sdk.txt`。

| 实际记录 | 结果 / 范围 |
|---|---|
| 训练命令 | exit0，30轮全部完成；每轮568张，无val/test输入 |
| 最后一轮增强训练 | loss=0.0056926587，Top1=567/568；这是该轮有增强的训练统计 |
| 优化循环时间 | 35.5845秒；不含此前图片读取缓存、模型加载和环境准备 |
| checkpoint | 只保留最终轮 `last.pt`；1,542,633参数，6,270,865字节 |
| 无增强CPU批量推理 | 569/569来源标签命中，其中去重训练图568/568；exit0 |
| 无增强CUDA批量推理 | 同上，exit0；每张保存Top3与分数 |
| CPU/CUDA对齐 | 569张Top1全部一致，最大Top1分数差0.00450051；没有声称逐元素等价 |
| 独立验证/测试准确率 | **N/A，BLOCKED：无可证独立实物** |
| 现场CLI软件路径 | 真实PNG及由该图转码JPG通过；缺图、坏图、无图片参数、batch0、缺模型、坏模型均正确返回exit2 |
| 全部单元测试 | `python -m unittest discover -s tests -v`：115项，114 PASS、1 SKIP、0 FAIL，20.121s，exit0；SKIP为Windows符号链接权限 |

568/568是对已见训练图的再代入检查，**不是验证集100%，不是现场泛化准确率**。同一物体、背景或拍摄过程的重复性可能使拟合很容易。没有根据上述结果调整轮数/学习率、选中间epoch或重新训练。新YOLO的已见图错误清单为空；旧Baseline在支持的两类上有5张potato→cucumber误判，且460张新类别不在旧词表内，其中88张被强制预测到旧类且分数≥0.9。详见 [旧模型真实诊断](OVEN_BASELINE_DIAGNOSTIC.md)，这些不是自动改标签或设拒识阈值的依据。

CPU/GPU批量作业有运行时间重叠，端到端记录主要包含PNG解码，**不用于比较设备速度**。每批计时含解码、方向/RGB、缩放补边、tensor、传输、forward、Top3及取回；排除预先校验/哈希、模型加载和JSON写盘。CUDA前后同步。36批（35批16张、最后9张）的原始时间留在predictions.json。旧Baseline的独立顺序诊断在YOLO训练前完成，计时范围不同，不直接横比。

## 本机立即使用

在本分支worktree根目录执行，现有 `.venv` 已可用。下列代码只读取文件，不连接相机。将 `$images` 换成你明确选择的新现场图片路径；必须每次使用新的输出目录。

```powershell
$py = '.\.venv\Scripts\python.exe'
$checkpoint = 'runs/oven-first-round-20261106/yolo11n-fixed30/last.pt'
$sha = '72367305b65b31920c59865758dcdd4a1ee506751bde57db113f017c5c314b25'
$images = @('<第一张真实现场图片.png>', '<第二张真实现场图片.jpg>')
& $py -m tools.oven_training.predict --checkpoint $checkpoint --expect-sha256 $sha --images $images --out runs/oven-field-check-NEW --device cpu --batch-size 1
# 已验证CUDA可用时，将device改为cuda:0。需要批量时显式设置batch-size。
```

成功返回 `EXPLORATORY_PREDICTIONS_READY`，输出 `predictions.json`；每图Top3使用上述实验类别，score未经校准。`--images`没有真值标签，因此不产生所谓现场准确率。未支持的食材仍会被迫落入九类之一；当前没有空炉、陌生类或多食材拒识，不可用于自动烹饪控制。权重加载使用 `weights_only=True`、完整SHA、metadata校验、严格state_dict和一次真实前向检查；失败返回 `ERROR_NOT_READY`/exit2，不生成预测文件。

现场应另记物理样本ID、session、位置和照明，逐张人工核对Top3。新收集的独立实物应先整体保留为待审数据；在形成批准的独立验证/测试清单前，不用于调参，也不要再次随机按帧切分。需要补采时按 [采集协议](PHASE2_OVEN_CAPTURE_PROTOCOL.md)操作，本脚本不操作硬件。

## 重现本轮操作

`$sourceZip` 指向实际收到的压缩包，`$baseline` 指向旧交付真实best.pt。图片、权重和环境均未入Git；换电脑必须单独恢复权重并核对完整SHA。当前 `.venv`依赖本机已有CUDA环境的只读引用，不能把虚拟环境目录直接搬到另一台机器；另一台机器应建立自己的项目环境、准备匹配的Torch/torchvision，再安装本工具锁定SDK依赖。

```powershell
& $py -m tools.oven_training.audit --zip $sourceZip --output runs/oven-first-round-20261106
& $py -m tools.oven_training.diagnose_baseline --rows runs/oven-first-round-20261106/candidate_rows.json --checkpoint $baseline --out runs/oven-baseline-NEW --device cuda
& $py -m tools.oven_training.train --rows runs/oven-first-round-20261106/candidate_rows.json --pretrained runs/phase2-edge/export-smoke/yolo11n-cls.pt --out runs/oven-yolo-NEW --device cuda:0
# 当前已训练权重的批量复查（输出路径不得存在）：
& $py -m tools.oven_training.predict --checkpoint $checkpoint --expect-sha256 $sha --rows runs/oven-first-round-20261106/candidate_rows.json --out runs/oven-batch-NEW --device cuda:0 --batch-size 16
& $py tests/smoke_oven_round1.py --checkpoint $checkpoint --expect-sha256 $sha --image '<一张实际PNG>' --out runs/oven-smoke-NEW
```

训练脚本拒绝已有run目录；audit重复执行只复核已提取文件，不覆盖原图，但会重写指定的本地审计JSON，应使用原ZIP和原参数或新的output保存新版本。此处新实验类别不能传给旧 `validate_manifest.py` 冒充公共冻结类别；将来接入Demo需要三人共同确认新契约。

## 产物和保全证据

本地 `runs/oven-first-round-20261106/` 下：`data_audit.json`、`candidate_rows.json`；旧模型 `baseline-diagnostic-20261106/{summary.json,predictions.jsonl,overlap_mismatches.json}`；新模型 `yolo11n-fixed30/{last.pt,run_info.json,fixed_config.json,history.csv}`；`yolo-gpu-batch/` 和 `yolo-cpu-batch/` 的 `predictions.json,failures.json`；`field-cli-smoke/verification.json`及每个错误路径日志。审计联系表和原图均留本地忽略目录。

新权重SHA256：`72367305b65b31920c59865758dcdd4a1ee506751bde57db113f017c5c314b25`。新实验class-list SHA256：`42bd80f04446cd87f8c719c78034bf274bd447fe7a60205713f118cd0b584b52`。

旧manifest SHA256仍为 `6bef9776c278d8efac86008a11c47386cb7d95b7164073b9bd07e84cb5cf2821`；旧classes文件仍为 `27d5337826c26a3b743ec0928ea8da050f4ae7bf0ef31e7efb051e4e907e9614`；旧best.pt仍为 `41056b3ea90d5fa718a18f48cf5182373d1010c5185cb426ba5c949b1ec1f4c5`。全部9个保全对象前后相同。

脱敏实际证据：[verification.json](phase2-evidence/20261106/verification.json)、[单元日志](phase2-evidence/20261106/unit-tests.log)、[训练日志](phase2-evidence/20261106/training.log)、[每轮记录](phase2-evidence/20261106/training-history.csv)。本轮没有做MNN/INT8再导出、板端性能或真实相机验收；新模型应先完成独立实物评估，再考虑正式冻结及部署。
