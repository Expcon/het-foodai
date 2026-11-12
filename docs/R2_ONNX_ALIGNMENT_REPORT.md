# R2 ONNX 转换一致性验收（2026-11-12）

结论：PASS。冻结 R2 checkpoint 经安全加载导出后，PyTorch、CPU ONNX Runtime 和 Desk 的 2,168 条真实图片记录全部通过转换检查。本报告不提供独立食材准确率，不构成生产或端侧部署验收。本轮没有训练、修改权重、重新划分数据或更新原始标签。

## 模型与契约

- 输入 checkpoint：`runs/oven-r2-whole-corn-20261112/R2/training/last.pt`，Epoch 30、1,922 张训练图、AdamW lr=0.0003；来源提交 `bceb927`。
- 使用现有 `load_model`：先校验预期文件 SHA256，`torch.load(weights_only=True)`，核对 metadata、有序类别、预处理及 state_dict，再实际 eval 前向。导出前后模型持久状态指纹一致，checkpoint 和原始 manifest 文件 SHA 未变化。
- EXIF 方向校正、RGB；Pillow bilinear 等比缩放至 224 内，居中补边 114；除以 255，FP32、NCHW。没有额外中心裁剪、BGR 输入或重复 Softmax。
- 静态输入 `[1,3,224,224]`，输出 `[1,12]`，概率输出，opset 17；`onnx.checker.check_model(full_check=True)` 通过。模型大小 6,202,899 bytes。
- R2 专属 JSON 从本次 checkpoint metadata 与实际新 ONNX SHA 生成，全部图片比较通过后才写入正式 `classes.json`。未修改共享八类 `configs/classes.json`。
- 契约 `het-oven-classifier/v2` 表示既有十二类协议版本，不代表模型是历史 v2；checkpoint、ONNX 和 JSON 哈希区分模型身份。

| 文件或契约 | SHA256 |
|---|---|
| R2 checkpoint | `10caa092f31dbf9f57d15509dd8e525d0bba29457d3f70895cc7c8c9af7532a2` |
| R2 `oven12.onnx` | `c3c00944c7765b74d02b4f890caf77b2fe77bbadb87dacdb436f99a541febaa6` |
| R2 `classes.json` | `fe0484729f3ec5619dbdf4372072f42fef1aefae11c7bd207d69d70bf266fc77` |
| 有序十二类 | `eeb5260a718f2a1bd42a4d600c8cb2e5ee290c98a5a24e38aa1822e7391827f7` |
| 原 R2 `r2_rows.json` | `9a580709f353903d5a5337acb7280d1b93437dceddd4d7bd08844bb76c334ff1` |
| 本次显式诊断清单 `diagnostic120.json` | `a8984f243ea2d58c99ebbc8bf21f057c54a17da6331b8a59f5083815fa5b0dce` |

ID 0–11：asparagus、bacon、broccoli、chicken_wing、croissant、french_fries、potato、salmon、toast、**corn**、sausage、steak。corn 仍为 ID 9，R2 范围为整根玉米，没有新增白玉米类别。

## 全量真实文件比较

PyTorch 和 ORT 接收同一个 FP32 数组。独立 Desk 环境从原文件重新解码和预处理，逐图 SHA256 与训练侧张量字节完全一致，再核对概率。两个阶段均要求 Top-1 和有序 Top-3 完全相同，最大概率绝对误差 ≤1e-4、全部值有限且概率合法。

| 覆盖范围 | 记录数 | Desk 张量字节一致 | PT/ORT/Desk Top-1 一致 | 有序 Top-3 一致 | 最大概率绝对误差 |
|---|---:|---:|---:|---:|---:|
| R2 全部训练图 | 1,922 | 1,922 | 1,922 | 1,922 | 5.3271651268e-7 |
| 已暴露的范围内诊断图 | 120 | 120 | 120 | 120 | 3.6656856537e-6 |
| 新整根白玉米全部图片，含排除的严重欠曝图 | 51 | 51 | 51 | 51 | 2.3841857910e-7 |
| 完整绑定清单 + 诊断追加记录 | **2,168** | **2,168** | **2,168** | **2,168** | **3.6656856537e-6** |

不一致、异常及非法概率向量均为 0。2168 条记录含 2167 个唯一文件 SHA；原始完整 R2 清单 2048 条 = 1922 训练 + 126 排除记录（124 旧半根黄玉米、1 重复、1 欠曝）。保留整份 checkpoint 绑定清单，再追加 120 条诊断记录。51 张新白玉米中的 50 张已在训练覆盖范围内，分组计数有交集，不能相加解释为独立样本量。被排除数据仅用于格式一致性检查，没有在本轮用于训练或选模。

原 144 张诊断图片已在历史开发中使用；本次仅加入原范围内 120 张，24 张白玉米半段不在整根玉米范围。没有重标为未使用测试集。

## 真实执行命令与证据

以下从仓库根执行；`$deskPython` 指既有隔离 Windows 构建环境的 Python。本轮没有安装依赖或改全局 Python/CUDA。需要复跑时更换全新输出目录，工具拒绝覆盖已有导出及报告。

```powershell
.venv/Scripts/python.exe -u -m tools.oven_training.export_desk export `
  --checkpoint runs/oven-r2-whole-corn-20261112/R2/training/last.pt `
  --expect-sha256 10caa092f31dbf9f57d15509dd8e525d0bba29457d3f70895cc7c8c9af7532a2 `
  --rows runs/oven-r2-whole-corn-20261112/r2_rows.json `
  --source-root "<source-root>" --r2-source-root "<whole-corn-root>" `
  --diagnostic-rows runs/oven-r2-demo-20261112/diagnostic120.json `
  --diagnostic-root "<diagnostic-root>" --out runs/oven-r2-demo-20261112/onnx

& $deskPython tools/oven_desk/tests/round2_alignment_check.py `
  --model runs/oven-r2-demo-20261112/onnx/oven12.onnx `
  --classes runs/oven-r2-demo-20261112/onnx/classes.json `
  --alignment runs/oven-r2-demo-20261112/onnx/onnx_alignment.json `
  --rows runs/oven-r2-whole-corn-20261112/r2_rows.json `
  --source-root "<source-root>" --r2-source-root "<whole-corn-root>" `
  --diagnostic-rows runs/oven-r2-demo-20261112/diagnostic120.json `
  --diagnostic-root "<diagnostic-root>" --out runs/oven-r2-demo-20261112/desk_alignment.json
```

两条命令退出码均为 0。证据根目录 `runs/oven-r2-demo-20261112/`：`export.log`、`onnx/onnx_alignment.json`、`onnx/pytorch_probabilities.npy`、`desk_alignment.log`、`desk_alignment.json`、`desk_cohort_summary.json`。逐图包含原图 SHA、输入张量 SHA、概率、Top-3 和误差，失败不会生成可加载批准契约。

导出环境：Python 3.12、Torch 2.14.1+cu130、torchvision 0.29.1+cu130、Ultralytics 8.3.230、ONNX 1.19.1、ORT 1.23.2、NumPy 2.5.2、Pillow 12.3.0。Desk 环境：Python 3.12.10、ORT 1.23.2、NumPy 2.2.6、Pillow 12.0.0、OpenCV 4.12.0.88。不同 Pillow/NumPy 环境下仍逐图字节一致。

导出使用原有显式 legacy exporter，产生弃用提示，但 checker 和全量数值验收实际通过。未迁移导出器、融合网络或量化。本轮比较使用 CPU FP32，不能据此报告 CUDA 时延、MNN、INT8 或 RK3506 已部署。

## 保护检查与限制

`preservation_after.json` 复核 606 个历史受保护文件没有变化；原数据目录 2925 文件、新白玉米 PNG/JSON 102 文件、诊断目录 144 文件逐文件 SHA 和文件集合均未变化。旧八类 manifest 两份仍为 `6bef9776c278d8efac86008a11c47386cb7d95b7164073b9bd07e84cb5cf2821`，共享八类 JSON、v1/v2/A/B/R0/R1/R2 历史权重及交付文件保持原样。

这里的“一致”是转换正确性，未使用人工标签计算准确率。训练图和已暴露诊断图不能证明独立泛化；整根玉米的现场可靠性、半根玉米行为及其他场景仍需真实独立实物验证。没有用转换结果选择或替换默认模型。
