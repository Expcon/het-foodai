# 真实八类模型 × Gradio Demo 端到端验收记录

> 历史时延勘误（2026-10-21）：旧 JSON 的 `latency_ms_internal_median` 实为最后一次内部耗时；旧 P95 使用 exclusive 分位外推；实际只对首图预热一次。以下表格和原 JSON 保留为历史，不代表修正后重测。新算法与本机重新测量见 [交付说明](MVP_V1_DELIVERY.md)。

- 验收日期：2026-10-21
- 运行环境：Windows · Python 3.12.4 · torch 2.8.0+cpu · CUDA available=False · device=cpu
- 模型：`stage8-20261020-164500-40a26d` · stage 8 · input 224px
- checkpoint SHA256：`41056b3ea90d5fa718a18f48cf5182373d1010c5185cb426ba5c949b1ec1f4c5`
- class_list_sha256：`afdc69e898ef6067b469b9f90eea3dfed84fbb76e7a5910ee1826a62f02326af`

> Softmax 分数不是模型准确率。本记录只验证「真实权重能跑通端到端流程」，不代表烤箱真实场景精度。

## 1. 八类冻结配置核对

| ID | key | 中文名 | 与 classes.json 一致 |
|---|---|---|---|
| 0 | `carrot` | 胡萝卜 | ✅ |
| 1 | `potato` | 土豆 | ✅ |
| 2 | `broccoli` | 西兰花 | ✅ |
| 3 | `pumpkin` | 南瓜 | ✅ |
| 4 | `tomato` | 西红柿 | ✅ |
| 5 | `white_radish` | 白萝卜 | ✅ |
| 6 | `cucumber` | 黄瓜 | ✅ |
| 7 | `bell_pepper` | 甜椒 | ✅ |

导出模型 `run_info.json` 中的 class_list 与 `configs/classes.json` 逐项一致，`class_list_sha256` 校验通过。

## 2. 真实图片 Top-3 与 CPU 时延

每张图预热 1 次后重复 5 次；`推理时延` 为 adapter 内部计时中位数，`墙钟中位` 为含 Python 调用的外部计时中位数。

| 图片 | Top-1 | score | Top-2 | Top-3 | 推理时延(ms) | 墙钟中位(ms) |
|---|---|---|---|---|---:|---:|
| 30564166_143402724126_2.jpg | 黄瓜 (cucumber) | 0.9997 | 西兰花 0.0001 | 甜椒 0.0001 | 53.55 | 64.76 |
| 34599220_233837911102_2.jpg | 土豆 (potato) | 1.0000 | 南瓜 0.0000 | 甜椒 0.0000 | 62.43 | 67.58 |
| 9448607_092644642000_2.jpg | 甜椒 (bell_pepper) | 0.9976 | 西红柿 0.0014 | 白萝卜 0.0004 | 60.5 | 74.86 |
| ScreenShot_2026-10-21_140353_276.png | 白萝卜 (white_radish) | 0.9866 | 西兰花 0.0084 | 土豆 0.0030 | 70.4 | 77.35 |
| ScreenShot_2026-10-21_140428_716.png | 西兰花 (broccoli) | 1.0000 | 土豆 0.0000 | 黄瓜 0.0000 | 71.7 | 76.35 |
| ScreenShot_2026-10-21_140609_497.png | 胡萝卜 (carrot) | 1.0000 | 白萝卜 0.0000 | 土豆 0.0000 | 72.72 | 84.0 |
| ScreenShot_2026-10-21_140640_098.png | 南瓜 (pumpkin) | 1.0000 | 西红柿 0.0000 | 白萝卜 0.0000 | 84.96 | 87.09 |
| 微信图片_20261021140243_327_1.png | 西红柿 (tomato) | 0.8974 | 甜椒 0.0909 | 南瓜 0.0102 | 58.28 | 79.98 |

## 3. 异常路径

| 用例 | 结果 |
|---|---|
| 损坏 JPG（文本改名） | PASS: 无法解码为图片：corrupt.jpg（文件可能已损坏） |
| 未上传图片 | 界面提示「请先上传图片」，Top-3 为空 |
| 缺失 checkpoint | status=not_ready；识别按钮 disabled；PASS: 模型尚未训练／未加载。此处没有模拟识别分数。 |

## 4. 界面截图

| 文件 | 内容 |
|---|---|
| `demo_ready_cucumber.png` | 模型 ready + 真实黄瓜图片 Top-3 + 110.3ms |
| `demo_corrupt.png` | 损坏图片错误提示，无预测 |
| `demo_no_upload.png` | 未上传图片提示 |
| `demo_missing_checkpoint.png` | 模型未就绪，识别按钮 disabled |

## 5. 结论

- 真实 EfficientNetV2-S 权重已成功加载并通过完整前向验证；
- 8 张独立真实图片均产生 Top-3，CPU 单张内部计时约 55–97ms；
- 三类异常路径均无虚假预测；
- **可用于课程项目展示**，但只能表述为「演示可运行」，不得把 softmax 分数或同源测试集 1067/1067 说成真实场景准确率。