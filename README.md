# HeT FoodAI / 和而泰食材识别

首版课程展示交付候选：**Dataset v1.1.0 + 真实 EfficientNetV2-S 八类模型 + 本地 Gradio**。模型权重和真实图片单独交付，不在 Git 仓库内；未恢复权重时页面会明确显示“模型未就绪”。

模型负责人陈谦负责训练与评测；杨熙珂负责数据、QA 与版本追溯；孙璞月负责预测适配器和 Gradio。当前正式类别为 `configs/classes.json`（frozen / v1.1.0）：

| ID | key | 中文 |
|---:|---|---|
| 0 | carrot | 胡萝卜 |
| 1 | potato | 土豆 |
| 2 | broccoli | 西兰花 |
| 3 | pumpkin | 南瓜 |
| 4 | tomato | 西红柿 |
| 5 | white_radish | 白萝卜 |
| 6 | cucumber | 黄瓜 |
| 7 | bell_pepper | 甜椒 |

Dataset v1.1.0 共 10,662 张，train/val/test = 8,527/1,068/1,067。Baseline 在该同源冻结 test 上为 **1067/1067**；这是特定数据版本和固定权重的结果，不能外推为烤箱、混合食材或独立场景准确率。Softmax 分数也不是准确率。

Windows PowerShell 快速开始（Python 3.12、Git；在仓库根目录）：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt gradio==6.30.0 pillow==12.3.0 numpy==2.5.2
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

按 [MVP v1.0 交付说明](docs/MVP_V1_DELIVERY.md)恢复单独提供的 `foodai-mvp8-baseline-v1.1.0.tar.gz`，核对权重 SHA，然后运行：

```powershell
$checkpoint = 'runs/food-v1.1.0-baseline/stage8-20261020-164500-40a26d/best.pt'
.\.venv\Scripts\python.exe app/app.py --check --checkpoint $checkpoint --classes configs/classes.json --device cpu --expect-checkpoint-sha256 41056b3ea90d5fa718a18f48cf5182373d1010c5185cb426ba5c949b1ec1f4c5
.\.venv\Scripts\python.exe app/app.py --checkpoint $checkpoint --classes configs/classes.json --device cpu
```

打开终端给出的本地地址，上传 JPG/PNG，查看 Top-3 与标明范围的时延。缺图、损坏图片、缺模型不会产生虚假预测。

- [完整交付、从零部署、压缩包恢复、来源限制与验收证据](docs/MVP_V1_DELIVERY.md)
- [Dataset v1.1.0 数据报告](DATASET_V1.1.0_REPORT.md)、[下载与哈希](docs/PROCESSED_DATA_DOWNLOADS.md)
- [数据契约](docs/DATA_CONTRACT.md)、[推理契约](docs/INFERENCE_CONTRACT.md)、[训练说明](docs/TRAINING.md)

235 张来源许可尚未核实；黄瓜 `0225.jpg` 仍待语义复核。不要把图片/权重/虚拟环境/密钥提交 Git，也不要将未知许可图片作为可自由再分发素材。旧版数据与 draft 类别仅供历史追溯，正式运行统一使用 v1.1.0 清单和 `configs/classes.json`。

独立跨场景正式评估、8→12→16 增量学习、ONNX 与真实端侧部署尚未完成。
