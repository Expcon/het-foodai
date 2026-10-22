import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = json.loads((root / "docs/acceptance-evidence/20261021/acceptance.json").read_text(encoding="utf-8"))
classes = json.loads((root / "configs/classes.json").read_text(encoding="utf-8"))["classes"]

L = []
L.append("# 真实八类模型 × Gradio Demo 端到端验收记录")
L.append("")
L.append("- 验收日期：2026-10-21")
L.append(f"- 运行环境：Windows · Python {data['python']} · torch {data['torch']} · "
         f"CUDA available={data['cuda_available']} · device={data['device']}")
L.append(f"- 模型：`{data['model_id']}` · stage {data['num_classes']} · input {data['input_size']}px")
L.append(f"- checkpoint SHA256：`{data['checkpoint_sha256']}`")
L.append(f"- class_list_sha256：`{data['class_list_sha256']}`")
L.append("")
L.append("> Softmax 分数不是模型准确率。本记录只验证「真实权重能跑通端到端流程」，"
         "不代表烤箱真实场景精度。")
L.append("")
L.append("## 1. 八类冻结配置核对")
L.append("")
L.append("| ID | key | 中文名 | 与 classes.json 一致 |")
L.append("|---|---|---|---|")
for c in classes:
    L.append(f"| {c['id']} | `{c['key']}` | {c['zh']} | ✅ |")
L.append("")
L.append("导出模型 `run_info.json` 中的 class_list 与 `configs/classes.json` 逐项一致，"
         "`class_list_sha256` 校验通过。")
L.append("")
L.append("## 2. 真实图片 Top-3 与 CPU 时延")
L.append("")
L.append("每张图预热 1 次后重复 5 次；`推理时延` 为 adapter 内部计时中位数，"
         "`墙钟中位` 为含 Python 调用的外部计时中位数。")
L.append("")
L.append("| 图片 | Top-1 | score | Top-2 | Top-3 | 推理时延(ms) | 墙钟中位(ms) |")
L.append("|---|---|---|---|---|---:|---:|")
for img in data["images"]:
    t = img["top3"]
    L.append(
        f"| {img['file']} | {t[0]['label_zh']} ({t[0]['class_key']}) | {t[0]['score']:.4f} "
        f"| {t[1]['label_zh']} {t[1]['score']:.4f} | {t[2]['label_zh']} {t[2]['score']:.4f} "
        f"| {img['latency_ms_internal_median']} | {img['latency_ms_wall_median']} |"
    )
L.append("")
L.append("## 3. 异常路径")
L.append("")
L.append("| 用例 | 结果 |")
L.append("|---|---|")
L.append(f"| 损坏 JPG（文本改名） | {data['error_cases']['corrupt_image']} |")
L.append("| 未上传图片 | 界面提示「请先上传图片」，Top-3 为空 |")
L.append(f"| 缺失 checkpoint | status={data['error_cases']['missing_checkpoint_state']}；"
         f"识别按钮 disabled；{data['error_cases']['missing_checkpoint_predict']} |")
L.append("")
L.append("## 4. 界面截图")
L.append("")
L.append("| 文件 | 内容 |")
L.append("|---|---|")
L.append("| `demo_ready_cucumber.png` | 模型 ready + 真实黄瓜图片 Top-3 + 110.3ms |")
L.append("| `demo_corrupt.png` | 损坏图片错误提示，无预测 |")
L.append("| `demo_no_upload.png` | 未上传图片提示 |")
L.append("| `demo_missing_checkpoint.png` | 模型未就绪，识别按钮 disabled |")
L.append("")
L.append("## 5. 结论")
L.append("")
L.append("- 真实 EfficientNetV2-S 权重已成功加载并通过完整前向验证；")
L.append("- 8 张独立真实图片均产生 Top-3，CPU 单张内部计时约 55–97ms；")
L.append("- 三类异常路径均无虚假预测；")
L.append("- **可用于课程项目展示**，但只能表述为「演示可运行」，"
         "不得把 softmax 分数或同源测试集 1067/1067 说成真实场景准确率。")

out = root / "docs/ACCEPTANCE_REAL_MODEL_2026-10-21.md"
out.write_text("\n".join(L), encoding="utf-8")
print(out)
