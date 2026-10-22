# 本地启动：HeT FoodAI 演示

> 本页面只做**单张主食材图片分类**演示。没有真实 checkpoint 时，它只会明确显示"模型未就绪"，不会给出任何分数。

## 0. 先决条件

- Windows + Python 3.12
- 项目内 `.venv`（`AGENTS.md` 禁止装到系统环境 / 覆盖现有环境）
- 两个输入缺一不可：
  1. `configs/classes.json`，`status` 必须是 `frozen`
  2. 模型 owner 训练产出的 `runs/<run_id>/best.pt`（同目录建议保留 `run_info.json`）

## 1. 建环境

```powershell
cd <repo>
py -3.12 -m venv .venv

# 先按官方 wheel selector 安装与本机 CUDA 匹配的 torch + torchvision
# 例如（仅示意，版本以官方 selector 为准）：
# .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
```

## 2. 启动前自检（不打开界面）

```powershell
.\.venv\Scripts\python.exe app\app.py --check
.\.venv\Scripts\python.exe app\app.py --check --checkpoint runs\<run_id>\best.pt
```

- `state` 为 `ready` 才能开始识别；
- `not_ready` = 没有 checkpoint / 没有冻结类别 / 路径写错；
- `contract_error` = checkpoint 的类别顺序、名称或 hash 与冻结配置不一致，必须由模型 owner 重新导出，**不能手动改配置迁就**。

## 3. 启动界面

```powershell
.\.venv\Scripts\python.exe app\app.py --checkpoint runs\<run_id>\best.pt
```

可选参数：

| 参数 | 作用 |
|---|---|
| `--classes configs\classes.json` | 指定冻结类别配置（默认就是这个） |
| `--expect-checkpoint-sha256 <前缀>` | 只接受 SHA256 前缀匹配的 checkpoint |
| `--device cuda` / `--device cpu` | 强制设备（默认自动） |
| `--host` / `--port` | 默认 `127.0.0.1:7860` |

打开 <http://127.0.0.1:7860>。

## 4. 界面状态说明

| 状态 | 触发条件 | 表现 |
|---|---|---|
| 模型未就绪 | 没有 checkpoint / 没有 `classes.json` / 配置为 draft | 顶部横幅显示**原文错误**，识别按钮禁用 |
| 契约不符 | 类别顺序/名称/`class_list_sha256`/hash 前缀不一致 | 横幅显示首个不一致项与修复方向，拒绝加载 |
| 输入无效 | 非 JPG/PNG、0 字节、无法解码 | 上传即报错，不产生任何预测 |
| 正常 | 以上都通过 | 预览 + Top-3（排名/中文名/key/score）+ 时延 |

`score` 是 softmax 类别分数，**不是**校准后的正确概率；页面不展示任何准确率。

## 5. 三条验收流程

1. **正常输入**：上传一张 JPG → 预览出现 → 点"开始识别" → Top-3 与 `latency_ms` 出现
2. **损坏图片**：把 `.txt` 改名成 `.jpg` 后上传 → 显示"输入不是有效图片：无法解码"
3. **缺失 checkpoint**：不传 `--checkpoint` 或指向不存在路径 → 显示"模型尚未训练／未加载"

## 6. 不在本次范围内

- ONNX 导出与端侧性能基准
- RK3506 兼容性结论（没有真实板子与内存数据之前不写）
- 任何准确率数字（只来自模型 owner 的正式评测，不在本页面）
