# App owner's workspace

实现见 `app/app.py`（Gradio）、`src/foodai/predictor.py`（真实 checkpoint adapter）、
`src/foodai/app_logic.py`（UI 状态与文案，纯函数）。

启动方式、状态说明与验收流程：见 `docs/RUN_APP.md`。
实际执行记录：见 `docs/EVIDENCE.md`。

## 边界

- 没有真实 checkpoint 时，只显示"模型未就绪"，**不显示任何模拟分数**；
- 不在本页面展示准确率；
- 不做 ONNX 导出，不声明 RK3506 兼容性；
- `configs/classes.json` 由数据 owner 冻结，本模块只读不写。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
