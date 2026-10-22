# 最终交付验证记录

集成开始于 2026-10-21；网页截图与收尾于 2026-10-22（Asia/Shanghai）。[verification.json](verification.json)为命令、退出码、统计、哈希和部署状态的索引。原始完整日志在本地 ignored `runs/delivery-20261021/`。

- `baseline-training-record.json`：用户交付 Baseline 的历史训练参数选摘，省略机器路径；没有重新训练。
- `baseline-history.csv`、`baseline-test-historical.json`：原始训练历史及 1067/1067 的同源冻结 test 证据。
- `test-recheck.json`：本机固定同一权重、同一 manifest 的一次 test 复核；不用于调参/模型选择。
- `ready-cpu.json`、`ready-cuda.json`：本机实际 `app/app.py --check`，exit 0。
- `acceptance-cpu-corrected.json`、`acceptance-cuda-corrected.json`：修复后新运行，真实图片，CPU 每图 5 次、CUDA 每图 10 次，各图先预热 1 次；包含原始时延样本。
- `image-sources.json`：八类已知许可 train 功能样本及真实 JPG→PNG 转码的来源/SHA。原图未提交 Git。
- `delivery-real-cpu-top3.png`：本机浏览器上传真实胡萝卜 JPG 后的完整模型 Top-3 与单次功能时延。
- `delivery-no-upload.png`：本机浏览器未上传时的提示，无预测。

正常上传和未上传的浏览器证据来自本次操作。损坏图片、缺失模型、CPU/CUDA JPG/PNG 的实际 HTTP 检查独立列在 JSON 中。组合 HTTP/浏览器等待脚本曾因中断后的浏览器等待超时退出 1；HTTP 断言已完成且保存证据，不能把整个组合脚本说成 exit 0。浏览器之后拒绝 origin 权限，清除/更多页面复测保持 BLOCKED/NOT_REPEATED，未通过其他方式绕过。

截图中的胡萝卜图片是 Dataset v1.1.0 的已知许可样本，来源 `kaggle_vegetable_v1`，许可记录 CC-BY-SA-4.0。原始来源/作者与使用范围沿用数据包 provenance 和 [来源说明](../../DATA_SOURCES.md)；截图仅作课程功能证据，图片部分沿用该许可。它来自训练集，既不是独立测试，也不是新的准确率证据。未使用 235 张 unknown 许可图片作本次功能样本。

Sun 的 `docs/acceptance-evidence/20261021/*.json` 保持原 Git 内容；旧 internal_median 实为最后一次内部耗时，旧 P95 为 exclusive 外推。历史截图与数字不冒充本次修正后重测。正式交付边界和可执行步骤见 [MVP_V1_DELIVERY.md](../../MVP_V1_DELIVERY.md)。
