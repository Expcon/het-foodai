# HeT FoodAI 第一次三人协作集成报告

日期：2026-10-20。程序级闭环 **PASS**；真实八类训练 **BLOCKED**。本报告没有真实食材准确率或部署性能结论。

## 集成环境与历史

- 原工作区 `D:\Projects\HeT_FoodAI` 保持 `main`，没有合并或提交到 main。
- 独立工作区 `C:\Users\31\.codex\worktrees\integration-mvp-v1\HeT_FoodAI`，分支 `integration/mvp-v1` 从本次 fetch 的最新 `origin/main`（`f229bc2`）创建。
- 使用原项目 `.venv`；最初缺少 Gradio，按现有 `requirements-demo.txt` 安装项目依赖，没有系统级安装或替换 PyTorch。
- 依次保留历史合入：训练 `9c9b60a`（合并 `9ae65f4`）、数据 `f9481c2`（合并 `1f76e39`）、Demo `22df321`（合并 `c4443aa`）。**零冲突**，未重写成员历史。

## 审计发现与必要修复

| 发现 | 集成处理 |
| --- | --- |
| Demo 两处 `torch.load(weights_only=False)`，独立解释 sidecar 与权重 | 删除重复加载/校验，复用训练模块 `load_classifier_checkpoint`；只以内嵌 metadata 为准 |
| 仅凭 run_info 就显示 ready，空/伪造模型可通过旧测试 | 必须严格装载全部 state_dict、移动到指定设备，并完成有限输出的完整模型前向后才发布 ready；失败不保留部分模型 |
| CUDA 计时无显式同步，文字与实际范围不符 | 起止同步；包含预处理、H2D、前向、softmax、Top-3、D2H；排除图片解码/校验、首次加载验证、上传和渲染 |
| Gradio Image 在业务校验前解码损坏输入，还可能引入额外图片处理 | 改为原文件 File 上传和独立预览；保持训练与推理共同的 RGB 解码及 get_transforms；换图/清除均清空旧结果 |
| 旧测试将 3 类/空参数 payload 当作 ready | 改为拒绝不受支持阶段/无效权重；使用真实完整 EfficientNetV2-S 临时合成 checkpoint 证明 ready 和 8 类配 16 类前缀 |
| 数据中有 38 个 Pillow 识别为 MPO 的 JPEG 容器 | 接受 JPEG/MPO 首帧，与训练读取一致；伪装 PNG 的 GIF 被拒绝；有独立回归测试 |
| 极短损坏 checkpoint 会抛出 IndexError、KeyError 或 struct.error，导致状态检查崩溃 | 仅在 weights-only 解析边界统一转换为 ContractError，不添加不安全回退；六个实际损坏 payload 回归通过 |
| 推理中换图/清除可能让旧结果写回新图片页面 | 上传和预测共用串行队列，整个预测链禁用上传与按钮，完成/失败均恢复；浏览器用真实 predictor 暂停屏障验证锁定与恢复 |

实际修复文件：`app/app.py`、`src/foodai/predictor.py`、`src/foodai/app_logic.py`、`src/foodai/checkpoint.py`、`tests/test_predictor.py`、`tests/test_integration.py`、`tests/smoke_training.py`、`tests/smoke_integration.py`。其余新增为本报告、简短计划、可机读证据与四张仅含合成输入的截图。整分支复核发现的两项边界问题均已修复，复核没有剩余阻塞代码问题。

没有改公共 contracts、README、requirements、类别配置、版本化数据或数据工程代码。推理仍返回原契约六个字段。只保留既有 8/12/16 前缀，不加入增量算法、自动冻结、续训或部署功能。

## 真实交付数据核验

先检查原项目、缓存和默认 Downloads，未发现所需包。只下载 selected ZIP（282,979,423 字节）和逐文件 inventory（3,911,953 字节）；未下载旧候选包。

- ZIP SHA256：`7b43bad0351353dfb336ffc1290d31d79eb715006671decfc04053db1629e964`，与 GitHub Release digest 和交付说明一致。
- inventory SHA256：`752090afe84cc3ab0acfb15edfec01eef7ecebf38a2304cc381e6081bfa8053e`。
- 所有 4,827 ZIP 成员通过 CRC 和逐文件 SHA；解压前拒绝路径逃逸、重复成员及不同本地内容覆盖。既有版本化元数据按字节比对后保持原样。
- 4,816 张图片均存在、文件 SHA 正确、可解码、短边至少 96；4,778 JPEG 和 38 MPO/JPEG 容器。
- train/val/test = **3,852/482/482**，external_test = **0**；ID、路径、image SHA 唯一，0–7 的 ID/key 顺序匹配类别提案。
- 按全局 group 和 SHA 检查未发现跨 split；发布的 pHash 检出关系 1,537 对，均同 split。这里重检了发布的 pHash 关系，没有重新计算全部感知哈希或证明不存在未知近重复。
- 全局 group 3,572，而 summary 的 source+group 计数 3,573：一个南瓜组跨两个来源、但均在 train，统计口径不同，没有改数据。
- 当前 manifest 无空 SHA、无 unknown license；许可记录为 CC-BY-SA-4.0 4,199 张、CC-BY-4.0 617 张。核验的是交付许可记录，没有独立逐图判定权属。

两份 manifest SHA 均为 `28920a5704066f747a27abd508658c9ef2e7b5d0f87db597a866945d6f0dee12`。类别提案 SHA 为 `6f0b26e10d493974a6e01d609e384e02304ad82e263faf5687875c808c38a96f`；summary SHA 为 `b26f60f5b188713f3bc8ead629b5cd2e115907cc29074fea96f4ab752934c896`。全部与数据分支 Git blob、Release 内容及集成后文件字节一致。

缓存保留在原项目 `data/cache/releases/data-processed-v1-20261019/` 供复用；解压图片只在隔离工作区，未加入 Git。

## 实际测试命令与结果

以下在隔离工作区运行，`$py = 'D:\Projects\HeT_FoodAI\.venv\Scripts\python.exe'`。

| 实际命令/检查 | 结果 |
| --- | --- |
| `& $py -m unittest discover -s tests -v`，origin/main 基线 | exit 0，8 项通过 |
| 同一命令，三分支原样合入后 | exit 0，52 项通过；说明原测试未捕捉错误 ready |
| `& $py -m unittest discover -s tests -p test_integration.py -v`，修复前 | exit 1，5 项按预期失败；sidecar 误报 ready、缺失参数被接受等 |
| 同一回归命令，修复后 | exit 0，5 项通过，含实际 CUDA 同步调用与完整模型 Top-3 |
| `& $py -m unittest discover -s tests -v`，当前代码 | exit 0，`Ran 60 tests in 17.183s` / `OK`，无跳过；包含损坏 checkpoint、JPEG/MPO、Windows spawn、确定性增强、原子保存和映射拒绝 |
| `& $py scripts\validate_manifest.py --manifest data\manifest.csv --classes data\processed\selected_dataset_v1\classes.proposed.json --check-files` | exit 0，4,816 行和全部实际文件 SHA 通过；明确 DRAFT warning |
| `& $py tests\smoke_integration.py --browser-check` | exit 0，完整 EfficientNetV2-S CUDA AMP 合成训练 2 epoch、17 train/8 val、singleton 尾批、确实更新参数、保存/安全加载、矩阵、test/external CLI、同一 checkpoint 的 adapter 和真实 Gradio 路由通过 |
| `& $py scripts\train_classifier.py --classes data\processed\selected_dataset_v1\classes.proposed.json --no-pretrained` | 实际 exit 2，`BLOCKED: Class list is DRAFT`；未启动真实训练 |
| Gradio 原始上传路径的首次 HTTP 冒烟 | exit 1，损坏 PNG 在框架 Image 预处理抛错；修复后重跑通过，保留 red 日志 |
| Codex 浏览器，127.0.0.1:7861 | 实际选择合成 PNG、预览、点击识别、Top-3/时延通过；清除后结果清空；损坏 PNG 上传和点击识别均明确报错且没有分数/时延 |
| Codex 浏览器，127.0.0.1:7862 | 未配置模型，not-ready 原文可见，识别按钮 disabled，没有分数 |
| Codex 浏览器，真实 predictor 的测试屏障 | 运行中按钮 disabled、上传清除入口消失；放行后真实 Top-3 返回、两控件恢复；清除图片后旧分数与时延清空 |
| 数据元数据逐字节比 Git blob；公共协议/共享草稿按 Git 规范化文本比较 | 全部通过；正式 `configs/classes.json` 仍不存在；新工作区普通文本为 CRLF、原工作区为 LF，未修改映射 |

完整日志保留在本地 ignored 的 `runs/integration-20261020/`。浏览器测试完成后关闭了本次两个本地服务，合成训练目录、临时 frozen 配置和 checkpoint 自动清理；截图只包含随机合成输入。GPU 时延是单次功能验证，不是延迟基准或食材准确率。

## 程序闭环与真实训练条件

程序级闭环 **已完成**：合法 manifest → 合成 GPU 训练 → best.pt → 统一安全加载 → 完整模型 RGB Top-3 → Gradio 正常与失败路径。没有用真实 test 集选模型或调参；真实照片只做文件完整性和解码检查。

真实八类训练 **BLOCKED**：

1. 共享草稿第 7 类为 `7/shrimp/对虾`，数据提案为 `7/shrimp/虾`。ID/key 未变，中文语义待三人确认；class_list_hash 包含 zh，两者哈希不同，不能互换或静默修正。
2. 数据负责人须复核来源只标 fish 的 `d102a111ea2e8686018640e7` 是否确为鲈鱼，目前在 train；也需确认生熟、包装和切分范围。若要修正 v1，应发布新版本和新哈希，不能覆盖本次已核验字节。
3. 三人共同确认后才创建正式 frozen 配置、锁定真实实验版本并由模型负责人开展预训练八类训练。当前没有真实训练模型或精度证据。

仍未验证：真实数据迁移效果、泛化/独立烤箱 external_test、真实 224px 训练显存与训练时长、灾难性遗忘、端侧部署、每张来源照片的物种/权属真伪。稀少类 test 样本很少且类别明显不均衡，后续需结合每类指标解读。

## 原负责人事项与 PR 建议

| PR | 建议 | 理由/负责人事项 |
| --- | --- | --- |
| #1 training | 建议合并代码，仍不启动真实训练 | 保持 train/val 选择、数据哈希、前缀映射与安全 checkpoint；完整 GPU 合成与集成通过。模型负责人等冻结后训练并交付真实参考预测 |
| #2 data-inventory | 建议作为 draft 数据交付合并 | 实际包、全部图片和 metadata 验证通过；合并不等于冻结。杨熙珂处理 fish/鲈鱼疑点及语义范围。未来新来源未匹配输入会保持 unknown，发布前应隔离未知来源/权利记录 |
| #3 app-shell | 原分支不建议直接合并；建议合入本集成修复后的版本 | 原版本有 unrestricted pickle、虚假 ready 和网页损坏输入问题。本集成版本已修复并实际验证；孙璞月后续沿用共享 loader、时延范围与错误路径 |

可通过集成 PR 一次评审三分支及修复；未自动合并任何 PR 或 main。审核程序闭环与冻结真实数据是两个独立决策。

## 证据

- [可机读验证记录](integration-evidence/20261020/verification.json)
- [合成输入 Top-3 网页](integration-evidence/20261020/foodai-integration-synthetic-top3.png)
- [损坏输入，无预测](integration-evidence/20261020/foodai-integration-corrupt.png)
- [缺失模型，按钮禁用](integration-evidence/20261020/foodai-integration-missing-model.png)
- [预测期间锁定控件](integration-evidence/20261020/foodai-integration-busy.png)

## 三条来源分支的实际修改文件

### feature/training

- `docs/TRAINING.md`
- `scripts/evaluate_classifier.py`
- `scripts/train_classifier.py`
- `src/foodai/checkpoint.py`
- `src/foodai/vision.py`
- `tests/smoke_training.py`
- `tests/test_training.py`

### feat/data-inventory

- `.gitattributes`
- `.gitignore`
- `data/manifest.csv`
- `data/processed/selected_dataset_v1/classes.proposed.json`
- `data/processed/selected_dataset_v1/manifest.csv`
- `data/processed/selected_dataset_v1/summary.json`
- `docs/CLASS_DEFINITIONS_PROPOSAL.md`
- `docs/DATASET_HANDOFF.md`
- `docs/DATASET_V1_REPORT.md`
- `docs/DATA_AUDIT_2026-10-19.md`
- `docs/DATA_AUDIT_LOCAL.md`
- `docs/DATA_DOWNLOAD_GUIDE.md`
- `docs/DATA_NEXT_STEPS.md`
- `docs/DATA_PIPELINE.md`
- `docs/DATA_SOURCES.md`
- `docs/PROCESSED_DATA_DOWNLOADS.md`
- `docs/YANG_XIKE_WORK_SUMMARY_AND_HANDOFF.md`
- `scripts/audit_images.py`
- `scripts/package_processed_data.py`
- `scripts/prepare_selected_dataset.py`
- `scripts/propose_groups.py`
- `scripts/render_audit_report.py`
- `tests/test_data_audit.py`
- `tests/test_selected_dataset.py`

### feat/app-shell

- `app/README.md`
- `app/app.py`
- `docs/EVIDENCE.md`
- `docs/RUN_APP.md`
- `src/foodai/app_logic.py`
- `src/foodai/predictor.py`
- `tests/test_app_logic.py`
- `tests/test_predictor.py`
