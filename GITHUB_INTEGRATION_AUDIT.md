# GitHub 全项目历史整合与同步审计

日期：2026-11-12。仓库：[Expcon/het-foodai](https://github.com/Expcon/het-foodai)，公开仓库。本文区分本地提交、公开脱敏对应提交、实际远端引用和已合并状态，不把“已提交”当作“已推送”。

## 1. 核对范围与初始状态

实际执行 `git status`、`git worktree list --porcelain`、`git branch -avv`、`git fetch origin`、全可达历史对象扫描及 GitHub PR 元数据查询。fetch 退出 0。GitHub 连接器的 PR 详情接口返回内部错误，改用已安装的 `gh pr list --state all --json ...` 获取当前状态；没有读取或输出认证 Token。

初始 4 个注册 worktree 均无未提交/暂存/未跟踪修改，没有 stash。以下使用逻辑名称，个人磁盘路径不公开：

| 工作树 | 初始分支 | 初始 HEAD | 初始远端状态 |
| --- | --- | --- | --- |
| 主工作树 | main | f229bc2 | 与 origin/main 相同 |
| 首次集成工作树 | integration/mvp-v1 | 667cdfe | 与同名远端相同 |
| MVP 交付工作树 | integration/mvp-v1.0-delivery | f77ced3 | 与同名远端相同 |
| Phase 2 工作树 | feat/app-oven-presentation-v1.3 | 5bc0d45 | 原提交仅本地 |

初始所有 refs 共 35 个可达提交，其中 14 个提交不在任何 origin 分支上。未安装依赖、修改模型或训练数据，没有接触其他项目/服务器；本轮只有 Git、源码级软件测试、脱敏和报告操作。

## 2. 原始分支与继承关系

```text
main f229bc2
  └─ integration/mvp-v1 667cdfe（合入训练/数据/Demo 原历史）
       └─ app-real-model-demo 8d6fda0
            └─ MVP delivery f77ced3（另 merge 数据 v1.1.0 524879a）
                 └─ Phase2 foundation 7054b16
                      └─ YOLO v1 20e71f3
                           └─ YOLO v2 5c02296
                                └─ v2数据审计 bda2636
                                     └─ train/eval模式审计 e6d1cb9
                                          └─ 固定eval/BN监控 4e8f060
                                               └─ BN A/B 2bbb1d5
                                                    └─ A不稳定诊断 b7164b5
                                                         └─ R0精确重放 8be5060
                                                              └─ R1学习率 d65b00f
                                                                   └─ 补充诊断 84cfac4
                                                                        └─ 身份陈述 793e8ec
                                                                             └─ R1演示 b3215ae
                                                                                  └─ R2训练 bceb927
                                                                                       └─ R2/v1.2.2交付 c7d7e62
                                                                                            ├─ R2 BN诊断 6e2f067
                                                                                            └─ v1.3.0展示 5bc0d45
```

重点：`6e2f067` 与 `5bc0d45` 是兄弟分支，共同祖先为 `c7d7e62`；二者原先并不互相包含。以上实验均为历史成果，本轮没有重新训练或导出模型。

原始本地研究分支按阶段保留：`feat/data-oven-v2-error-audit`、`feat/model-oven-v2-mode-audit`、`feat/model-oven-training-monitor`、`feat/model-oven-bn-real-ab`、`feat/model-oven-a-instability-audit`、`feat/model-oven-r0-observable-replay`、`feat/model-oven-r1-lr-control`、`feat/model-oven-independent-validation-audit`、`feat/model-oven-r1-experimental-demo`、`feat/model-oven-r2-whole-corn`、`feat/model-oven-r2-experimental-demo`、`feat/model-oven-r2-bn-diagnosis` 及原 UI 分支。没有重定向这些 refs。

## 3. PR #1–#9 实际状态及依赖

初始查询全部 OPEN、均未合并；不是根据标题或旧聊天推测。

| PR | head → base | 状态/处理建议 |
| --- | --- | --- |
| #1 | feature/training → main | OPEN；历史已由后续集成分支包含，勿重复覆盖新加载器 |
| #2 | feat/data-inventory → main | OPEN；v1.1.0 已合入 #6 的历史，来源许可问题仍需负责人判断 |
| #3 | feat/app-shell → main | OPEN；原实现已被 #4 安全修复，不建议直接以旧版覆盖 |
| #4 | integration/mvp-v1 → main | Draft OPEN；已被 #5/#6 继承 |
| #5 | feat/app-real-model-demo → integration/mvp-v1 | OPEN；已被 #6 继承 |
| #6 | integration/mvp-v1.0-delivery → main | Draft OPEN，f77ced3 |
| #7 | feature/phase2-oven-edge-foundation → main | Draft OPEN，7054b16；实际包含 #6，即使 base 写 main |
| #8 | feat/model-oven-first-round → Phase2 foundation | Draft OPEN，20e71f3；依赖 #7 |
| #9 | feat/model-oven-second-round → YOLO first-round | Draft OPEN，5c02296；依赖 #8 |

本轮不自动改这些 PR 的 base、关闭、合并或发布 Release。

## 4. 公开性审查与保全

扫描初始全部 369 个可达 blob 版本（358 个文本、11 个二进制截图），包括后来已删除/被替换的文件，不只查看 HEAD 或 .gitignore；另检查提交消息、转义路径和待推送对象。原有 refs/标签、冻结文件及历史运行产物元数据先保全。

- 未检出私钥头或常见实际凭据 Token 模式；未发现待推送原图、权重、EXE、ZIP、虚拟环境、原始逐图日志。静态模式审查不是绝对无泄漏证明。
- 发现已公开历史报告有个人工作目录。11 张历史截图实际视觉复核，4 张含个人路径，另 1 张含真实食材图且本轮未确认公开转载授权；5 张先逐字节保全本机，再从公开同步树撤下。没有制作替代截图冒充历史证据。
- 重要：普通删除提交不能抹去 GitHub 既有历史，旧分支和旧提交仍可能访问这些内容。没有执行 filter-repo、强推、删除旧分支/标签等清理。是否清理公开历史，需仓库负责人另行决策。
- 待同步研究报告中的本机型号、实际显示设备配置和真实本机数据目录已脱敏。逐目录/逐实物引用表只在原本地分支保留；公开附录改为明确标记的聚合摘要。数量、类别 ID、研究结论和独立验证 BLOCKED 不变。
- 聚合实验 CSV 保留，不上传逐图原始路径或原始采集 JSON。4 份独立验证 CSV 模板均只有表头，没有伪造身份记录。
- 既有公开 Dataset Release 中的来源许可缺口（历史报告记录 235 张未核实）仍需数据负责人处理；本轮未下载/重新分发 Release 图像、未逐项审计 Release 附件。不能把 Git 对象扫描解释为再分发许可通过。
- `.gitignore` 补充二进制交付包排除规则，但推送门禁以所有实际 outgoing blob 版本和提交消息为准，不能靠忽略规则作结论。

## 5. 为什么使用两个公开同步分支

不能把含本机记录的原始未推送提交整条推上公开仓库。因此保留原始本地分支不动，逐提交制作公开脱敏副本：保留原作者和对应顺序，每个提交附 `Source-Commit` 原 SHA。没有改写任何远端历史，也没有宣称原 SHA 已经同步。

- `feat/model-oven-research-public`：基于远端 #9，包含模式审计、监控、A/B、R0/R1/R2、v1.2.2 模型契约和 R2 BN 诊断的公开对应历史。
- `feat/app-oven-integrated-v1.3`：在研究公开分支上接入原 v1.3.0 源码，并正常 merge 研究分支的隐私修复；同时包含 BN 诊断及展示软件。
- 原分支仍用于本机完整溯源；后续不要再把原始研究链整体 merge 进公开副本，避免重复合并或重新引入已撤下内容。新工作从下文指定的公开基线分出。

| 原始本地提交 | 公开对应提交 | 处理 |
| --- | --- | --- |
| `bda2636` | `d055cd2` | 源码/测试/汇总原样对应 |
| `e6d1cb9` | `ae98ba2` | 报告脱敏 |
| `4e8f060` | `d8d91e7` | 报告脱敏 |
| `2bbb1d5` | `ca56e51` | 报告脱敏 |
| `b7164b5` | `3b101a6` | 源码/测试/汇总原样对应 |
| `8be5060` | `02b3cd4` | 报告脱敏 |
| `d65b00f` | `7e5bd3a` | 报告脱敏 |
| `84cfac4` | `5ac3eb2` | 报告脱敏 |
| `793e8ec` | `2bc2209` | 报告脱敏 |
| `b3215ae` | `8a95ecc` | 源码/测试/汇总原样对应 |
| `bceb927` | `0acadf5` | 报告脱敏 |
| `c7d7e62` | `de49486` | 报告脱敏 |
| `6e2f067` | `283659b` | 报告脱敏 |
| `5bc0d45` | `dba96a8` | 报告脱敏 |

另有隐私保护、配方字节保留与正常集成提交；它们没有改动训练参数或模型权重。

## 6. 实际测试及限制

| 检查 | 退出码 / 结果 |
| --- | --- |
| 研究公开分支 `.venv/Scripts/python.exe -m unittest discover -s tests -v` | 0；236 项，235 PASS、1 SKIP，22.554s |
| 完整软件分支相同命令 | 0；261 项，260 PASS、1 SKIP，31.293s |
| Windows 干净检出，显式 core.autocrlf=true | PASS；3 个配方 JSON 与原 Git blob 字节一致 |
| 真实模型 `presentation_smoke.py --catalog ... --baseline ... --out ...` | 0；v2→R1→R2→v2 × 真实 JPG/PNG 共 8 次，原始输出与旧基线逐项相同；6 组软件布局检查 PASS |
| 原研究/软件源码对照 | tools/tests/src/scripts/app/configs/data 分别与 6e2f067 / 5bc0d45 的 Git 内容一致 |
| 发布前历史对象门禁 | 全部 outgoing 版本扫描，无新增二进制、私钥、凭据、个人路径、现场登记引用或本机设备档案命中 |

SKIP 是 Windows 不允许创建测试目录符号链接。软件测试包含明确标注的临时合成输入，不代表新真实训练。真实图片回归只核对软件一致性，不计算新准确率；没有打开摄像头、改模型或重建交付 EXE。

发现并修复的实际问题：首次研究全套测试 236 项有 1 项失败，原因是 core.autocrlf=true 将已 byte-pin 的配方 JSON 改成 CRLF。Git blob 的配方数值/原始 LF 字节正确。新增 `.gitattributes` 的 recipes/*.json -text，恢复原 Git blob 字节，原 SHA 断言未放宽；4 项针对性测试和全套复测通过，干净检出也通过。没有改配方参数、数据划分或训练结果。失败日志保留本机，不把首次失败写成通过。

独立只读复核已完成：没有 Critical/Important 发现；核对 14 组 Source-Commit/作者日期映射、24 个原始 refs/标签、8 个冻结/核心文件哈希及 729 个历史产物元数据，均保持一致。复核未重跑训练或接触服务器。

保留的一项 Minor 文档问题：`docs/ACCEPTANCE_REAL_MODEL_2026-10-21.md` 和 `docs/delivery-evidence/20261021/README.md` 的两处旧截图清单尚未就地增加撤下旁注。它们属于历史记录；本报告已明确 5 张截图只在本机保全、当前公开树不包含这些图片，不能将旧清单理解为当前文件列表。后续可由对应文档负责人补说明，不修改历史数值或伪造替代证据。

## 7. 实际推送与 Draft PR

当前阶段尚未执行推送/PR 创建；测试和安全审查完成后，将用实际远端引用核对及 PR 返回结果更新本节。此处不是已同步声明。

历史 #1–#9 仍未合并；main 仍为 f229bc2。本轮禁止的 force push、main 合并、旧分支/工作树删除、模型覆盖及 Release 均未执行。

## 8. 人工决策与 main 整合顺序

1. 先决定既有公开个人路径/截图的历史处理，以及数据/图像再分发许可；本轮删除当前树中的文件不等于公开历史已清除。旧验收 JSON 的数值记录没有伪改。
2. 按依赖审阅 #6 → #7 → #8 → #9 → 研究公开 PR → 完整软件 PR。#6 已包含 #1–#5 的有效集成历史，通常无需再次逐个重复合入旧版本；旧 PR 的关闭由负责人确认。
3. 若逐个合入会让 main 短暂包含本次已撤下文件，应先将隐私修复纳入相应交付 PR，或由团队整体审阅最终完整集成头后一次合入 main。二选一是人工发布决策，本轮不执行。
4. 堆叠 PR 的父分支合入 main 后，人工核对/调整子 PR 的 base 与 diff；不能只因标题写了“已交付”就跳过测试、依赖及文件审查。
5. BN A/B、R0/R1/R2 保持探索研究属性。训练图回看、已暴露诊断图片与转换一致性不是正式泛化验证；独立实物验证继续 BLOCKED，不据回看最高点自动替换现场模型。

## 9. 下一轮基线

- 全项目、软件演示或同时需要研究工具：**feat/app-oven-integrated-v1.3**，包含最新完整源码、v2/R1/R2 切换、v1.3.0 首页与 BN 诊断公开报告。
- 专注后续模型机制研究：**feat/model-oven-research-public**，含最新 R2/BN 研究代码及脱敏证据，保持旧展示页面基线，避免顺手改生产推理。
- 原始完整研究记录：本地 feat/model-oven-r2-bn-diagnosis；原已验收 v1.3.0 源码：本地 feat/app-oven-presentation-v1.3。它们不是“已原样上传”的分支。
- MVP 历史复现仍用 integration/mvp-v1.0-delivery；不要从仍为 starter 的 main 开始下一轮模型开发。

公开仓库仅交付源码/测试/脱敏报告；真实权重、原始数据、EXE/ZIP、逐图结果和完整本机证据需按现有私下交付说明另行恢复。未恢复真实模型时显示 not_ready 是预期保护，不是可以用模拟预测替代的缺陷。
