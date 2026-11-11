# 烤箱训练监控与BN受控实验准备

2026-11-11。基于v2审计提交 `e6d1cb9`，独立分支 `feat/model-oven-training-monitor`。本次修复训练验收机制，未启动真实食物30轮训练，未训练/交付v3，没有修改Desk生产推理、旧模型或公共契约。

## 已实现的训练验收

原 `fit_exploratory` 保留最终轮保存策略、优化器和训练参数。所有原调用方自动获得下列监控；两组受控入口也复用该函数。

- 初始化及每个Epoch结束，对同一训练图片缓存按固定顺序、无随机增强、`model.eval()` 前向。训练回看与独立验证明确分开，不将train回看伪装为val。
- 逐类记录样本量、正确数、Top-1、按样本加权的CE Loss；保存完整有序混淆矩阵（行=目标，列=预测）。缺席类别Loss/Top-1为null，空数据不能报告通过。
- 监控39层BN的running mean/var、eps、momentum、counter、训练模式、affine可训练状态、前后统计指纹及逐层变化量。非有限值、负方差、非预期counter更新、冻结组统计变动会中止运行。
- 每轮记录所有实际批次的样本索引、数量、类别组成及增强输入SHA。最后一批另记录BN变化、EMA权重和小批次标记。
- eval在无梯度模式执行，并在成功/失败后恢复原来的混合模块模式、Python/NumPy/Torch随机状态；独立DataLoader generator不消耗训练shuffle RNG。固定eval前后核对BN指纹。
- 每轮调用`model.train()`后重新施加冻结策略，防止第二轮无意重新开启BN更新。没有将部署推理改成train模式。

训练原在线指标 `train_accuracy_augmented` 保留，但它混合每批更新前的模型状态和随机增强，不能当最终模型指标。新增 `history.csv` 字段为 `train_eval_loss`、`train_eval_top1`、`val_eval_loss`、`val_eval_top1`；无合格验证集时后两项为空，JSON明确BLOCKED。epoch秒数包含批次证据记录和轮末eval，不与旧纯训练秒数直接比较；初始化和epoch0 eval另在此计时之外。

## 缓存、初始化及原始预训练BN检查

仍缓存确定性的224px等比缩放、中心补边114的RGB PIL图像，随机翻转和ColorJitter在每次训练访问时采样，没有将随机增强缓存成固定训练图。固定eval共享该确定性缓存，使用无增强变换，原图只读。未使用SDK默认裁剪替换项目预处理。

实际检查本机Ultralytics8.3.230的 `ClassificationTrainer.get_model`：pretrained开启时加载权重，不走`not pretrained`的reset_parameters分支；项目实际训练调用自有循环，并不运行SDK Trainer。项目 `new_model` 从固定SHA的官方文件加载模型、调整最后线性头为12类，并使参数可训练，没有重置BN running统计。

本次实际加载的官方预训练模型39层BN均为eps=1e-5、momentum=0.1、counter=0，affine均可训练；其已保存mean/var保留。**counter=0不代表mean/var是未训练默认值。** 官方原件SHA：

`c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7`

历史v2的39层counter=1890=63×30，说明自有训练循环发生过running统计更新。其原始1996张使用batch32、不丢尾批，每轮63批，末批12张。momentum0.1下末批与完整32张批次具有相同10% EMA权重；按样本粗分，12张尾批的每张影响约为完整批次的32/12倍。实际BN还对空间位置统计，这不是精确的逐图片归因。

该机制可能增加对尾批构成的敏感性，但**尚未证明它是v2失败的上游原因**。本次没有丢弃尾批、调momentum、改变batch size或重估v2 BN；两组均保留全部数据。新日志为后续单因素实验保留证据。

## 两组受控方案

| 项目 | A：正常BN | B：冻结running统计 |
| --- | --- | --- |
| `bn_policy` | `update` | `frozen_stats` |
| running mean/var/counter | 随训练批次更新 | 保持初始化值 |
| gamma/beta | 可训练 | 同样可训练 |
| 其他模型参数/优化器成员 | 可训练 | 相同 |

这里“冻结BN”专指冻结running统计，**不是连gamma/beta一起冻结**。这样单独比较已证实的统计机制；不混入可训练参数数量变化。冻结预训练统计可能与新烤箱分布及持续变化的上游特征不匹配，不能预设B更好。

两个JSON配方仅 `bn_policy` 不同：相同数据清单、十二类顺序、官方初始权重、seed42、224/RGB/中心补边114/255、AdamW、lr0.001、weight_decay0.0005、batch32、workers0、无AMP、同一翻转/ColorJitter、drop_last=False、固定30轮、只保存最后轮，不按训练回看选择模型。30轮是未来受控配方定义，本次未执行真实数据训练。

运行前检查prepared文件的数据SHA和配方；运行后比较initialization的模型状态指纹、可训练参数数，以及每轮各批次的索引和增强输入SHA。任何不一致都必须先解释，不能只比较分数。未增加自动选择“更好”一组的逻辑。

## 独立实物验证：当前BLOCKED

对原v2训练池执行两次只读准备，均得到1996张、相同类别及清单SHA，没有请求训练：

- 1996张均缺少已核实物理身份/会话身份及人工签字记录。
- 569张记录缺少可用physical_sample_id和session_id，其余已有字符串也未经核实；不能把文件前缀或编号直接升级为实物身份。
- 验证候选0张，十二类验证覆盖均缺失。没有新建一个假独立划分。

提供空白 `tools/oven_training/recipes/physical-validation-intake.csv`。它是新增实验侧登记表，不替代九字段公共manifest。收集真正新实物后，由采集者/数据负责人填写：

1. 给物理食材分配持久、跨层位/锡纸/拍摄会话不变的实物ID；会话另给持久ID。同一物体换角度或换盘不能变成另一实物。
2. 记录原图relative_path、来源、许可、类别ID/key、字节SHA、像素SHA、去重关联group_id；保留原文件。分类和实物关系须由采集记录与人工确认，不由预测生成。
3. 核实后填写physical_identity_verified、session_identity_verified、identity_reviewed_by/at；验证标签另需trusted_label和label_reviewed_by/at。CSV布尔字段只接受true/false，空白仍为未确认。
4. 训练登记也要能证明物理/会话身份，否则无法证明新验证实物与训练图不重叠。先完成已知重复/近重复关联，再确定val；最终test另行冻结，不传给本训练入口。
5. 以来源索引和原图审计支撑登记。当前gate检查声明、类别映射、SHA/像素SHA/组/实物ID/会话ID跨集合重叠，以及验证集内部重复字节/像素和同图冲突标签，不能重复同图凑类别覆盖；身份声明不是物理独立性的自动证明，仍需核验。资格状态也不代表统计样本充足、类别冻结或正式泛化结论。

提供不合格验证文件会直接拒绝，不会静默回退成train-as-val；不提供验证文件可准备/显式启动探索训练，状态保持BLOCKED。即使未来登记合格，checkpoint仍保留现有实验格式 `independent_validation=False`，验证监控单独标记，不擅自升级十二类正式契约。

## 复现命令

沿用项目 `.venv`。以下变量指向用户明确授权的新数据目录，不做其他目录搜索。权重及运行输出均不得提交Git。所有输出须使用**不存在的新runs子目录**。

```powershell
# 只检查数据、初始化文件和方案，默认不会训练。
.venv/Scripts/python.exe -m tools.oven_training.controlled --rows runs/oven-round2-20261107/audit/combined_rows.json --source-root $approvedDatasetRoot --recipe tools/oven_training/recipes/bn-update.json --pretrained runs/phase2-edge/export-smoke/yolo11n-cls.pt --out runs/bn-plan-A
.venv/Scripts/python.exe -m tools.oven_training.controlled --rows runs/oven-round2-20261107/audit/combined_rows.json --source-root $approvedDatasetRoot --recipe tools/oven_training/recipes/bn-frozen-stats.json --pretrained runs/phase2-edge/export-smoke/yolo11n-cls.pt --out runs/bn-plan-B

# 未来明确决定做探索实验时，另选新输出目录并显式添加 --run。
# 有合格验证登记时添加 --validation-rows <CSV或JSON>；不可传最终test。
# 当前没有合格验证集，不得比较两组正式泛化性能。

# 短程软件冒烟：65张随机像素、任意合成标签，两组各2轮。
.venv/Scripts/python.exe -u tests/smoke_oven_bn_control.py --out runs/SYNTHETIC-bn-pair --pretrained runs/phase2-edge/export-smoke/yolo11n-cls.pt --device cuda:0

.venv/Scripts/python.exe -m unittest discover -s tests -v
```

不要用`python -m tests...`运行本仓库冒烟脚本：当前环境有同名第三方tests包。首次错误调用退出1，日志已保留；正确的直接脚本命令退出0。该错误导入新建了一个用户级SDK设置文件，核对创建时间/内容指纹后已移除，未修改既有配置或安装依赖。清理其新建空目录时自动审批返回`blocked by policy`，目录保留，未换方式重试。之后实际SDK配置均写入新runs目录。

## 实际验证与证据

全部证据位于忽略目录 `runs/oven-training-monitor-20261111/`。

| 实际命令/检查 | 结果 |
| --- | --- |
| 改动前全部单测 | 183项，182通过/1跳过，68.886秒，退出0 |
| 新监控TDD首次运行 | 缺少monitor模块，预期失败，退出1；`tdd-red.log` |
| CSV登记TDD新增测试 | 缺少read_rows，预期失败，退出1；`tdd-intake-red.log` |
| 验证重复图TDD新增测试 | 同图重复能错误READY，预期失败后修复，退出1；`tdd-duplicate-red.log` |
| 新增定向测试（修复后） | 8项通过，5.861秒，退出0；`targeted-tests-after-review.log` |
| 两组真实数据prepare | 各1996张；同一rows SHA；validation BLOCKED；training_requested=false；各退出0 |
| 合成GPU A/B，现有本机 CUDA GPU（型号不公开） | 各65张、2轮；完整YOLO11n-cls；3批/轮=32/32/1；退出0 |
| 新监控只读重放真实v2训练图 | 1996张；1857与目录一致；土豆59/163、Loss3.515585；香肠71/97、Loss1.007825；全参数/缓冲不变；退出0 |
| 改动后全部单测（复核修复后） | **191项，190通过/1跳过，0失败，55.788秒，退出0**；`unittest-after-review.log` |

既有跳过项为Windows测试目录符号链接权限；不是GPU或新增监控失败。合成软件测试不代表真实食品准确率。真实回放是已见训练图与候选目录标签的一致性诊断，不是验证集或新场景泛化成绩。首次全套190项日志保留在`unittest-final.log`，以修复后的191项结果为最终验证。

只读复核发现并修复验证集内部重复图漏洞，回归测试覆盖同标签重复、冲突类别、重编码像素重复及真正不同图正常放行。复核者确认该问题已关闭。最终保全核对2,925个分类目录文件、220个现场文件、14个历史受保护文件，全部SHA一致，详见`preservation-final.json`。

GPU合成A/B实际证明：相同初始模型状态指纹及可训练参数量；两轮所有shuffle/增强输入逐批完全一致；正常组每轮39层BN计数各增加3；冻结组running指纹始终不变；全部固定eval的BN指纹不变；新增的两个合成checkpoint均可由既有安全加载器读取。最后1张尾批在正常组仍以EMA0.1更新；第二轮最后一次BN均值更新的层间最大绝对量为0.1248889，仅是合成输入软件证据，不能归因到真实v2。

主要输出：

- `prepared.json`：配方、类别顺序、训练清单/预训练SHA、验证资格、是否实际请求训练。
- `training/monitor/initialization.json`：初始模型及BN状态、缓存说明、源码SHA和可训练参数数。
- `training/monitor/epoch_000.json`及每轮JSON：逐类固定eval、Loss、Top-1、混淆矩阵、BN变化、批次追踪、末批诊断；无验证时明确BLOCKED。
- `training/history.csv`、`fixed_config.json`、`run_info.json`、`last.pt`：新增独立运行的历史与最终权重，不覆盖旧运行。
- 本次合成冒烟使用 `SYNTHETIC-gpu-pair/update/` 和 `frozen_stats/`；目录下 `summary.json`汇总软件断言，不能交给现场当v3模型。
- `real-v2-fixed-eval.json`：新监控在真实v2上的只读对照；`preservation-final.json`记录历史文件保全。

## 已定位与待验证

已定位：固定输入上BN统计方式会造成v2模式差异；历史在线增强总体统计掩盖固定eval逐类问题；缺少合格独立实物验证。现有新增监控能暴露这些问题，并保证冻结统计策略跨Epoch持续生效。

仍属假设：末批权重、EMA记忆、学习率/特征漂移、背景/尺度及真实物体覆盖各自对v2失败的贡献；冻结统计能否改善合格验证集；新实物泛化能力。本次不报告哪一组更好。

下一步先补齐实物/会话登记和真正独立验证数据，再按同一配方运行A/B；先比较固定eval行为和BN证据，合格验证集上的结论需报告样本数及实物数。ROI/更高分辨率、全BN affine冻结、MNN/INT8及部署均未在本次实施。旧v1/v2权重、ONNX及MVP配置/数据清单保持不变，不合并现有PR。
