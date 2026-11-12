# v2 冻结权重的 train / eval 专项审计

日期：2026-11-11。分支 `feat/model-oven-v2-mode-audit`，基于 `bda2636`。此次没有训练、优化器、反向传播、BN重估或checkpoint/ONNX改写。视觉标签审计见 [OVEN_V2_VISUAL_REVIEW.md](OVEN_V2_VISUAL_REVIEW.md)。

## 结论

**已实验证明：BN使用批次统计还是保存的running统计，是本checkpoint在固定输入下train/eval差异的直接机制。** 将分类头模式与BN模式拆开后，输出严格按BN统计方式成对一致。仍未证明running统计异常来自哪一个训练条件；没有发现NaN、负方差、未更新计数或重建时eps/momentum错配，也不能把全部错误归结于BN。

没有发现土豆/香肠的实际训练目标ID错位，或分类头logits/probabilities处理错误。97.7%是训练过程在线、增强、所有十二类合并的统计，不是最终权重固定eval的每类结果。训练实现缺少固定eval回看、每类记录和train/eval差异告警，导致土豆问题没有被训练日志暴露；这属于验收和监控缺口，不能将97.7%当部署指标。

## 原件及实际环境

| 对象 | 固定SHA256 |
| --- | --- |
| v2 `last.pt` | `44878014afe8381055ed2a26fffff6269982e3f5b57c2da0fbedc596127770d5` |
| v2 `oven12.onnx` | `8c696ce5dc63a25053198f1781504b4a53cb801f2d971e3bceeba52eed789689` |
| `combined_rows.json` | `d74e39a55f7e4225b4e023d7cf5762379358ac48aef95ab3b7c827378f963181` |
| 有序类别指纹 | `eeb5260a718f2a1bd42a4d600c8cb2e5ee290c98a5a24e38aa1822e7391827f7` |

使用现有项目 `.venv`、Windows、Python3.12、Torch2.14.1+cu130、torchvision0.29.1、Ultralytics8.3.230、本机 CUDA GPU（型号不公开） CUDA。无全局依赖安装、外网下载或硬件操作。新诊断入口固定针对上述v2，安全加载使用既有 `weights_only=True` 逻辑。

## 同输入、独立副本、零状态更新实验

从真实 `fit_exploratory` 源码提取并执行其嵌套Dataset类，**不调用训练函数**。1996张训练原图SHA全部验证；Dataset先关闭随机增强，再逐项核对标签、DataLoader拼批标签及输入张量。通过相同seed42的独立DataLoader重建第30轮混排；63批，最后一批12张且含多个类别，不是按类别排序的单类尾批。

每个实验独立安全加载checkpoint。所有模型前向均在 `torch.inference_mode()` 下，没有梯度或optimizer。需要批次统计时将副本BN设为训练归一化、`track_running_stats=False`，使其不写running mean/var或counter；该做法与无running缓冲的 `F.batch_norm(training=True)` 一致，已用合成单测验证。这里改变的仅为临时副本模式开关，不回写原件。唯一Dropout的p=0，未引入随机输出。

| 固定无增强输入上的模式 | 全1996张目录一致 | 旧土豆49 | 新土豆114 | 香肠97 |
| --- | ---: | ---: | ---: | ---: |
| 全模型eval，BN用running统计 | 1857 | 9 | 50 | 71 |
| 全模型train，BN用批次统计但禁止缓冲写入 | 1918 | 27 | 82 | 83 |
| 全模型train，仅BN切eval用running统计 | 1857 | 9 | 50 | 71 |
| 全模型eval，仅BN用批次统计且禁止写入 | 1918 | 27 | 82 | 83 |

两组对照各1996张Top-1完全一致、全部概率最大绝对差0。eval与批次BN模式有81张Top-1变化，最大概率绝对差0.9930382371。

土豆原104张错误中51张切批次BN后与目录一致、53张仍不同，同时有1张原本一致的图变错，净增加50张。香肠26张中12张恢复、14张仍错，无新增错误。因而BN影响显著，但没有消除错误。

四个副本前后全部参数和持久缓冲的联合指纹均为：

`dbc50673ebedb6c6af2ad105212a5eb364fae4459b0c449306cf6d20bf1acaa0`

## BN、分类头与日志口径

39个BN层的running统计均有限，方差无负值，`num_batches_tracked=1890=63×30`。重建副本与固定SHA的原官方预训练结构比较，eps均1e-5、momentum均0.1；未发现这些非state_dict属性重建错配。计数说明发生过训练统计更新，不能单凭计数断言统计质量充分。

在相同固定输入上挂钩观察每层BN输入分布，不写统计。eval路径按通道标准化均值偏差RMSE最大0.246882；各层“观察方差/保存方差”的通道中位数范围0.843954–1.875446。它们是整体输入和级联网络的描述，不是校准标准，也没有定位某一层为唯一病因。原始逐层数据保留在 `bn_initial.json`、`comparison/bn_observed_stats.json`。

本机实际安装的Classify.forward在train返回**logits**；eval返回 `(softmax(logits), logits)`。真实训练CE使用logits，诊断分别正确解析；没有重复softmax导致的类别索引错误。训练源代码先得到logits、执行loss/backward/step，再用已计算的logits统计正确数；数值对应每批更新前权重。第30轮在线指标混合63个权重状态，且用随机翻转/ColorJitter、批次BN。最终保存是该轮全部step后的单一权重，二者不是同一个估计量。

历史第30轮：loss=0.0828900344，`train_accuracy_augmented=0.9769539078=1950/1996`。没有保存当时每批logits、全部权重状态或增强RNG轨迹，不能精确恢复1950个正确样本及其每类分布。本次没有伪称重放出历史97.7%。

增强为横向翻转p0.5和brightness/contrast/saturation各0.1，224全图等比缩放、中心补边114、RGB/255。Dataset缓存frame_image后transform再次frame_image，对已为224的图是同尺寸处理；本次DataLoader逐张输入与诊断缓存完全一致。

## 固定权重的增强与批次构成控制

另外5个独立副本均不更新参数/缓冲。增强实验使用新seed42生成的一组固定增强，不是历史第30轮增强回放。

| 条件 | 全部目录一致/数量 | 土豆/163 | 香肠/97 |
| --- | ---: | ---: | ---: |
| eval，同一新固定增强 | 1840/1996 | 63 | 64 |
| 批次BN，同一新固定增强、第30轮混排 | 1906/1996 | 111 | 77 |
| 批次BN，无增强、改第1轮混排 | 1919/1996 | 112 | 84 |
| 批次BN，无增强、故意按单类分批 | 438/1996 | 14 | 10 |
| 批次BN，无增强、逐张推理，仅土豆香肠 | 0/260 | 0 | 0 |

改变批次构成会大幅改变同一张图的预测。**不能把生产推理切成train模式作为修复。** 表中的0/260是当前冻结模型、指定输入与归一化条件的实测，不能外推为其他模型的性质。新增强并未单独解释历史在线指标与最终回看的全部差异。

9个副本的状态前后指纹全部相同，原checkpoint SHA也始终相同。保存的 `control_probability_tensors.pt` 仅为预测概率张量，不是新训练权重。

## 真实训练目标核对

读取原训练rows、实际Dataset源码及真实DataLoader拼批，全部1996张目标与有序类别一致；输出头为12维，CE索引土豆=6、香肠=10。

| 目标 | 实际训练数量 | 目标索引 |
| --- | ---: | ---: |
| 土豆 | 163（旧49+新114） | 6 |
| 香肠 | 97（均新） | 10 |

顺序保持 asparagus、bacon、broccoli、chicken_wing、croissant、french_fries、potato、salmon、toast、corn、sausage、steak。训练目录弱标签不等于人工真值；本次841张独立视觉实查没有发现明确主食材错标，但香肠成分和实物关联仍未证实。

## 130张错误图的来源、场景和尺度

逐图清单 `priority130_clusters.csv` 保存来源、SHA、原标签、视觉标签、两种BN预测、目视背景、原元数据层位、曝光代理、尺寸/外观信息。完整对照母集为 `target260_clusters.csv`，不会只展示成功恢复的样本。

| 错误组 | 数量 | eval的其他Top-1 | 批次BN恢复目录一致 |
| --- | ---: | --- | ---: |
| 旧土豆 | 40 | 三文鱼24、可颂16 | 18 |
| 新土豆 | 64 | 可颂46、三文鱼18 | 33 |
| 土豆目视暗盘 | 88 | 可颂51、三文鱼37 | 43 |
| 土豆目视锡纸 | 16 | 可颂11、三文鱼5 | 8 |
| 香肠目视暗盘 | 22 | 鸡翅20、培根2 | 10 |
| 香肠目视锡纸 | 4 | 鸡翅4 | 2 |

同一批图的名义high/low/mid/unknown错误数量：土豆23/15/26/40，香肠7/12/7/0。层位来自文件/sidecar，未经现场校验，不从画面倒推真实层位。轮次、层位、foil前缀和文件序号不能代表独立物理实物；所有具体实物数和跨场景泛化仍未知。

像素统计对224×126图像内容区计算整幅亮度、暗像素和亮像素比例，不包括灰边。土豆88张暗盘错误的整幅亮度中位数约15.22、锡纸16张约67.77；香肠分别14.87和62.67。**这些是背景强烈混杂的全图曝光代理，不是食材曝光测量。** 不能依据它们认定食物欠曝导致错误。

所有原图1920×1080，进模型后为224×126内容及上下各49像素补边。额外实际查看6张原图，作近似外接矩形，并输出红框和224输入放大图；不是分割真值或精确食材像素占比。

| 匿名图 | 目视食材/背景 | 224输入近似外接框 | eval→批次BN |
| --- | --- | --- | --- |
| V0001 | 土豆、暗盘 | 32×40px | 可颂→土豆 |
| V0002 | 土豆、锡纸 | 49×52px | 可颂→可颂 |
| V0015 | 土豆、暗盘 | 70×49px | 可颂→土豆 |
| V0024 | 浅色香肠、暗盘 | 57×38px | 鸡翅→鸡翅 |
| V0108 | 浅色香肠、锡纸 | 54×67px | 鸡翅→香肠 |
| V0253 | 浅色香肠、暗盘 | 89×34px | 鸡翅→香肠 |

小目标损失细纹理有实图证据，但较大目标也会错；同一固定输入仅改变BN模式即可翻转预测。因此“只因目标太小”“只因反光”均不足以解释当前结果。没有对余下124张伪造外接框，逐图尺寸字段明确为not_annotated、食材像素占比unknown。外观归组为棕黄带皮斑点块茎、浅色斑点成形圆柱，不能外推品种、生熟状态或配料。

## 原因证据分级

| 方向 | 已有证据 | 尚未确定 |
| --- | --- | --- |
| 实现/验收 | BN统计模式因果对照；在线指标不等于最终eval；缺少每类固定eval监控 | running统计不适合当前输入的上游原因；哪一层/学习率/EMA/最后几批的贡献 |
| 已排除的具体实现假设 | Dataset/DataLoader目标错位、头输出语义错、BN eps/momentum重建错配、未更新BN计数、NaN/负方差 | 不代表已证明训练实现所有方面无问题 |
| 分布/覆盖 | 白玉米训练缺口、目标缩小、暗盘/锡纸与外观差异、批次依赖 | 背景、光照、尺度各自的因果贡献；独立新实物泛化能力 |
| 标签 | 841张实际视觉复核，重点130张未见明确主食材错标 | 香肠具体成分、鱼种/成熟度、实物ID；AI判断仍待小规模人工抽查 |

## v3建议（本次未执行）

1. 先建立有可靠实物/会话ID的新实物验证集和最终测试隔离；现场已见错误图片保持诊断用途。无独立数据时只允许探索性描述。
2. 在下一次训练入口增加每轮固定无增强eval回看、每类数量/混淆、train/eval差异和BN统计记录；保留在线增强指标但明确其口径。训练BN计数、最后一批大小及模式入日志。
3. 先固定224预处理与其余配方做受控BN实验：现有正常训练BN方案与明确冻结BN的方案比较。若未来尝试用train-only数据重估BN，须生成独立模型副本、单独版本及元数据，再在合格验证集评估，绝不能覆盖v2或使用最终测试校准。本轮未做任何BN重估。
4. 补采独立土豆、浅色香肠和白/黄玉米实物，记录原料、生熟、来源、背景和层位；先进行精简人工确认，不按预测改标签。
5. BN和数据问题有合格验证证据后，再单独比较ROI或更高输入分辨率，同步训练/推理预处理。禁止同时改多个因素后宣称已提升。

## 命令、退出码及文件

实际运行，全部使用项目解释器；`$approvedDatasetRoot`为用户已授权的新数据目录。新回放输出必须使用不存在的runs子目录。

```powershell
.venv/Scripts/python.exe runs/oven-v2-mode-audit-20261111/inspect_modes.py
.venv/Scripts/python.exe -u -m tools.oven_training.mode_audit --source-root $approvedDatasetRoot --out runs/oven-v2-mode-audit-20261111/comparison --device cuda:0
.venv/Scripts/python.exe -u runs/oven-v2-mode-audit-20261111/controls.py
.venv/Scripts/python.exe runs/oven-v2-mode-audit-20261111/cluster_disagreements.py
.venv/Scripts/python.exe runs/oven-v2-mode-audit-20261111/record_size_examples.py
.venv/Scripts/python.exe runs/oven-v2-mode-audit-20261111/verify_preservation.py
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

上述命令退出码均0。视觉页生成/记录/关联和抽查也退出0；这些记录脚本需要助手实际查看后输入判断，不能自动生成图像真值。

全项目单测：**183项，182通过、1跳过、0失败，70.190秒，退出码0**。跳过是既有Windows符号链接权限夹具。新4项为合成软件测试：无BN写入的批次统计、训练头模式配合running BN等价eval、logits/概率语义、真实Dataset类标签及sampler，不作为真实食品识别指标。

`runs/oven-v2-mode-audit-20261111/` 保存 `inspect.log`、`comparison.log`、`controls.log`、`unittest-final.log`/退出码、逐图四模式概率、BN统计、标签核对、场景判断、尺寸示例及保全结果。`preservation-final.json` 为PASS：2,925个分类目录文件、220个现场文件、14个受保护文件、额外原训练/预处理源码均未变；9个副本参数与缓冲未变。

源代码新增仅 `tools/oven_training/mode_audit.py`、`tests/test_oven_mode_audit.py` 和两份专项报告。原图、权重、ONNX、个人路径和大型预测输出均在忽略runs中。未改共享契约、MVP八类配置/清单、训练/推理产品代码；未合并或推送PR。本轮没有重新执行ONNX推理，上一轮841张PyTorch/ONNX对齐结果及原件SHA仍可核验，不冒充本轮新测。
