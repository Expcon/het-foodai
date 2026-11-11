# 真实烤箱食物BN A/B探索实验

2026-11-11。软件基线 `4e8f060f2334e4c93d55f66df22a4db159bb96b5`，报告分支 `feat/model-oven-bn-real-ab`。两组都实际完成30 Epoch，命令均含显式`--run`，退出码均0。训练和推理软件未改、原数据未改、未导出ONNX或EXE、未自动替换v2。

## 适用范围

**以下全部是同一1,996张已见训练图与候选目录标签的一致性回看，不是独立测试或泛化准确率。** 独立实物验证仍BLOCKED。两组都按预先固定的第30轮保存，没有按曲线高点、现场错误或冻结test选择模型。

训练池为原v2的1428张新增烤箱图加568张去重旧图，另1条旧重复记录排除，顺序、分组和标签保持原样。未加入后来的现场错误照片或额外白玉米诊断集。来源、许可与重新分发限制沿用原审计；原图、权重、个人路径、逐图预测仅存本地忽略runs目录。十二类实验映射与八类MVP协议分开，未冻结或改写共享类别文件。

## 受控绑定与实际运行

- 旧两份prepared文件除`bn_policy`外完整一致；当前配方与它们相同。新运行prepared仅把`training_requested`改为true，两个实际运行之间仍只差策略。
- 数据清单SHA：`d74e39a55f7e4225b4e023d7cf5762379358ac48aef95ab3b7c827378f963181`，实际1996张字节SHA逐项核验，类别ID与既有顺序一致。
- 官方预训练SHA：`c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7`。不从v2接着训练，而是两组从同一官方初始化和新12类头开始。
- A正常更新running mean/var/counter；B冻结初始预训练running统计。两组gamma/beta及其余参数同样可训练，参数量相同。
- 其他条件：seed42、224等比缩放居中补边114、RGB/255、AdamW lr0.001/wd0.0005、batch32、workers0、无AMP、同一翻转/ColorJitter、drop_last=False。相同源码/环境下沿用同一优化器默认项，不引入scheduler或额外EMA。
- 实际GPU：本机 CUDA GPU（型号不公开）；Python3.12、Torch2.14.1+cu130、CUDA13.0。两组顺序执行，没有彼此争用GPU；墙钟时间含数据校验/解码缓存/初始化/epoch0及全部逐轮监控，不是推理时延或端侧性能。

两组初始持久状态指纹相同：`21f84340fb8f13784e2fe184374b01befa70a1702f65a99bc7e4455cfa6d363e`。全部30轮每组1890个训练批次逐一配对，样本索引、组成和增强后输入SHA逐批完全一致，合并追踪指纹为`cef437bdec6f43f92ad36a458dadafd6bb271aa47d43b7348a0353750425dded`。每轮62个32张批次加1个12张尾批，全部样本恰好出现一次；原训练数据和全部配置未出现非预期差异。

39层BN在A中每轮计数各增加63，最终增加1890；B保持初始running指纹和计数。每轮固定eval都验证未写BN缓冲，验证字段始终BLOCKED、未伪造val分数。最终权重的只读统计切换诊断也没有更新任何参数/缓冲。

## 最终逐类结果

| 组别 | 最终在线增强Loss | 在线增强目录一致数 | 固定回看Loss | 固定回看目录一致数 | 总墙钟秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.082890 | 1950/1996 | 0.352672 | 1857/1996 | 587.14 |
| B | 2.452266 | 252/1996 | 2.449785 | 252/1996 | 583.74 |

| ID | 类别 | 图片数 | A固定回看一致数 | B固定回看一致数 | B−A | A Loss | B Loss |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | asparagus | 174 | 174 | 0 | -174 | 0.000017 | 2.453523 |
| 1 | bacon | 160 | 160 | 0 | -160 | 0.001432 | 2.565598 |
| 2 | broccoli | 179 | 179 | 0 | -179 | 0.000038 | 2.371959 |
| 3 | chicken_wing | 191 | 191 | 0 | -191 | 0.001582 | 2.358094 |
| 4 | croissant | 167 | 167 | 0 | -167 | 0.000396 | 2.442273 |
| 5 | french_fries | 182 | 182 | 0 | -182 | 0.008177 | 2.411435 |
| 6 | potato | 163 | 59 | 0 | -59 | 3.515585 | 2.502836 |
| 7 | salmon | 153 | 145 | 0 | -145 | 0.179384 | 2.582292 |
| 8 | toast | 199 | 199 | 0 | -199 | 0.005500 | 2.339261 |
| 9 | corn | 252 | 251 | 252 | +1 | 0.009908 | 2.095583 |
| 10 | sausage | 97 | 71 | 0 | -71 | 1.007825 | 2.944623 |
| 11 | steak | 79 | 79 | 0 | -79 | 0.000018 | 3.143845 |

所有十二类均列出，包含退化和未改善的类别。每类Loss为按图片数加权的交叉熵；类别名称、图片数和母体独立实物数不是同一概念。上述图片高度相关且标签未全部人工签字，不能用这些数宣称真实新场景水平。

## 逐轮证据与复现性

完整31个时点（初始化epoch0及1–30）总体对照见[逐轮CSV](experiments/oven_bn_ab_20261111_epochs.csv)，十二类所有时点见[逐类逐轮CSV](experiments/oven_bn_ab_20261111_per_class_epochs.csv)。原始逐轮JSON保留Loss、Top-1、12×12混淆矩阵、39层BN数组及变化量、所有批次索引/增强SHA、尾批类别组成及EMA权重。没有只保存或展示有改善的轮次。

A全部30轮在线Loss/增强一致数精确重现历史v2；236个持久模型张量逐项完全相同。A文件SHA与v2不同是新metadata等序列化内容不同，不能据此说模型参数不同。旧v2原文件SHA仍不变。A第24和27轮固定回看均1996/1996，而第28轮跌至1750/1996，第30轮1857/1996，说明只看在线总体或挑选回看高点会掩盖问题；这些高点没有用于选模。

## BN差异能解释多少已见错误

有两种不同的对照，不能混为一谈：

1. **固定A最终权重，只替换BN使用的统计方式。** 无增强输入和真实第30轮批次固定，训练归一化使用批次统计但禁止写running缓冲，独立加载副本、无梯度/optimizer。A原有139个目录不一致中68个恢复，同时引入7个新错误，剩余78个。这是固定权重下的直接统计机制效应，依赖指定批次，不能作为生产改train模式的理由。
2. **从同一初始化训练A和B。** BN策略影响整个优化轨迹和最终权重，不只是最后换一次统计。B纠正A的1个错误，同时新增1606个；最终B错误1744个。不能把两组所有变化都归因于同权重上的running统计替换，更不能将其称为泛化改进。

B最终权重只读切换为当前批次统计后也仅164/1996与目录一致，比其固定running统计回看更差。这不是可用修复，两组诊断均未改变生产推理模式。

| 类别 | A错误 | B错误 | A→B纠正 | A→B新增错误 | A同权重切批次BN恢复 | A同权重切批次BN新增错误 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| asparagus | 0 | 174 | 0 | 174 | 0 | 0 |
| bacon | 0 | 160 | 0 | 160 | 0 | 1 |
| broccoli | 0 | 179 | 0 | 179 | 0 | 0 |
| chicken_wing | 0 | 191 | 0 | 191 | 0 | 0 |
| croissant | 0 | 167 | 0 | 167 | 0 | 0 |
| french_fries | 0 | 182 | 0 | 182 | 0 | 1 |
| potato | 104 | 163 | 0 | 59 | 51 | 1 |
| salmon | 8 | 153 | 0 | 145 | 4 | 4 |
| toast | 0 | 199 | 0 | 199 | 0 | 0 |
| corn | 1 | 0 | 1 | 0 | 1 | 0 |
| sausage | 26 | 97 | 0 | 71 | 12 | 0 |
| steak | 0 | 79 | 0 | 79 | 0 | 0 |
| ALL | 139 | 1744 | 1 | 1606 | 68 | 7 |

**本轮B未改善当前问题，发生了全部图片Top-1为corn的类别塌缩。** corn的252/252只是始终预测corn的结果，其预测集合中仅252/1996与目录一致，其余十一类回看均为0；不能将这一行单独宣传为玉米识别提升。potato的B平均CE较A低，也不能掩盖B的Top-1为0。A完整重现v2，没有产生新的性能版本；两组都没有自动接入生产。

新初始化的冻结策略不等于“对训练完成的v2做一次统计校准”。重建并核对相同初始化后，两组119个参数张量均发生变化，其中78个BN affine、41个其他参数张量全部变化，最终参数均有限；这排除了将B误写成全网络冻结的具体实现问题，但不说明更新是有益的。具体核对见本地`final_diagnostics.json`。

## 未解决问题

- running统计选择直接影响预测已确认；其上游原因中，尾批EMA、背景/尺度分布、增强与特征漂移、优化过程各自贡献仍未单独隔离。尾批12张与整批32张采用相同0.1 EMA权重只是一项机制证据，不是已证实根因。
- 冻结全部预训练running统计只是本次B方案；其结果不能代表所有冻结、分层冻结、warm-up或重估统计方案。没有根据回看临时改学习率、batch size、分辨率、裁剪或配方。
- 没有合格独立实物验证、最终测试或跨场景结论；白玉米等实际覆盖缺口并未通过本次同池训练解决；生产v2及现场软件保持原样。
- 本轮仅使用seed42各一次训练，未估计跨随机种子、独立实物和来源的变异；不能把相关帧数当独立样本量计算泛化置信度。

## 产物、实际命令与保全

本地目录：`runs/oven-bn-ab-20261111/`。A权重在`A-update/training/last.pt`，B在`B-frozen-stats/training/last.pt`，均为探索实验产物，不是已验收部署版本。SHA分别为：

- A：`685d252664f1d89f65ed434f056b404173bd2dc0aca7a3691f4f3bbdf6cc0055`
- B：`ed29d5cbfcc991673b81d45fb96161a904ed7f19e40ac2c959fe8147429f9d9b`

```powershell
.venv/Scripts/python.exe -u -m tools.oven_training.controlled --rows runs/oven-round2-20261107/audit/combined_rows.json --source-root $approvedDatasetRoot --recipe tools/oven_training/recipes/bn-update.json --pretrained runs/phase2-edge/export-smoke/yolo11n-cls.pt --out runs/oven-bn-ab-20261111/A-update --device cuda:0 --run
.venv/Scripts/python.exe -u -m tools.oven_training.controlled --rows runs/oven-round2-20261107/audit/combined_rows.json --source-root $approvedDatasetRoot --recipe tools/oven_training/recipes/bn-frozen-stats.json --pretrained runs/phase2-edge/export-smoke/yolo11n-cls.pt --out runs/oven-bn-ab-20261111/B-frozen-stats --device cuda:0 --run
.venv/Scripts/python.exe runs/oven-bn-ab-20261111/compare_epochs.py
.venv/Scripts/python.exe -u runs/oven-bn-ab-20261111/diagnose_final.py
.venv/Scripts/python.exe runs/oven-bn-ab-20261111/plot_results.py
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe runs/oven-bn-ab-20261111/preserve_after.py
```

上述命令均实际运行并退出0；再次复现须改成新的输出目录，不覆盖本次结果。完整单测：191项，1项既有Windows符号链接权限跳过，0失败，53.977秒，退出0。原始A/B日志、完成状态/墙钟、preflight、全部逐轮JSON、错误清单及分析脚本均留在本地；原图和权重未提交Git。

最终保全：2925个分类目录文件、220个现场文件、14个历史受保护文件、127个原跟踪非docs文件及官方预训练原件字节均不变。登记模板SHA`c7b54445c6cd210eaed12322f408b86c48a3c6c063b9786374bcbec06db367b0`保持原值。无软件修补、无生产替换、无PR合并。

真正独立验证的登记表保留在原处，最小现场方案见[独立实物采集方案](OVEN_INDEPENDENT_VALIDATION_MIN_CAPTURE.md)。计划数量不冒充实际采集，不猜测或伪造实物身份；同一实物各角度/层位全部绑定同一集合。现有资格状态保持BLOCKED。
