# R1单变量学习率：真实探索实验结果

2026-11-11。基线 `8be5060`，独立分支 `feat/model-oven-r1-lr-control`。依据[R0报告预注册方向](OVEN_R0_RESULTS.md)，实际运行一组30轮R1，仅把恒定lr从0.001改为0.0003。原配方、原图、v1/v2/A/B/R0、已有ONNX及生产软件保留，没有训练正式v3或部署。

**本次seed42轨迹中，较低lr对应更小的参数更新与更稳定的后期训练图固定回看；未证明独立实物泛化提升，也没有依据回看最高点选模。** 全文数值是同一1,996张已见训练图与原候选目录标签的一致性，不是正式准确率。独立验证仍 **BLOCKED**，标签、实物/会话身份及来源限制继承原审计。

## 1. 实现与预检

- 新增私有训练配方 `tools/oven_training/recipes/r1-lr-0003.json`，旧bn-update和bn-frozen-stats文件SHA不变。默认recipe校验仍锁定lr=0.001；只有显式lr_control/--observe-r1接受预注册的float 0.0003，并要求正常BN更新。其他lr、错误类型、额外字段及改变seed/batch/weight_decay/BN策略等拒绝。
- AdamW实际读取经验证的options.lr，解决原入口硬编码0.001的问题；其他优化器配置不变。合成和真实checkpoint均检查实际param_groups，不只检查配置文本。
- --observe-r1必须提供已完成精确R0作为reference-run；完整配置仅lr不同，来源/顺序/有序类别/环境、初始持久状态及初始AdamW组除lr外均一致。逐批输入门禁保留；R1允许后续Loss、BN统计和权重分化。
- R0精确复现模式仍要求所有指标与最终张量相等。R1 gate明确为comparison_mode=lr_only、all_reference_metrics_and_batches_equal=false，不能冒充精确R0或直接绕过原cross入口的精确复现要求。

真实数据仍为1,428张新增烤箱图+568张去重旧图，共1,996张；清单SHA `d74e39a55f7e4225b4e023d7cf5762379358ac48aef95ab3b7c827378f963181`。官方初始化文件SHA `c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7`。十二类顺序未改：asparagus、bacon、broccoli、chicken_wing、croissant、french_fries、potato、salmon、toast、corn、sausage、steak。

共同条件：seed42、30轮、batch32/尾批12、workers0、AdamW wd0.0005及相同默认beta/eps等、BN update及相同eps/momentum/affine、224等比缩放居中补114/RGB除255、同一翻转/ColorJitter、无AMP/无scheduler。没有新样本、裁剪、重采样或标签修正；未使用MVP冻结test。

## 2. 实际运行与单变量证据

先通过CPU合成测试、实际YOLO框架CUDA两轮合成对照/恢复、208项完整单元测试，再显式--run启动真实R1。本机 CUDA GPU（型号不公开），Python3.12.10、Torch2.14.1+cu130、torchvision0.29.1+cu130，与R0相同。真实训练退出码0，总墙钟285.3475641秒，包含校验、解码缓存、训练、eval及观测写入，不是算法加速测量。

| 核查项 | 实测 |
| --- | --- |
| 初始持久模型指纹 | 与R0相同：21f84340fb8f13784e2fe184374b01befa70a1702f65a99bc7e4455cfa6d363e |
| 初始模块模式、BN属性和优化器 | 快照逐项一致；optimizer仅lr不同 |
| 全1,890个batch | 索引、类计数、增强输入SHA逐批相同 |
| 固定输入 | 两组31轮共123,752条逐图记录，源SHA、输入SHA、标签/顺序相同 |
| BN更新 | 两组39层均每轮+63，末轮计数1890；统计有限/方差非负，eval不写状态 |
| 真实R1记录 | 31份epoch完整checkpoint+五轮tail pre/post，共41份；每份安全读回/SHA校验 |
| R1尾批恢复 | 5个实际12张pending tensor重放，Loss、模型及357个优化器状态张量均精确匹配post |

第一个训练batch在更新前CE都为2.445793，head/backbone梯度L2都为7.275249/5.707751。首次参数更新L2由R0的0.587863/1.095685降为R1的0.176359/0.328706，约0.3倍，证明实际优化步使用了新lr。后续权重、梯度、动量及统计随轨迹变化，不能把所有后续步当作原R0步的简单0.3倍。

## 3. Epoch 24–30及全部十二类

全31时点/两组62行见[逐轮CSV](experiments/oven_r1_20261111_epochs.csv)；全部12类两组744行含逐类CE、Top-1和样本量见[逐类逐轮CSV](experiments/oven_r1_20261111_per_class_epochs.csv)。所有轮次均保留，没有只报告较好的点。

| Epoch | R0一致数/1996 | R1一致数/1996 | R0固定CE | R1固定CE |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 1996 | 1996 | 0.000385 | 0.000903 |
| 25 | 1995 | 1989 | 0.000628 | 0.010408 |
| 26 | 1983 | 1996 | 0.018069 | 0.000104 |
| 27 | 1996 | 1996 | 0.001308 | 0.000473 |
| 28 | 1750 | 1988 | 1.795395 | 0.012935 |
| 29 | 1975 | 1994 | 0.035098 | 0.003472 |
| 30 | 1857 | 1995 | 0.352672 | 0.008145 |

预注册24–30窗口内，一致数范围R0为1750–1996（跨度246），R1为1988–1996（跨度8）；相邻最大下跌分别246和8。**R1仍有波动，E25还低于R0。** 全部30轮R1最低1938（E8），最大固定CE0.086274；不能把末期较稳定解释为每一轮都满分。

R1重点四类在该窗口的实际一致数：

| 类别/张数 | E24 | E25 | E26 | E27 | E28 | E29 | E30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| potato/163 | 163 | 157 | 163 | 163 | 159 | 163 | 163 |
| sausage/97 | 97 | 97 | 97 | 97 | 94 | 97 | 97 |
| salmon/153 | 153 | 153 | 153 | 153 | 153 | 153 | 153 |
| corn/252 | 252 | 252 | 252 | 252 | 252 | 252 | 252 |

最终第30轮全部十二类，**没有按回看最高点换轮次**：

| 类别 | 样本 | R0一致 | R1一致 | R0 CE | R1 CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| asparagus | 174 | 174 | 174 | 0.000017 | 0.002150 |
| bacon | 160 | 160 | 160 | 0.001432 | 0.000001 |
| broccoli | 179 | 179 | 179 | 0.000038 | 0.000093 |
| chicken_wing | 191 | 191 | 191 | 0.001582 | 0.000003 |
| croissant | 167 | 167 | 167 | 0.000396 | 0.000209 |
| french_fries | 182 | 182 | 181 | 0.008177 | 0.073937 |
| potato | 163 | 59 | 163 | 3.515585 | 0.00000017 |
| salmon | 153 | 145 | 153 | 0.179384 | 0.005678 |
| toast | 199 | 199 | 199 | 0.005500 | 0.000002 |
| corn | 252 | 251 | 252 | 0.009908 | 0.00000031 |
| sausage | 97 | 71 | 97 | 1.007825 | 0.015425 |
| steak | 79 | 79 | 79 | 0.000018 | 0.000111 |
| ALL | 1996 | 1857 | 1995 | 0.352672 | 0.008145 |

R0末轮139个目录不一致在R1末轮均恢复，但新增1张french_fries→chicken_wing（Top-3为chicken_wing/french_fries/corn）。保留本地逐图索引995、SHA、源路径、分数和Loss供复核，未改原标签。部分Top-1不变类别的CE仍增加，不能宣传为所有方面都改善。在线增强末轮Loss/一致数：R0为0.082890/1950，R1为0.027223/1980；在线与固定eval条件不同，不能拿差值当泛化差距。

## 4. 参数更新与BN变化

参数范数按原R0观测定义：head包含model.10所有参数，backbone为其余参数；每步更新L2先算，再取均值，不是整轮净参数差值，也不是分类头线性层单独的范数。

| 描述量 | R0 | R1 |
| --- | ---: | ---: |
| 全1,890步head更新L2均值 | 0.063988 | 0.019460 |
| 全1,890步backbone更新L2均值 | 0.147302 | 0.050413 |
| E24–30每步head更新L2均值 | 0.060705 | 0.029477 |
| E24–30每步backbone更新L2均值 | 0.139400 | 0.070417 |
| E24–30的BN标准化均值漂移中位数再取7轮均值 | 0.062323 | 0.042661 |

BN指标沿用 `sqrt(mean((mu_e-mu_prev)^2/(var_prev+eps)))`，先每层，再39层等权中位数。各组用各自上一轮方差归一化，只描述其统计漂移，不能当作校准误差或正确率敏感度。逐层均值/方差漂移及所有逐batch更新日志保留本地analysis；公共逐轮CSV包含同定义汇总。

降低lr主动改变优化更新，BN数组差异是后续轨迹的一部分。上述结果支持“更新幅度/特征与统计配合参与原失稳”的方向，但没有将最终差异唯一分解为参数或BN的因果百分比。没有同时改BN策略、动量或scheduler。

## 5. 尾批前后及W/S诊断

沿用R0的五轮观测点，R1另做20个只读W/S组合，同样完整1,996张固定输入。W含BN affine，S仅running mean/var/counter；所有混合副本仅作诊断，不保存为部署权重。post对角线精确匹配各轮记录，所有checkpoint SHA保持不变。

| Epoch | R0尾批前→后 | R0净变化 | R1尾批前→后 | R1净变化 |
| ---: | --- | ---: | --- | ---: |
| 24 | 1996→1996 | 0 | 1996→1996 | 0 |
| 27 | 1996→1996 | 0 | 1996→1996 | 0 |
| 28 | 1722→1750 | +28 | 1992→1988 | -4 |
| 29 | 1961→1975 | +14 | 1995→1994 | -1 |
| 30 | 1936→1857 | -79 | 1993→1995 | +2 |

完整四格数值及转移见[尾批CSV](experiments/oven_r1_20261111_tails.csv)。E30的R1四格为Wpre/Spre=1993、Wpre/Spost=1993、Wpost/Spre=1994、Wpost/Spost=1995；统计替换对固定W分别净变化0/+1，参数替换分别+1/+2。R0该步固定S换参数曾净损90/87。**R1未重现该末步的大幅退化，但不是所有尾批都改善：E28仍新增4错，E29新增1错；E24仅换S还会引入1错，完整一步则恢复。**

R1进入尾批前的参数、BN和AdamW历史已与R0不同，不能声称“只把R0最后一步缩小就一定恢复79张”。这是整个30轮lr条件的受控比较，不是固定同一pre状态的单步lr反事实。

## 6. 产物与实际验证

全部新权重、逐图文件及本地路径保存在Git忽略的 `runs/oven-r1-20261111/`：

- `R1/training/last.pt`：预定第30轮探索权重，SHA `fec9b89f29e65dc5f35976e704ebce9900a9c34a9b1c231f5adfdd93b02813cd`；不是best选择或可部署版本。
- `R1/training/observations/checkpoints/`：epoch000–030及24/27/28/29/30尾批pre/post，41份共812,551,585字节，带SHA清单/旁文件、完整优化器/RNG/loader状态。
- `R1/training/observations/eval/`：31轮逐图logits/分数/Top-1/Top-3/Loss/输入及源SHA，及逐类指标、混淆矩阵；原monitor/history与batch_diagnostics保留。
- `analysis/`：62行总体、744行逐类、2340行逐层BN描述和汇总；`tails/`：20组逐图W/S结果；`final_disagreements.json`：剩余候选不一致；`real_tail_recovery.json`：五个真实续步验收。
- `preflight.json`、`verification.json`、`commands.json`、原始训练/测试/诊断日志及exitcode。commands.json含实际本机命令，不提交个人源路径到Git。

| 实际验证命令（项目.venv） | 退出码 | 结果 |
| --- | ---: | --- |
| `python -m unittest discover -s tests -v`（训练前） | 0 | 208项，207通过/1系统符号链接跳过，21.829秒 |
| `python tests/smoke_oven_r1.py --out … --pretrained …` | 0 | 65张合成图，实际YOLO CUDA，A/R0/R1各2轮；同初始化/6批输入相同，仅lr不同，安全续步精确 |
| `python -m tools.oven_training.controlled … --observe-r1 --reference-run … --run` | 0 | 本机 CUDA GPU（型号不公开）真实30轮，1890批输入门禁通过，41份完整观测 |
| 本地compare_runs/replay_real_tails/tail_diagnostics脚本 | 均0 | 全类别对照、五个实际尾步恢复、20个R1尾批组合验证 |
| `python -m unittest discover -s tests -v`（最终） | 0 | 208项，207通过/1相同系统跳过，22.032秒 |

可复现的新运行命令（先在本机设置FOODAI_OVEN_DATA_ROOT，out必须是新目录；本报告没有再次执行此示例）：

```powershell
.venv/Scripts/python.exe -m tools.oven_training.controlled `
  --rows runs/oven-round2-20261107/audit/combined_rows.json `
  --source-root "$env:FOODAI_OVEN_DATA_ROOT" `
  --recipe tools/oven_training/recipes/r1-lr-0003.json `
  --pretrained runs/phase2-edge/export-smoke/yolo11n-cls.pt `
  --out runs/oven-r1-repeat-NEW --device cuda:0 `
  --observe-r1 --reference-run runs/oven-r0-20261111/R0/training --run
```

最终重新逐文件核对2925个数据目录文件、220个现场文件、14个受保护历史文件、官方预训练、145个原A/B文件、454个原R0文件及135个实验启动前软件文件，全部未变。原两个配方也保留字节SHA。没有执行ONNX/EXE构建、相机操作或部署；没有安装/覆盖全局环境。

## 7. 限制与下一步

- 仅一对seed42，且全部为已见训练图候选标签；未有新实物独立验证、跨烤箱/光照正式评估或真实场景性能比较。不能将1995/1996称为新场景准确率。
- R1也有E8、E25、E28等波动，薯条出现新的不一致；不根据模型预测自动改标签或只展示已改善的类别。
- 权重、优化器与BN沿整条轨迹共同变化，尚不能确定具体哪些骨干/BN affine通道或动量项主导；本轮不再扩展超参数搜索。
- 优先按[真实实物登记流程](OVEN_INDEPENDENT_VALIDATION_INTAKE_GUIDE.md)建立可核验来源/母体/会话分组及人工标签的独立开发验证。旧训练来源证据缺口未补证前继续BLOCKED，不伪造ID。
- 如后续还需机制复核，先预注册额外成对seed，在相同seed内保持唯一lr差异并重复相同日志检查；不在本轮启动。正式v3、选模、ONNX/EXE/现场替换均需另行合格验证与决策，不由本次训练回看自动推进。
