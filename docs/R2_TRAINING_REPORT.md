# R2整根玉米：真实探索训练报告

2026-11-12。基于`b3215ae`，独立分支`feat/model-oven-r2-whole-corn`。依据[R2数据审计](R2_DATA_AUDIT.md)，在本机 CUDA GPU（型号不公开）实际完成30轮，不是准备状态或合成训练。**没有合格独立验证集；以下Loss/Top-1均为训练候选标签诊断，不是泛化准确率。** 没有根据训练最高点选模，没有替换R1软件或默认模型。

## 固定配方与不可避免的差异

| 项目 | 实际R2 |
| --- | --- |
| 架构/类别 | YOLO11n-cls，12类，corn仍为ID9 |
| 数据 | 1,922张：其他11类1,744+旧整根128+新整根白玉米50 |
| 初始化 | 同一官方预训练文件，SHA `c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7` |
| 初始持久模型指纹 | `21f84340fb8f13784e2fe184374b01befa70a1702f65a99bc7e4455cfa6d363e`，与R1完全相同，训练前门禁核对 |
| 优化器 | AdamW，lr0.0003、weight_decay0.0005，beta/eps等沿用R1默认设置；无scheduler |
| 轮次/种子 | 固定30轮、seed42，不选择best轮 |
| 输入 | 224×224，RGB，Pillow双线性等比缩放居中补114，除255，NCHW |
| 增强 | horizontal_flip概率0.5，brightness/contrast/saturation各0.1的ColorJitter，缓存原固定框图而非缓存随机增强 |
| BN | 正常更新，affine可训练，39层momentum均0.1 |
| 批次 | 基数32，显式merge_short_tail/min8；59×32+34=60批/轮，全样本恰好一次 |
| 运行 | workers0、无AMP、确定性算法、cudnn.benchmark=false；项目既有`.venv`，未安装依赖 |
| 验证/测试 | 无合格val、无冻结test；不使用MVP测试集 |

R1原数据1,996/尾12/63步每轮；R2删除124张旧黄色切段、加入50张合格新图，尾34/60步每轮。数据、研究范围、批次组成、BN更新次数和总步数都不同；末批平均Loss的归一化样本数也不同。**不能把R2相对R1变化单独归因于删除半根玉米或某一种尾批措施。** min8是预先记录的工程策略，不是正式验证选出的最优参数。

正常旧入口仍保持原recipe约束；R2必须显式`r2_control`才接受新尾批字段，R1的单变量学习率门禁没有放宽。新增`record_only`只记录独立实验，不冒充R0精确复现或R1单变量比较。

## 实际训练及完整观测

实际命令在项目根目录运行；下列source变量指向用户授权的两个只读数据根，使用既有项目环境：

```powershell
.venv/Scripts/python.exe -u -m tools.oven_training.r2 `
  --rows runs/oven-r2-whole-corn-20261112/r2_rows.json `
  --expect-rows-sha256 9a580709f353903d5a5337acb7280d1b93437dceddd4d7bd08844bb76c334ff1 `
  --old-source-root $oldSourceRoot --new-source-root $newSourceRoot `
  --pretrained runs/phase2-edge/export-smoke/yolo11n-cls.pt `
  --reference-initialization runs/oven-r1-20261111/R1/training/monitor/initialization.json `
  --out runs/oven-r2-whole-corn-20261112/R2 --device cuda:0 --run
```

先运行不带`--run`的独立preflight目录，确认1,922张/60批/尾34，再通过全部软件及CUDA合成预检后运行真实训练。训练退出**0**，metadata训练阶段计时233.022秒（不含初始化/第0轮，也不作速度比较）；实际Python3.12.10、Torch2.14.1+cu130、Ultralytics8.3.230、CUDA13.0，与既有环境一致。

- 0–30轮共31份固定无增强eval，含逐图源SHA、固定输入SHA、12维logits/分数、Loss、Top-1/Top-3；共59,582条逐图记录。
- 每轮完整12类Loss/Top-1、12×12混淆矩阵、BN状态；全部1,800个训练batch记录索引、增强输入SHA、类别构成和参数/梯度变化。
- 0–30轮31份完整训练状态+24/27/28/29/30轮各一对尾批pre/post，共**41份checkpoint**，保存optimizer、步数、Python/NumPy/Torch/CUDA RNG和LoaderCursor/生成器状态，逐份SHA校验并`weights_only=True`安全读回。
- 每个真实epoch均核对1,922个索引恰好一次，批量始终59×32+34；全部BN状态有限，各层每轮counter+60，最终1800；固定eval不写BN。
- 对五个真实34张尾批的pre状态独立加载、恢复RNG并重新执行一次更新，模型持久张量及每次357个optimizer状态张量均与post精确相同。只做重放诊断，没有追加正式训练轮次或改写原checkpoint。

全量证据在`runs/oven-r2-whole-corn-20261112/R2/training/`：`history.csv`、`fixed_config.json`、`monitor/epoch_*.json`、`observations/eval/`、`observations/checkpoints.jsonl`和`observations/checkpoints/`。`training_verification.json`提供上述覆盖/重放检查及本轮新增/修改的五个运行模块SHA，含新增`batching.py`和`r2.py`；checkpoint基础binding仍保留其原四模块字段，不事后改写。

## 每轮固定训练回看

全部31时点见[逐轮CSV](experiments/oven_r2_20261112_epochs.csv)，12类×31时点共372行见[逐类逐轮CSV](experiments/oven_r2_20261112_per_class_epochs.csv)。初始0轮也保留，未只展示最优轮。

| Epoch | 与训练候选一致/1922 | 固定eval CE |
| ---: | ---: | ---: |
| 1 | 1918 | 0.013226 |
| 10 | 1919 | 0.012541 |
| 11 | 1876 | 0.056856 |
| 12 | 1922 | 0.000264 |
| 24 | 1922 | 0.000012 |
| 25 | 1910 | 0.023494 |
| 26 | 1895 | 0.042316 |
| 27 | 1920 | 0.003351 |
| 28 | 1920 | 0.004668 |
| 29 | 1920 | 0.003015 |
| 30 | 1922 | 0.000154 |

R2仍有波动，已训练轮中最低第11轮1876/1922；第25、26轮也退化。不能声称合并尾批已消除BN风险。最终轮增强训练Loss0.005136、一致1920/1922；固定eval为1922/1922、Loss0.000153688。增强在线统计和无增强固定eval条件不同，不可将差值称泛化差距。

最终30轮完整十二类如下；均为训练图：

| ID / 类别 | 张数 | Top-1一致数 | 固定CE |
| --- | ---: | ---: | ---: |
| 0 asparagus | 174 | 174 | 0.00000506 |
| 1 bacon | 160 | 160 | 0.00002482 |
| 2 broccoli | 179 | 179 | 0.00004453 |
| 3 chicken_wing | 191 | 191 | 0.00003388 |
| 4 croissant | 167 | 167 | 0.00018227 |
| 5 french_fries | 182 | 182 | 0.00006167 |
| 6 potato | 163 | 163 | 0.00000221 |
| 7 salmon | 153 | 153 | 0.00000080 |
| 8 toast | 199 | 199 | 0.00001473 |
| 9 corn | 178 | 178 | 0.00086877 |
| 10 sausage | 97 | 97 | 0.00078594 |
| 11 steak | 79 | 79 | 0.00000178 |

末轮12×12矩阵位于`monitor/epoch_030.json`及`observations/eval/epoch_030_metrics.json`，其对角计数即上述张数，非对角为0。每轮完整矩阵均保存，不仅末轮。

## checkpoint、测试与限制

最终可安全读取的模型：`runs/oven-r2-whole-corn-20261112/R2/training/last.pt`，SHA256：
`10caa092f31dbf9f57d15509dd8e525d0bba29457d3f70895cc7c8c9af7532a2`。
`run_info.json`、`fixed_config.json`与本轮`r2_recipe.json`保留完整配置。没有覆盖v1/v2/A/B/R0/R1。

训练前及最终完整单测均为232项、231通过、1项Windows软链接权限跳过、0失败、退出0；最终20.333秒，见`unittest_final.log`及对应退出码。7项R2专项与两组真实YOLO/CUDA合成尾批验证均通过。真实训练、41份安全加载/覆盖/尾批重放核验与成对诊断命令均退出0。原始失败、合成RED测试和异常日志都留存，不冒充通过。

最终保全核验：212项现有历史保护文件SHA不变；新来源102个PNG/JSON、旧分类来源2,925个文件、原补充144张及旧现场220项均与训练前快照相同，见`preservation_final.json`。任务开始前已缺少的R1 ZIP单独列出，没有被本轮删除。共享classes.json、旧训练/测试清单和现有UI未改。

本轮没有导出ONNX、修改实验列表或制作R2 EXE，现有R1/v2演示软件及默认配置保持原样。未完成转换一致性检查前，不提供可部署R2版本。真正独立测试仍**BLOCKED**；50张新增训练图及原已暴露诊断不能作正式成绩。与R1的完整逐图对照、改善及退步见[R1/R2比较](R1_R2_COMPARISON.md)。
