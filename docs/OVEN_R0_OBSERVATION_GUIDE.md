# R0观测、校验及恢复接口

2026-11-11。[真实R0结果](OVEN_R0_RESULTS.md)已完成；以下接口用于可复核实验，不是GUI模型格式或正式v3训练入口。原训练默认路径保持原行为，只有显式启用R0记录器才保存新观测格式。

## 重新运行需使用新目录

使用项目.venv和已核验的官方预训练文件。先在本机设置FOODAI_OVEN_DATA_ROOT为已有只读源目录；不要在Git文档填写个人路径。下面out仅作**未来新运行**示例，不能复用已经完成的 `runs/oven-r0-20261111/R0`：

```powershell
.venv/Scripts/python.exe -m tools.oven_training.controlled `
  --rows runs/oven-round2-20261107/audit/combined_rows.json `
  --source-root "$env:FOODAI_OVEN_DATA_ROOT" `
  --recipe tools/oven_training/recipes/bn-update.json `
  --pretrained runs/phase2-edge/export-smoke/yolo11n-cls.pt `
  --out runs/oven-r0-repeat-NEW `
  --device cuda:0 --observe-r0 `
  --reference-run runs/oven-bn-ab-20261111/A-update/training --run
```

移除--run只做prepared预检。--observe-r0和--reference-run必须成对提供；构造记录器时核对完整配置及初始化，每个batch核对索引/组成/增强SHA，每轮核对online/fixed eval/BN状态，最后核对所有持久张量。任何不一致抛错并退出，没有gate PASS不得继续cross。

输出已有即拒绝，所有产物只能在Git忽略runs子树。原模型或旧实验目录不能作out。写入采用排他临时文件、fsync及无覆盖hard-link发布；失败时可能留下.writing或不完整材料，不能据此声称该snapshot完成，禁止删除旧文件后原地重试。使用新目录调查。

## 快照和逐图记录

`observations/checkpoints.jsonl`是追加清单：文件相对路径、SHA256、epoch、global step、phase、字节数及模型状态指纹。每份.pt有.sha256旁文件，写完立即用weights_only=True读回核对。31轮末checkpoint含epoch0；24/27/28/29/30还存tail_pre和tail_post。41份.pt占812,538,401字节，是本轮实测值，未来不同数据/环境不可套用作容量保证。

`format=oven-r0-observation/v1`保存：model_state、optimizer_state、epoch/global_step/phase、所有Python/NumPy/Torch/CUDA RNG、train/fixed（及存在时validation）的private generator、模块训练模式、BN属性和完整绑定。NumPy状态转基础列表，未序列化任意Python模型对象；只能使用安全载荷读取，不添加weights_only=False。

逐图eval包括固定索引、源SHA/路径/候选标签、输入张量SHA、raw logits、CPU float32 softmax分数、Top-1/Top-3及逐图CE；softmax分数不是校准可信度。每轮额外输出样本加权总体/逐类Loss、Top-1和混淆矩阵。固定eval恢复混合模块模式和全局RNG，并检查模型/BN未被写入。

## 精确恢复API（workers=0）

接口位于 `tools/oven_training/observation.py`。恢复应在**相同源文件、预处理、模型结构、代码/版本、device及确定性配置**下，先核对binding中的SHA和配置，再使用：

```python
from tools.oven_training.observation import load_snapshot, restore_training, resume_batches
from tools.oven_training.monitor import set_training_mode

payload = load_snapshot(snapshot_path, expected_sha_from_verified_index)
# model、optimizer及train/fixed loaders按train.py同样构造；在此之前核验环境与来源。
# verified_current_binding应来自当前环境/来源核验，不能盲信未核查的checkpoint声明。
restore_training(payload, model, optimizer, loaders, verified_current_binding)
if payload['phase'] == 'epoch_end':
    set_training_mode(model, payload['binding']['config']['bn_policy'])
for images, labels, indices in resume_batches(loaders['train'], payload):
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(images), labels)
    loss.backward()
    optimizer.step()
    # 从payload.global_step继续；结果只写新的恢复实验目录。
```

该接口恢复数值状态与loader迭代，不提供把旧run原地续写历史CSV的CLI；恢复程序应在新目录延续epoch/step与指标累计，禁止覆盖原快照或把部分轮指标标成整轮。

- epoch_end：该epoch已经完成，恢复后是下一轮第一批，epoch计数应+1，进入该轮前必须按保存配方调用set_training_mode。epoch0忠实保留了官方初始化的混合模式（根/分类头eval，新线性层train）；不能直接将eval返回的tuple送入CE。
- pre_forward：当前批已由loader取出/增强但尚未forward；pending保存原张量，不能再抽一次增强。其余梯度会在下一步zero_grad清掉。
- post_step：当前批已优化完成，不再执行一次。如果是尾批，迭代结束后进入下一轮。

pre_forward/post_step先保持已保存模块模式；其迭代结束后进入下一轮时同样调用set_training_mode。恢复API不自动把所有BN改train，以免破坏已保存的frozen_stats或其他混合模式。

stock loader未替换采样器。恢复mid-epoch时，按保存的epoch-start generator重建原迭代器、跳过已取批次、验证索引和current generator，然后恢复所有训练RNG，最后返回pending或下一批。跳过阶段重新计算的增强会丢弃，其随机消耗也被恢复；只有workers=0及同样的只读确定性缓存/数据集才受此保证。不支持多worker、随机外部IO或改变sampler后声称精确恢复。尾批文件包含完整实际排列及取样游标，epoch末保存private generator供下一轮shuffle。

软件测试已覆盖Python/NumPy/Torch随机增强、dropout、CPU中途/跨epoch、CUDA尾批及AdamW状态精确性；实际YOLO合成续步及五个真实12张尾步也逐项一致。完整30轮从中间点恢复到结尾没有另跑一次，不将单步证明表述为该项已实际完成。

## W/S只读交叉诊断

先要求R0 gate同时为PASS、reference_bound=true、原A精确匹配，再运行：

```powershell
.venv/Scripts/python.exe -m tools.oven_training.cross_states `
  --observations runs/oven-r0-20261111/R0/training/observations `
  --rows runs/oven-round2-20261107/audit/combined_rows.json `
  --source-root "$env:FOODAI_OVEN_DATA_ROOT" `
  --out runs/oven-r0-cross-NEW --device cuda:0
```

每个组合独立模型副本，W含所有参数及BN affine；S仅39层running统计/计数。严格核对键、shape、dtype、BN eps/momentum/track设置及源图SHA，使用相同batch32无增强eval。不能配不同类别顺序或预处理。未知额外buffer拒绝混合，避免未定义的权重/统计归属。

两对epoch与五对尾批共28组合。对角线须匹配原轮记录（尾批pre此前没有原轮末指标）；每张图配对检查target/input SHA，保存恢复/新增错误及wrong→wrong变化，不只记录净分数。完成后重新核对全部41个输入checkpoint的SHA，混合模型从不落盘为部署权重。

真实独立验证仍BLOCKED，cross不评估泛化、不自动选最高回看模型。原有12类实验顺序及MVP八类配置完全保留，不运行ONNX导出或HeT OvenCapture生产替换。
