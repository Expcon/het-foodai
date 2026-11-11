# 独立验证现场登记：先填简表，再做资格审核

2026-11-11。本次只优化登记工作流程，没有采集新图、核实新身份或改变任何已有清单。旧训练池身份/会话证据不完整，资格仍为 **BLOCKED**。本流程配合[最小现场采集方案](OVEN_INDEPENDENT_VALIDATION_MIN_CAPTURE.md)，不改变其来源隔离及封存要求。

## 原流程的问题与最小调整

原 `physical-validation-intake.csv` 同时要求人填写图像SHA、ID、标签与审核字段，没有层位/锡纸等显式记录位置。组员容易将技术字段手工抄错，或把文件序号误当实物ID。**原技术表和controlled资格门禁保留原样**；新增三张空白简表，实物/会话的共同信息只登记一次，每张图只补路径、关联ID和场景。

| 空白模板（tools/oven_training/recipes/） | 一行代表 | 现场填写 |
| --- | --- | --- |
| [physical-validation-objects.csv](../tools/oven_training/recipes/physical-validation-objects.csv) | 一个有来源证据的真实实物 | 持久实物ID、共同母体/来源组、批次、类别候选与原料依据、身份/标签审核、预定集合 |
| [physical-validation-sessions.csv](../tools/oven_training/recipes/physical-validation-sessions.csv) | 一次真实拍摄会话 | 会话ID、烤箱/相机代号、实际起止时间、照明、身份审核 |
| [physical-validation-photos.csv](../tools/oven_training/recipes/physical-validation-photos.csv) | 一张实际存在的照片 | 相对路径、实物ID、会话ID、层位、锡纸、背景、角度；照明变化时填override |

模板只有header，不含虚构实物或签名。复制到新的本地忽略目录后再填：

```powershell
$intakeStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$intakeDir = Join-Path 'runs' ('validation-intake-' + $intakeStamp)
New-Item -ItemType Directory -Path $intakeDir -ErrorAction Stop
Copy-Item -LiteralPath 'tools/oven_training/recipes/physical-validation-objects.csv','tools/oven_training/recipes/physical-validation-sessions.csv','tools/oven_training/recipes/physical-validation-photos.csv' -Destination $intakeDir
```

可用表格软件填写，保存UTF-8 CSV；不要把填好的原图路径、设备序列标识、人员信息或表格提交Git。使用组内可追溯代号，真实设备敏感信息在私下登记。

## 现场最少四步

1. **先摆实物、再编号。** 采集者凭采购/制作/交接记录填写objects，保持ID长期不变；同一土豆切块、同一玉米切段及相同母体关联到同一parent_source_group_id。批次与证据不知道就unknown；不要由图片序号、目录或模型生成实物身份。类别依据实际原料/人工观察，无法确认则保留候选，trusted_label空或false。
2. **开一次会话，填一次sessions。** 按实际烤箱、相机、时间和照明登记。换时间/设备/来源导致新会话时新增行，不能为拆集合而随意改会话。一次会话拍多个实物也共享同一session_id，后续不得跨集合。
3. **每张图填photos九列。** 同一实物ID和会话ID可以填充复制；路径须对应实际文件，层位按设备现场描述，foil填true/false/unknown；background记录实际托盘/锡纸等，angle记实际视角，不确定填unknown。照明不变时继承session的lighting，变化时明确override。不从AI预测或未经验证的图像外观反推层位/来源身份。
4. **另一名组员看原图和来源依据后签字。** physical_identity_verified、session_identity_verified、trusted_label只有实际审核后才显式填true，附对应reviewer和带时区时间。空、false或unknown都不表示已通过；没有证据不能为了程序绿灯补签。人工修正候选在本地新表留痕，不覆盖原图、旧JSON或旧训练清单。

类序仍为0 asparagus、1 bacon、2 broccoli、3 chicken_wing、4 croissant、5 french_fries、6 potato、7 salmon、8 toast、9 corn、10 sausage、11 steak。不要用MVP八类的ID解释十二类；类别含义不确定先留待团队确认。

## 数据负责人合并登记：没有自动晋级

这三表是**采集附表，不能直接作为controlled --rows或--validation-rows输入**；当前没有新增自动转换器。每次交接由数据负责人完成并保留join记录：

1. 检查路径唯一且照片存在、可解码；每个实物/会话ID必须在对应表存在且唯一，冲突或缺失退回。混合主食材图暂不进入单标签池。技术侧复用既有只读 `scripts.audit_images.image_facts` 计算文件SHA/像素SHA与尺寸；不让采集者手抄哈希，不复制/修改原图。
2. 按真实实物、共同母体/来源组、拍摄会话及已知重复关系的**传递连接分量**分组，再统一分配train/val/封存test；不是每张图自行决定集合。planned_split仅为预定用途，若冲突则BLOCKED，不能删关联ID掩盖。相同会话拍了多件食物时需整个会话留在同一集合，因此应提前分开拍验证和封存测试。
3. 依据已审核class_key用固定12类映射派生class_id；合并实物/会话审核签名到保留的 `physical-validation-intake.csv` 技术字段。objects/sessions的identity审核者/时间不同时，在身份审核记录中保留两者及证据，由最终审核者在技术表签字；不将一个审核者默默替代另一个。技术表的physical_identity_verified、session_identity_verified、trusted_label只接受true/false/空；附表unknown保留为原证据，交接技术表时按false/空并保持BLOCKED，不能直接抄入unknown或自动变true。
4. relative_path/source遵循既有入口实际支持的根目录规则；新照片不是source=new的任意路径通配符。继承原origin/license（未知就unknown，不得公开分发）。将shelf、foil、background、angle、lighting、oven_id、camera_id及source_batch_id作为附加场景列保留；照片override优先于会话默认。生成到新忽略目录，正式传入前逐项核对SHA及class_id。
5. 对技术登记运行现有资格预检。程序检查的是声明字段、重复和隔离，不会替代来源调查；任何READY仍不自动冻结数据或证明泛化。旧train缺真实身份时依然BLOCKED，不能把新表空白值自动变true。封存test不得送训练入口、用于挑Epoch或BN策略。

现有 `tools.oven_capture` 的review/export使用冻结MVP八类配置，**不能将本十二类表强塞给该接口或修改共享classes.json绕过限制**。本流程复用其哈希/元数据原则和现有采集软件，不改相机兼容、防覆盖、UI或生产模型。若将来需要十二类自动交接，另行协调数据/推理负责人，增加受测转换器；当前不宣称已自动接通。

## 交接时应观察到的结果

- 每张照片都能反查一个真实实物及会话、层位/背景；不知道的项明确unknown。
- 照片数、真实实物数、共同母体/来源组数、会话数分开统计；换角度/层位不会增加独立实物数。
- 任何跨集合实物/母体/会话或重复关系冲突都明确BLOCKED。新身份表不追溯虚构旧训练实物。
- 标签仍由人工来源依据确认；不因为模型预测一致就打勾。训练池来源证据补齐前，不承诺存在合格独立验证集。
