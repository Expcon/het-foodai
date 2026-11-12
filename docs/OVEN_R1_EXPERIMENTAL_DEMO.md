# R1错误分组与可切换实验演示版

2026-11-11，基于 `793e8ec`，分支 `feat/model-oven-r1-experimental-demo`。本次没有训练v3、调整权重、改标或改变生产默认模型。完成43张剩余错误的实际视觉分组、安全R1 ONNX导出、真实图片对齐、现有GUI实验列表和独立Windows打包。

**R1只用于可切换实验演示，不是独立泛化评估通过的部署模型。** 当前144张已暴露开发诊断图及11个采集者陈述的实物/母体组，仍缺采购/会话与训练来源独立性证据；不是独立test。参见[原对照](OVEN_SUPPLEMENT_V2_R1_COMPARISON.md)和[采集者陈述附录](OVEN_SUPPLEMENT_COLLECTOR_ADDENDUM.md)。来源许可/公开再分发授权仍unknown，所有照片、逐图路径、模型文件和交付包仅保留本机忽略目录，不上传。

## 1. 43张实际视觉复核与分类联系表

实际查看原43张完整画面联系表全部3页，并额外查看唯一土豆错误原图1920×1080；新生成**8组10页**清晰全帧联系表，覆盖43张各一次。分组依据分别保留：食材为原目录候选与采集者陈述，背景为既有预测前AI视觉观察，玉米完整/切段为采集者关系+视觉，其他形态为本次视觉观察。没有用模型预测确认真值，没有覆盖标签。

| 候选类别 | 背景 | 外观 | 不一致图像数 | 涉及采集者母体组 | R1输出 |
| --- | --- | --- | ---: | ---: | --- |
| corn | 黑盘 | 白玉米半段 | 11 | 1 | chicken_wing 11 |
| corn | 黑盘 | 完整白玉米 | 3 | 1 | toast 2、chicken_wing 1 |
| corn | 锡纸 | 白玉米半段 | 7 | 1 | chicken_wing 4、toast 3 |
| potato | 锡纸 | 带皮、近圆完整外观 | 1 | 1 | croissant 1 |
| salmon | 黑盘 | 橙色条状鱼肉切块 | 2 | 2 | bacon 1、croissant 1 |
| salmon | 锡纸 | 橙色条状鱼肉切块 | 5 | 2 | potato 3、bacon 1、corn 1 |
| sausage | 黑盘 | 浅色短圆柱 | 1 | 1 | chicken_wing 1 |
| sausage | 锡纸 | 浅色短圆柱 | 13 | 3 | chicken_wing 13 |

各行母体数有重叠，不能相加；43张共涉及8个陈述分组，**经核验的独立来源数仍unknown**。玉米两半共用同一parent，表中18张半段错误只有一个母穗来源。土豆隐蔽切面、香肠完整/切断状态不能从这些视角可靠判定，保留unknown；不能为凑完整/切段分组而强行补信息。层位、采购批次、会话仍未核实。

直接证据与限制：

- 43张中30张输出chicken_wing（玉米16、香肠14），集中于浅色食物；这是稳定的诊断混淆方向。浅色纹理/轮廓可能相关，但没有输入干预或独立验证来证明因果。
- 白玉米完整组3/12图不一致，两个半段合并18/24不一致；两半来自同一穗，完整样本来自另一穗，**形态与母体混杂**，不能宣称“切段造成下降”。完整玉米锡纸6张均与候选一致，黑盘3/6不一致；切段黑盘11/12、锡纸7/12不一致。这些只描述已暴露诊断数据。
- 香肠错误为锡纸13/18、黑盘1/18，三个陈述实物均涉及；图中可见浅色表面、裂纹及锡纸高光。三文鱼也有锡纸5/18、黑盘2/18不一致，但白玉米黑盘问题更大，不能把所有错误归为锡纸。角度、曝光、实物和尺度尚未受控。
- 唯一土豆图为褐色带皮近圆外观，背景有明显镜像反射；R1 croissant约0.564、potato约0.423。这张保留为待复核反例，不根据预测改成可颂。反射/小尺度是否主因未知。
- 视觉上未发现足以支持批量改标的证据；原料身份最终仍依赖采集者/人工来源记录。颜色、食材在224输入中的小尺度和背景高光是下一轮应受控测量的假设，不是已确定原因。

本机 `runs/oven-r1-demo-20261111/classified-errors/contacts.html` 为联系表入口；`classified_errors.json`保留43张原路径、SHA、候选、v2/R1原预测、Top-3/全部概率、母体、背景/外观及各自证据；`group_summary.json`提供完整索引。未复制原图作为新数据集，未重新计算食材准确率。

## 2. 安全导出与真实图片对齐

直接复用原 `tools.oven_training.export_desk`，没有使用Ultralytics对象pickle加载R1。安全加载为 `torch.load(weights_only=True)`，固定源文件SHA、metadata与strict state_dict校验，模型eval且真实前向验证。R1为最终第30轮，未按本批分数选择Epoch。

| 文件/契约 | SHA256 |
| --- | --- |
| R1 `runs/oven-r1-20261111/R1/training/last.pt` | `fec9b89f29e65dc5f35976e704ebce9900a9c34a9b1c231f5adfdd93b02813cd` |
| 新R1 ONNX | `ca039fbbaadda787039a83a6ed90cb4249760b705243574c2f086746398e7b7e` |
| 新R1 classes.json | `f08d024f42e4959682dd7968798f1554c992b27d38a12c7ce74989f917760114` |
| 保留v2 ONNX | `8c696ce5dc63a25053198f1781504b4a53cb801f2d971e3bceeba52eed789689` |
| 十二类有序列表 | `eeb5260a718f2a1bd42a4d600c8cb2e5ee290c98a5a24e38aa1822e7391827f7` |

ID保持0–11：asparagus、bacon、broccoli、chicken_wing、croissant、french_fries、potato、salmon、toast、corn、sausage、steak。白玉米仍为corn/9，不修改旧八类classes.json或任何训练清单。

静态FP32、opset17、6,202,899字节，输入`images [1,3,224,224]`，输出`probabilities [1,12]`，输出已含softmax。沿用EXIF转正、RGB、Pillow双线性contain、居中补114、float32除255、NCHW。训练端与Desk端文件解码/预处理逐元素核对，不增加中心裁剪或二次softmax。验收门槛沿用所有排序Top-3一致、最大绝对概率误差≤1e-4且概率合法。

| 实际核对 | 张数/记录 | Top-1/排序Top-3一致 | 预处理完全相同 | 最大概率绝对误差 |
| --- | ---: | --- | --- | ---: |
| PyTorch CPU FP32→ORT CPU，checkpoint绑定的原导出清单 | 1,997 | 全部 | 同一输入 | 2.562999725341797e-6 |
| 独立Desk依赖环境，完整文件解码→ORT | 1,997 | 全部 | 1,997 | 2.562999725341797e-6 |
| 本批真实诊断图，PyTorch→完整Desk文件推理 | 144 | 全部 | 144 | 3.933906555175781e-6 |
| 实际EXE依赖环境重查上述144张 | 144 | 全部 | 144 | 3.933906555175781e-6 |

1,997条导出记录包括1条原排除重复，不是1,997个独立训练样本，更不是独立测试。所有数字只表示推理格式一致性。原R1持久张量前后指纹相同，原图SHA前后相同。没有修改、量化、再训练模型；没有开展MNN/INT8/RK3506验收。

项目环境为原Python3.12.10 / Torch2.14.1+cu130 / Ultralytics8.3.230 / ONNX1.19.1 / ORT1.23.2 / Pillow12.3.0；独立打包环境为既有NumPy2.2.6 / Pillow12.0.0 / OpenCV4.12.0.88 / ORT1.23.2，无Torch。没有pip安装或改全局Python/CUDA。独立144重查首次误用了间接导入Torch的观测写入助手，退出1；改用本地标准库写新报告后退出0，失败日志保留，未给桌面环境安装Torch。

## 3. 现有GUI实验模型列表

保留现有 `tools/oven_desk/` GUI，只增加可选本地 `experimental_models.json` 列表，和源码可用的 `--model-catalog` 参数。列表文件/模型位于新演示目录，**不覆盖旧交付包或用户配置**。启动仍是手动选择、模型未加载；没有自动选择R1或改变旧默认。v2保留为列表首项，可随时切回。

选择条目只设置匹配的ONNX、classes.json、十二类模式和224，并清除ready；用户点击“载入真实模型”后，哈希、类别/预处理契约、真实前向和checkpoint身份核对成功才ready。忙时拒绝切换；手工改了已选条目的路径/模式会拒绝加载，需重新选择或切回手动。模型线程再次核对选定checkpoint，避免选择后加载另一个实验。

目录路径必须相对本地交付根，拒绝绝对路径、越界、重复身份/标签和不支持的格式。文件SHA用于发现错配，不是数字签名。当前仍为本机单用户实验工具，不宣称对恶意文件替换提供完整安全边界。

相机兼容 `camera_devices.py`、拍照/录像防覆盖 `capture_core.py` 字节完全不变；采集线程、相机打开、拍照回调及用户设置保存逻辑均未修改。当前帧使用原BGR解码帧→RGB完整预处理。全部现场功能仍需用户自行连接并确认设备，测试没有打开硬件。

## 4. 实际测试与交付边界

| 命令/路径 | 实际结果 |
| --- | --- |
| `.venv/Scripts/python.exe -m unittest discover -s tests -v` | 最终退出0；225项，224通过、1项Windows软链接权限跳过、0失败，23.426秒；日志`unittest_final.log`。此前并行运行一次同样通过，61.300秒，不作性能比较 |
| Desk专项单测 | 49项通过；新增7项合成catalog/状态/竞态测试只证明软件边界 |
| `gui_catalog_smoke.py`，真实Tk/ORT工作线程 | 退出0；启动不加载、v2→R1→v2加载及真实文件预测、忙时拒绝、无帧拒绝、缺模型清ready；文件注入，未连相机 |
| 既有 `gui_model_smoke.py --mode oven_round2`，R1 | 退出0；实际PNG/JPG图集2成功+损坏图1失败、空目录拒绝、失败重载等8项通过；JPG由真实PNG派生，不是新增独立采样 |
| PyInstaller独立构建 | 退出0，未安装依赖；新EXE路径独立，不改旧EXE |
| 新EXE `--healthcheck` | 退出0；Tk/ORT/照片保存编号自检通过；MNN未打包，未作为本轮目标 |
| 新EXE `--model-selfcheck` | R1 PNG、R1派生JPG、v2 PNG均退出0、ready=true；坏图/错配classes/缺模型均按预期退出1、ready=false |
| 原生窗口视觉点击验收 | **BLOCKED**：computer-use能定位新EXE窗口，但工具无法绑定/恢复最小化窗口；没有取得可复核截图，不冒充EXE人工点击已验收 |
| 保全 | 144张来源图片及199项历史保护文件SHA不变，含旧审计JSON/预测、权重、ONNX/EXE及采集核心；真实用户settings.json未改变 |

原生窗口限制不替代已通过的真实Tk工作线程测试与EXE模型自检；用户仍需在实际桌面还原窗口完成点击检查。真实相机、用户现场设备的采集/识别、独立食材泛化均未验收。程序对齐通过不意味着已解决43张错误。

本机新交付：`runs/oven-r1-demo-20261111/delivery/`，含EXE、外部实验列表、v2/R1各自ONNX+classes.json、使用说明、SHA256SUMS；不含原图、训练pt或个人设置。

- EXE：`HeT-OvenCapture-v1.2.1-r1-experimental.exe`，SHA `e4810a5daf84020d2214a0e346ed9bdf090a5deb12628ed7c2ebc4fdcf0dc9fe`。
- ZIP：`runs/oven-r1-demo-20261111/HeT-OvenCapture-v1.2.1-R1-experimental-v2-R1.zip`，94,009,673字节，SHA `6e4dd977c52bf14da083217f56f68b41edc0f95af4e8189d724e418b9ec682ba`，CRC及文件哈希检查通过；没有上传或发布Release。
- 全量逐图概率/张量SHA：`onnx/onnx_alignment.json`、`desk_alignment.json`、`supplement_alignment.json`、`supplement_desk_alignment.json`。
- 实际GUI及EXE记录：`gui_catalog.json`、`gui_R1.json`、`exe-checks/`；全部原始日志/退出码保留，含失败记录。

## 5. 当前怎么演示

1. 双击新delivery目录EXE，或将ZIP整体解压到新目录再运行；如窗口最小化，从任务栏还原。旧v2演示包保持原样。
2. 进入“模型 / 图集”，在“实验模型”列表选 **R1（低学习率·实验12类）**，再点“载入真实模型”。应显示已加载、12类、224×224、实验预测/无独立验证；选择条目本身不代表ready。
3. “图集测试”选择**直接包含JPG/PNG**的目录（当前不递归子目录）；预测CSV保存到“相机 / 采集”页的保存目录。测试文件夹可选补充数据的某个12张子目录；无真值只显示Top-3及分数，不能叫准确率。
4. 需要对照时等待任务结束，列表切回 **v2（原版·12类）**，再加载。不要交换两个目录的classes.json；保持整个交付目录结构。
5. 当前画面识别由现场人员按原流程自行连接相机、确认预览后点击。此次没有用硬件实测，不自动控制加热或驱动；无当前帧应拒绝预测。

源码运行可用 `python tools/oven_desk/oven_capture_gui.py --model-catalog <新delivery目录>/experimental_models.json`，使用既有隔离Desk环境；没有新列表时继续原手动模式。源码重新构建可用原 `build_windows.ps1 -SkipMNN`，但本轮实际复用已有隔离环境直接PyInstaller，参数留档于本机 `executed_build_command.ps1`、输出见`build.log`；留档脚本拒绝覆盖本次构建，重新打包必须另选输出目录，避免覆盖旧证据。

## 6. 下一轮最优先补的数据

1. **白玉米优先。** 采购/来源可核实的新母穗，记录白/黄外观、已知品种和状态；对同一穗先完整后切段拍成对图，始终共享parent且不跨集合。现有半段错误18张只有一根母穗，不能通过继续连拍该实物补足独立覆盖。
2. **浅色香肠优先。** 新实物及可追溯制作/采购批次，记录实际原料、颜色、表面/裂纹、生熟或加工状态；人工确认标签。对同一实物同一位置、角度和照明，成对拍黑盘/锡纸，先只改背景，避免同时改变层位、曝光和姿态。
3. **三文鱼与土豆反例。** 新来源鱼肉记录共同母体，覆盖厚薄/条块与纹理方向；新土豆覆盖带皮近圆外观及反光场景。记录实际层位与会话，再分别改变尺度或照明。可在独立标注副本中画食材框测量224输入中的目标尺寸，不改原图或本轮推理预处理。
4. **先定组和用途再采。** 保持实物/母体/会话关联；开发验证和封存最终test分开采集、登记并隔离。可沿用每类3个新上游独立实物、2开发+1封存、每实物3层位×2背景的最小试采，仍需补训练池来源独立性证据，且少量实物不足以支持宽泛结论。其余八类也需独立覆盖，不能只展示改善的类别。

没有因这次ONNX/EXE对齐而解锁v3训练或生产替换。下一步应补来源/会话和受控场景数据，再评价R1；不把训练图回看或当前144张诊断当作独立测试。

## 修改文件

- `tools/oven_desk/model_catalog.py`：可选本地模型列表、目录约束及checkpoint身份核对。
- `tools/oven_desk/oven_capture_gui.py`：现有页面增加显式实验选择、ready/忙时保护及可选CLI入口，版本1.2.1；不动相机/拍照流程。
- `tools/oven_desk/build_windows.ps1`：仅更新新实验版本号。
- `tools/oven_desk/tests/test_model_catalog.py`、`gui_catalog_smoke.py`：合成边界与真实文件Tk切换测试。
- `tests/smoke_oven_r1_supplement_alignment.py`：可复现的真实诊断图片PT/ORT/Desk逐图输入/输出对齐。
- 本报告。联系表生成驱动、模型、原始日志和逐图结果均位于新的忽略run，不提交私人路径或企业照片。
