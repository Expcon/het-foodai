# 证据记录 — UI / Prediction Adapter

- 日期：2026-10-19
- 环境：Windows · Python 3.12.4 (Anaconda) · 未安装 torch / gradio
- 说明：以下全部为**实际执行输出**，未修改、未美化。命令中的 `D:\anaconda3\python.exe` 为本机解释器路径。

---

## E1 单元测试

命令：
```
D:\anaconda3\python.exe -m unittest discover -s tests -v
```

输出：
```
test_contract_error_banner_disables_button (test_app_logic.TestBanners.test_contract_error_banner_disables_button) ... ok
test_empty_upload_is_neutral (test_app_logic.TestBanners.test_empty_upload_is_neutral) ... ok
test_not_ready_banner_disables_button (test_app_logic.TestBanners.test_not_ready_banner_disables_button) ... ok
test_ready_banner_enables_button (test_app_logic.TestBanners.test_ready_banner_enables_button) ... ok
test_failure_returns_no_scores (test_app_logic.TestResults.test_failure_returns_no_scores) ... ok
test_latency_states_scope (test_app_logic.TestResults.test_latency_states_scope) ... ok
test_no_accuracy_wording_anywhere (test_app_logic.TestResults.test_no_accuracy_wording_anywhere) ... ok
test_top3_rows (test_app_logic.TestResults.test_top3_rows) ... ok
test_class_hash_prefix_stable (test_contracts.TestContracts.test_class_hash_prefix_stable) ... ok
test_clean_split (test_contracts.TestContracts.test_clean_split) ... ok
test_draft_blocked (test_contracts.TestContracts.test_draft_blocked) ... ok
test_exact_duplicate_leakage_rejected (test_contracts.TestContracts.test_exact_duplicate_leakage_rejected) ... ok
test_group_leakage_rejected (test_contracts.TestContracts.test_group_leakage_rejected) ... ok
test_missing_model_is_explicit (test_contracts.TestContracts.test_missing_model_is_explicit) ... ok
test_path_escape_rejected (test_contracts.TestContracts.test_path_escape_rejected) ... ok
test_wrong_label_mapping_rejected (test_contracts.TestContracts.test_wrong_label_mapping_rejected) ... ok
test_checkpoint_sha_prefix_match (test_predictor.TestContractChecks.test_checkpoint_sha_prefix_match) ... ok
test_checkpoint_sha_prefix_mismatch (test_predictor.TestContractChecks.test_checkpoint_sha_prefix_mismatch) ... ok
test_class_list_hash_mismatch_rejected (test_predictor.TestContractChecks.test_class_list_hash_mismatch_rejected) ... ok
test_missing_metadata_is_contract_error (test_predictor.TestContractChecks.test_missing_metadata_is_contract_error) ... ok
test_prefix_of_frozen_list_is_ready (test_predictor.TestContractChecks.test_prefix_of_frozen_list_is_ready)
8 类配置下，3 类 checkpoint 属于合法前缀，应就绪。 ... ok
test_wrong_class_order_rejected (test_predictor.TestContractChecks.test_wrong_class_order_rejected) ... ok
test_wrong_display_name_rejected (test_predictor.TestContractChecks.test_wrong_display_name_rejected) ... ok
test_dict_matches_contract_fields (test_predictor.TestContractShape.test_dict_matches_contract_fields) ... ok
test_empty_file_rejected (test_predictor.TestImageValidation.test_empty_file_rejected) ... ok
test_real_png_accepted (test_predictor.TestImageValidation.test_real_png_accepted) ... ok
test_text_file_named_jpg_rejected (test_predictor.TestImageValidation.test_text_file_named_jpg_rejected) ... ok
test_unsupported_suffix_rejected (test_predictor.TestImageValidation.test_unsupported_suffix_rejected) ... ok
test_draft_classes_are_rejected (test_predictor.TestNotReadyStates.test_draft_classes_are_rejected) ... ok
test_missing_checkpoint_mentions_path (test_predictor.TestNotReadyStates.test_missing_checkpoint_mentions_path) ... ok
test_missing_classes_config_is_not_ready (test_predictor.TestNotReadyStates.test_missing_classes_config_is_not_ready) ... ok
test_unconfigured_message_is_identical_to_contract (test_predictor.TestNotReadyStates.test_unconfigured_message_is_identical_to_contract) ... ok

----------------------------------------------------------------------
Ran 32 tests in 0.247s

OK
```

---

## E2 未配置模型（默认启动，--check 模式）

命令：
```
D:\anaconda3\python.exe app\app.py --check
```

输出：
```
{
  "state": "not_ready",
  "message": "模型尚未训练／未加载。此处没有模拟识别分数。",
  "checkpoint": null,
  "model_id": null,
  "num_classes": null,
  "input_size": null,
  "class_list_sha256": null,
  "checkpoint_sha256": null,
  "device": null
}
```

退出码：1（非 ready）

---

## E3 checkpoint 路径不存在

命令：
```
D:\anaconda3\python.exe app\app.py --check --checkpoint runs\stage8-run001\best.pt
```

输出：
```
{
  "state": "not_ready",
  "message": "模型尚未训练／未加载。此处没有模拟识别分数。\n\ncheckpoint 不存在：runs\\stage8-run001\\best.pt\n请让模型 owner 提供 runs/<run_id>/best.pt，再用 --checkpoint 指定。",
  "checkpoint": "runs\\stage8-run001\\best.pt",
  "model_id": null,
  "num_classes": null,
  "input_size": null,
  "class_list_sha256": null,
  "checkpoint_sha256": null,
  "device": null
}
```

退出码：1（非 ready）

---

## E4 只有 draft 类别配置

命令：
```
D:\anaconda3\python.exe app\app.py --check --checkpoint work\dummy\best.pt --classes configs\classes.draft.json
```

输出：
```
{
  "state": "not_ready",
  "message": "模型尚未训练／未加载。此处没有模拟识别分数。\n\n类别配置不可用：configs\\classes.draft.json\nClass list is DRAFT; obtain team approval before formal training\n需要数据 owner 提交并冻结 configs/classes.json（status=frozen）后才能推理。",
  "checkpoint": "work\\dummy\\best.pt",
  "model_id": null,
  "num_classes": null,
  "input_size": null,
  "class_list_sha256": null,
  "checkpoint_sha256": null,
  "device": null
}
```

退出码：1（非 ready）

---

## E5 损坏图片（文本文件改名 .jpg）

命令：
```
D:\anaconda3\python.exe -c "from foodai.app_logic import image_banner; print(image_banner('work/fake.jpg')[0])"
```

输出：
```
⛔ **输入不是有效图片**

无法解码为图片：fake.jpg（文件可能已损坏或扩展名与实际格式不符）

> 请重新导出为 JPG 或 PNG 后再试。
```

---

## E6 0 字节图片

命令：
```
D:\anaconda3\python.exe -c "from foodai.app_logic import image_banner; print(image_banner('work/empty.png')[0])"
```

输出：
```
⛔ **输入不是有效图片**

文件为空（0 字节）：empty.png

> 请重新导出为 JPG 或 PNG 后再试。
```

---

## E7 未就绪状态下点击识别

命令：
```
D:\anaconda3\python.exe -c "from foodai.app_logic import run_prediction; from foodai.predictor import PredictionAdapter; print(run_prediction(PredictionAdapter(), 'work/fake.jpg')[0])"
```

输出：
```
⛔ **无法完成识别**

`ModelNotReadyError`: 模型尚未训练／未加载。此处没有模拟识别分数。

> 没有输出任何预测分数。
```

关键点：错误分支返回的表格为空、时延为空串，**没有任何分数**。

---

## 尚未完成的证据（必须等真实 checkpoint）

| 用例 | 状态 | 阻塞原因 |
|---|---|---|
| T1 正常输入 → Top-3 + 时延 | **PENDING** | 仓库内没有 best.pt；configs/classes.json 尚未冻结 |

模型 owner 交付后补跑：

```
D:\anaconda3\python.exe app\app.py --check --checkpoint runs\<run_id>\best.pt
D:\anaconda3\python.exe app\app.py --checkpoint runs\<run_id>\best.pt
```

需要记录：Top-3 表格内容、latency_ms、checkpoint_sha256 前 16 位、界面截图。
**在此之前，不得把任何数字写成识别效果或准确率。**

