# 数据下载与交接说明

日期：2026-10-19。当前未下载图片；请保留平台原始下载包，不手动混合类别、删除目录或修改文件名。

## 1. Kaggle

打开 https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset ，通过页面下载完整 ZIP。若要求登录，请自行在浏览器登录，不需要提供账号、密码或令牌。

将压缩包放在项目的 `data/raw/kaggle_vegetable/` 下，保留平台生成的原文件名。无需自行解压、选取类别或沿用其 train/test/validation 划分。随包许可、README 和引用文件一并保留。

## 2. Roboflow

打开 https://universe.roboflow.com/ece-capstone-fridgegenie/fridge-genie ，进入可用的 Dataset / 版本页面，选择下载或导出。

- 优先选择 **COCO JSON**：需要同时包含图片、检测框及类别信息。不要只下载图片或只下载模型。
- 选择 ZIP 下载到本机。若要求登录，由你在浏览器完成；无需发送下载代码或 API Key。
- 将完整 ZIP 放入项目 `data/raw/roboflow_fridge_genie/`。保留全部 train/valid/test、标注 JSON、README 与许可文件。
- 记录实际版本号、页面标注的增强及预处理设置。已有版本若含增强，不自行重建或额外增强，下载后检查父图关联。
- 若没有 COCO JSON，可选包含图片、labels 和 data.yaml 的 YOLO 检测格式，告诉我选择的格式即可。
- 若没有下载权限、只有付费选项或没有包含图片的导出，告知页面提示文字；不必购买、创建项目或公开上传图片。

官方说明：https://docs.roboflow.com/datasets/versions/dataset-versions/exporting-data.md 。支持 COCO 和 ZIP 下载；下载代码可能包含私有密钥。导出可能压缩图片和规范化标签，不能把导出字节视为原图字节。

## 3. 交接

本地项目为 `D:\烤箱数据识别\het-foodai`。`data/raw/` 已被 Git 忽略。

```text
het-foodai/
  data/raw/
    kaggle_vegetable/       # 完整 ZIP
    roboflow_fridge_genie/  # 完整 ZIP + 版本说明（如有）
```

如果放在其他目录，告知两个 ZIP 的完整路径即可，不必在聊天上传大型文件。接入前检查包完整性、类别、数量和许可；解压时校验路径，保留原包。

## 4. 到位后的顺序

核对版本与许可 → 完整性检查与计数 → 标签映射与抽查 → 原图去重和分组 → 框审核及裁剪 → 裁剪近重复复查 → 分组划分草案 → 质量报告 → 共同确认类别和划分 → 生成冻结清单并校验。

比例暂议 80/10/10；不在图片未审核时生成虚构 manifest，不用测试集调参。鲈鱼来源和对虾语义未解决时明确报告缺口，不用泛 fish/shrimp 填满八类。
