# MVTec AD Bottle 缺陷检测正规流程计划

## 1. 当前数据状态

已下载并解压数据：

```text
data/
├── bottle.tar.xz
└── bottle/
    ├── ground_truth/
    ├── test/
    ├── train/
    ├── license.txt
    └── readme.txt
```

当前 `bottle` 类别数据统计：

| Split | Type | Count |
| --- | --- | ---: |
| train | good | 209 |
| test | good | 20 |
| test | broken_large | 20 |
| test | broken_small | 22 |
| test | contamination | 21 |
| ground_truth | broken_large mask | 20 |
| ground_truth | broken_small mask | 22 |
| ground_truth | contamination mask | 21 |

结论：

- 数据目录结构符合 MVTec AD 官方格式。
- 缺陷测试图像共 `63` 张。
- 缺陷 mask 共 `63` 张，与缺陷测试图像数量一致。
- 可以继续进入数据检查、基线建模和评估阶段。

## 2. 正规项目路线

MVTec AD 原始设定更偏向异常检测：

- 训练集只有正常样本。
- 测试集包含正常样本和异常样本。
- 异常样本提供像素级 `ground_truth` mask。

因此正规项目建议分成两个阶段：

### 阶段 A：先做可展示的二分类 MVP

目标：

- 快速跑通工业缺陷检测项目闭环。
- 输出 Accuracy、Precision、Recall、F1-score、Confusion Matrix。
- 生成预测正确和预测错误样本图。
- 适合 GitHub、简历、面试讲解。

注意：

- 这是 supervised binary classification MVP。
- 由于 MVTec 官方训练集没有缺陷样本，需要从 `test` 中划分一部分缺陷样本作为训练/验证数据。
- README 中需要明确说明该实验是为了演示工业二分类检测流程，不等同于 MVTec 官方 anomaly detection benchmark。

### 阶段 B：后续升级为更正规的异常检测

目标：

- 只用 `train/good` 训练正常模式。
- 在 `test/good` 和 `test/defect` 上判断异常。
- 使用 mask 做异常定位或像素级评估。

可选方法：

- AutoEncoder
- PaDiM
- PatchCore
- FastFlow
- EfficientAD

阶段 B 不是当前最小可行版本的重点，可以作为后续增强方向。

## 3. 数据检查流程

在写训练代码前，先实现或手动完成以下检查。

### 3.1 目录结构检查

确认以下目录存在：

```text
data/bottle/train/good/
data/bottle/test/good/
data/bottle/test/broken_large/
data/bottle/test/broken_small/
data/bottle/test/contamination/
data/bottle/ground_truth/broken_large/
data/bottle/ground_truth/broken_small/
data/bottle/ground_truth/contamination/
```

检查结果需要输出：

- 每个目录是否存在。
- 每个目录图片数量。
- 是否存在空目录。

### 3.2 图片文件检查

检查内容：

- 文件扩展名是否为 `.png`。
- 图片是否能被 PIL / OpenCV 正常读取。
- 是否存在损坏图片。
- 图片通道数是否符合预期。
- 图片尺寸是否一致。

建议输出：

```text
image_path, split, defect_type, width, height, channels, readable
```

保存为：

```text
results/data_check.csv
```

### 3.3 mask 对齐检查

仅对缺陷样本检查：

- `test/broken_large/000.png` 是否对应 `ground_truth/broken_large/000_mask.png`
- `test/broken_small/000.png` 是否对应 `ground_truth/broken_small/000_mask.png`
- `test/contamination/000.png` 是否对应 `ground_truth/contamination/000_mask.png`

检查内容：

- 每张缺陷图是否有对应 mask。
- 每张 mask 是否有对应缺陷图。
- mask 是否能正常读取。
- mask 尺寸是否和原图一致。

虽然二分类 MVP 暂时不使用 mask，但正规项目中必须检查并保留该信息，方便后续做异常定位。

### 3.4 标签规则检查

二分类标签：

| Folder | Label Name | Label ID |
| --- | --- | ---: |
| good | good | 0 |
| broken_large | defect | 1 |
| broken_small | defect | 1 |
| contamination | defect | 1 |

同时保留原始缺陷类型：

```text
defect_type = good / broken_large / broken_small / contamination
```

这样后续可以分析模型在哪一种缺陷上表现较差。

### 3.5 数据泄漏检查

需要避免：

- 同一张图片同时进入训练集和测试集。
- 从 `test` 划分训练缺陷样本后，又在最终测试集中重复使用。
- 数据增强后的图片被当作独立原图进入测试集。

建议：

- 先生成一个固定的 metadata 文件。
- 用随机种子固定划分。
- 所有训练、验证、测试都从 metadata 读取。

推荐保存：

```text
results/metadata.csv
```

字段：

```text
image_path, mask_path, original_split, defect_type, binary_label, experiment_split
```

## 4. 数据划分方案

因为官方 `train` 中只有正常样本，二分类 MVP 推荐这样划分：

### 4.1 训练集

来源：

- `train/good` 中的大部分正常样本。
- `test` 中一部分缺陷样本。

建议：

- 正常样本：从 `train/good` 中取约 80%。
- 缺陷样本：从每个缺陷类型中取约 60%。

### 4.2 验证集

来源：

- `train/good` 中剩余一部分正常样本。
- `test` 中每个缺陷类型剩余一部分缺陷样本。

用途：

- 选择最佳模型。
- 观察过拟合。
- 调整 threshold 或超参数。

### 4.3 测试集

来源：

- `test/good`。
- 每个缺陷类型中未进入训练/验证的样本。

用途：

- 只在最终评估时使用。
- 输出最终指标和图像结果。

### 4.4 固定随机种子

建议使用：

```text
seed = 42
```

所有划分结果写入 `metadata.csv`，保证结果可复现。

## 5. 实现步骤

### Step 1：创建数据检查脚本

新增：

```text
src/check_data.py
```

功能：

- 扫描 `data/bottle`。
- 输出类别数量。
- 检查图片可读性。
- 检查 mask 对齐。
- 生成 `results/data_check.csv`。
- 生成 `results/metadata.csv`。

运行：

```bash
python src/check_data.py --data_dir data/bottle
```

### Step 2：创建 Dataset

新增：

```text
src/dataset.py
```

功能：

- 从 `results/metadata.csv` 读取数据。
- 根据 `experiment_split` 加载 train/val/test。
- resize 到 `224 x 224`。
- 输出 `image, label, metadata`。

预处理：

- PIL 读取 RGB。
- Resize。
- ToTensor。
- ImageNet Normalize。

训练增强：

- RandomHorizontalFlip
- RandomRotation
- ColorJitter

测试增强：

- 只做 Resize 和 Normalize。

### Step 3：训练 ResNet18

新增：

```text
src/train.py
```

模型：

- `torchvision.models.resnet18`
- 使用 ImageNet 预训练权重。
- 替换最后一层为 2 分类。

配置：

```text
image_size = 224
batch_size = 16 或 32
epochs = 10
learning_rate = 1e-4
optimizer = Adam
loss = CrossEntropyLoss
```

输出：

```text
results/best_model.pth
results/train_log.csv
```

### Step 4：评估模型

新增：

```text
src/evaluate.py
```

输出指标：

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- per-defect-type recall

保存：

```text
results/metrics.json
results/confusion_matrix.png
```

### Step 5：可视化结果

新增：

```text
src/visualize.py
```

输出：

```text
results/sample_predictions.png
results/misclassified_samples.png
```

图片上展示：

- true label
- predicted label
- confidence
- defect type

如果预测错误样本为空，也要生成说明图或在 README 中说明。

### Step 6：整理 README

README 包含：

- 项目背景。
- 数据集说明。
- 数据检查结果。
- 二分类标签定义。
- 数据划分方式。
- 模型结构。
- 运行步骤。
- 实验结果。
- 可视化图片。
- 工业落地价值。
- 局限性与下一步。

重点写清楚：

- MVTec AD 官方训练集只有正常图。
- 本项目第一版为了展示二分类检测流程，对测试缺陷样本进行了固定划分。
- 后续可升级为真正的异常检测或异常定位。

## 6. 推荐最终文件结构

```text
mvtec-defect-detection/
├── data/
│   ├── bottle.tar.xz
│   └── bottle/
├── src/
│   ├── check_data.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── results/
│   ├── data_check.csv
│   ├── metadata.csv
│   ├── train_log.csv
│   ├── best_model.pth
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   └── misclassified_samples.png
├── README.md
├── requirements.txt
├── plan.md
└── workflow_plan.md
```

## 7. 质量标准

完成后项目应满足：

- 数据检查脚本能独立运行。
- 数据划分可复现。
- 训练脚本能从零开始训练并保存模型。
- 评估脚本能加载模型并输出指标。
- 结果图能直接放进 README。
- README 能让面试官快速理解工业质检价值。
- 不提交原始数据和模型大文件到 GitHub。

## 8. GitHub 提交建议

`.gitignore` 应包含：

```text
data/
results/*.pth
results/*.csv
results/*.json
__pycache__/
.venv/
```

可以提交：

```text
src/
README.md
requirements.txt
plan.md
workflow_plan.md
results/confusion_matrix.png
results/sample_predictions.png
results/misclassified_samples.png
```

如果结果图片用于展示，可以保留小尺寸图片；大模型权重不要提交。

## 9. 里程碑

### Milestone 1：数据检查完成

- `check_data.py` 完成。
- `data_check.csv` 生成。
- `metadata.csv` 生成。
- README 写入数据统计。

### Milestone 2：训练闭环完成

- `dataset.py` 完成。
- `train.py` 完成。
- 能保存 `best_model.pth`。

### Milestone 3：评估闭环完成

- `evaluate.py` 完成。
- 生成指标和混淆矩阵。

### Milestone 4：展示材料完成

- `visualize.py` 完成。
- README 完成。
- 项目可以作为 GitHub 简历项目展示。
