<p align="right">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">中文</a>
</p>

# 基于可解释深度学习的工业零件缺陷检测

本项目基于 MVTec AD 数据集中的 `bottle` 类别，构建一个工业视觉质检流程。项目从监督式缺陷二分类模型开始，并进一步加入 Grad-CAM 可解释性分析、PatchCore 无监督异常检测和基于真实 mask 的缺陷定位评估。

分类任务定义为：

- `good = 0`
- `defect = 1`

除了分类准确率，项目还会分析模型是否真正关注到了 MVTec 提供的真实缺陷区域。

## 项目亮点

- 可复现的数据检查与基于 metadata 的训练/验证/测试划分
- 使用 ResNet18 迁移学习，并通过 class-weighted loss 缓解类别不平衡
- 输出 Accuracy、Precision、Recall、F1-score、混淆矩阵和分缺陷类型召回率
- 使用 Grad-CAM 生成模型关注区域热力图
- 使用只依赖正常样本训练的 PatchCore 异常检测 baseline
- 使用 MVTec ground-truth mask 做像素级定位评估
- 生成正确预测、错误样本和缺陷热力图叠加等可视化报告

## 数据集

数据集：MVTec AD

类别：`bottle`

本地目录结构：

```text
data/
└── bottle/
    ├── ground_truth/
    ├── test/
    └── train/
```

已检查的数据数量：

| Split | Type | Count |
| --- | --- | ---: |
| train | good | 209 |
| test | good | 20 |
| test | broken_large | 20 |
| test | broken_small | 22 |
| test | contamination | 21 |

`data/` 目录已被 Git 忽略，不应提交到 GitHub。

## 实验设计

MVTec AD 官方设定更偏向异常检测：

- 训练集通常只包含正常样本
- 测试集包含正常样本和缺陷样本
- 缺陷样本提供像素级 ground-truth mask

本项目以两种方式使用该数据集：

1. **监督式二分类实验：** 通过固定 metadata 划分，将部分缺陷样本分配到训练集、验证集和测试集。
2. **可解释性与定位分析：** 在测试集缺陷样本上，将 Grad-CAM 热力图与真实缺陷 mask 进行对比。
3. **无监督异常检测：** PatchCore 只使用 `train/good` 建立正常特征记忆库，使用 `val/good` 选择阈值，并在测试集上评估。

默认划分规则：

- `train/good` 中 80% 进入训练集
- `train/good` 中 20% 进入验证集
- `test/good` 全部保留为测试集
- 每个缺陷类型中 60% 进入训练集，20% 进入验证集，其余进入测试集
- 随机种子：`42`

当前实验划分：

| Experiment Split | Good | Defect | Total |
| --- | ---: | ---: | ---: |
| train | 167 | 38 | 205 |
| val | 42 | 12 | 54 |
| test | 20 | 13 | 33 |

## 项目结构

```text
.
├── data/
│   └── bottle/
├── src/
│   ├── check_data.py
│   ├── compare_methods.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── model.py
│   ├── patchcore.py
│   ├── run_patchcore.py
│   ├── train.py
│   └── visualize.py
├── results/
│   ├── confusion_matrix.png
│   ├── gradcam_mask_overlay.png
│   ├── sample_predictions.png
│   └── misclassified_samples.png
├── README.md
├── README.zh-CN.md
├── pyproject.toml
└── requirements.txt
```

## 环境安装

本项目使用 `uv`。

```bash
uv sync
```

默认配置安装 CPU 版本 PyTorch。对于当前数据规模和项目展示已经足够。

## 运行流程

### 1. 数据检查与 metadata 生成

```bash
uv run python src/check_data.py --data_dir data/bottle
```

输出：

```text
results/data_check.csv
results/metadata.csv
```

### 2. 训练模型

```bash
uv run python src/train.py --epochs 10 --batch_size 16
```

输出：

```text
results/best_model.pth
results/train_log.csv
```

模型使用 ImageNet 预训练 ResNet18，并将最后的全连接层替换为 2 分类层。

如果无法下载预训练权重，可以运行：

```bash
uv run python src/train.py --epochs 10 --batch_size 16 --no_pretrained
```

### 3. 分类评估

```bash
uv run python src/evaluate.py
```

输出：

```text
results/metrics.json
results/predictions.csv
results/confusion_matrix.png
```

### 4. 生成预测可视化

```bash
uv run python src/visualize.py
```

输出：

```text
results/sample_predictions.png
results/misclassified_samples.png
```

### 5. 生成 Grad-CAM 与定位指标

```bash
uv run python src/explain.py --max_visualizations 8
```

输出：

```text
results/gradcam_localization.csv
results/localization_metrics.json
results/gradcam_mask_overlay.png
```

该脚本会针对 defect 类别生成 Grad-CAM，并将归一化热力图与真实 mask 对比，输出 IoU、Dice、pointing-hit rate 以及 mask 内外平均激活值。

### 6. 运行 PatchCore 异常检测

PatchCore 与前面的监督分类模型相互独立。运行 PatchCore 前不需要先执行 `src/train.py`，`src/run_patchcore.py` 也不会加载 `results/best_model.pth` 或任何在 MVTec 缺陷标签上训练过的 checkpoint。PatchCore 只用 `train/good` 建立 memory bank，只用 `val/good` 选择阈值。

```bash
uv run python src/run_patchcore.py --batch_size 4 --coreset_ratio 0.05 --max_visualizations 8
```

输出：

```text
results/patchcore/patchcore_metrics.json
results/patchcore/patchcore_predictions.csv
results/patchcore/patchcore_localization.csv
results/patchcore/patchcore_heatmaps.png
results/patchcore/patchcore_score_distribution.png
results/patchcore/anomaly_maps/
```

PatchCore 默认使用冻结的 ImageNet 预训练 ResNet18 中间层特征，这是 PatchCore 常见做法，不使用任何 MVTec 缺陷标签。上面的结果表格基于这个默认 pretrained backbone。

如果无法下载预训练权重，或你希望使用完全未预训练的 backbone，可以运行：

```bash
uv run python src/run_patchcore.py --no_pretrained
```

`--no_pretrained` 只是 fallback 或消融实验设置。它使用随机初始化且冻结的 ResNet18 特征，因此 anomaly score、recall 和定位指标都可能明显低于默认 PatchCore 结果。

### 7. 用统一报告格式对比方法

```bash
uv run python src/compare_methods.py
```

输出：

```text
results/method_comparison.csv
results/method_comparison.md
```

该报告使用同一套 image-level 和 localization-level 指标，对比监督式 ResNet18 分类模型与无监督 PatchCore 异常检测模型。

## 实验结果

### 统一方法对比

| Method | Learning setting | Training data used | Defect labels used for training | Supervised checkpoint used | Accuracy | Precision | Recall | F1-score | Localization IoU | Localization Dice | Pointing-hit rate | Main heatmap |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ResNet18 classifier + Grad-CAM | Supervised binary classification | train/good + train/defect | Yes | Yes | 0.939 | 1.000 | 0.846 | 0.917 | 0.216 | 0.321 | 0.538 | Grad-CAM |
| PatchCore anomaly detector | Unsupervised normal-only anomaly detection | train/good | No | No | 0.939 | 0.923 | 0.923 | 0.923 | 0.403 | 0.555 | 1.000 | PatchCore anomaly map |

这张表体现了两种方法的取舍：监督分类模型在当前测试划分上 Precision 更高，没有误报；PatchCore 不使用缺陷标签训练，但 Recall 和定位指标更好，更贴近 MVTec AD 的无监督异常检测设定。

当前测试集分类结果：

| Metric | Value |
| --- | ---: |
| Accuracy | 0.939 |
| Precision | 1.000 |
| Recall | 0.846 |
| F1-score | 0.917 |

分缺陷类型召回率：

| Defect Type | Recall |
| --- | ---: |
| broken_large | 1.000 |
| broken_small | 1.000 |
| contamination | 0.500 |

在 13 张带 mask 的缺陷测试样本上的定位分析：

| Metric | Value |
| --- | ---: |
| Mean CAM IoU | 0.216 |
| Mean CAM Dice | 0.321 |
| Pointing-hit rate | 0.538 |
| Mean CAM activation inside mask | 0.620 |
| Mean CAM activation outside mask | 0.230 |

PatchCore 无监督异常检测结果：

| Metric | Value |
| --- | ---: |
| Accuracy | 0.939 |
| Precision | 0.923 |
| Recall | 0.923 |
| F1-score | 0.923 |
| Threshold | 0.450 |
| Memory bank size | 6546 |
| 是否使用监督 MVTec checkpoint | No |

PatchCore 在 13 张带 mask 的缺陷测试样本上的定位分析：

| Metric | Value |
| --- | ---: |
| Mean anomaly IoU | 0.403 |
| Mean anomaly Dice | 0.555 |
| Pointing-hit rate | 1.000 |
| Mean anomaly inside mask | 0.777 |
| Mean anomaly outside mask | 0.252 |

## 可视化报告

### 混淆矩阵

![Confusion matrix](results/confusion_matrix.png)

### 正确预测样本

![Sample predictions](results/sample_predictions.png)

### 错误样本分析

![Misclassified samples](results/misclassified_samples.png)

### Grad-CAM 与真实 mask 叠加

![Grad-CAM mask overlay](results/gradcam_mask_overlay.png)

### PatchCore 异常热力图

![PatchCore heatmaps](results/patchcore/patchcore_heatmaps.png)

## 工业应用价值

该项目模拟了一个基础但完整的工业视觉质检流程：

- 自动识别产品外观异常
- 通过 Precision 和 Recall 分析误检与漏检风险
- 通过分缺陷类型召回率定位薄弱缺陷类别
- 使用 Grad-CAM 分析模型是否关注了合理区域
- 使用 PatchCore 在不训练缺陷样本的情况下检测异常
- 使用统一报告格式对比监督式分类和无监督异常检测
- 将模型关注区域与真实缺陷 mask 对比，评估定位能力
- 为后续异常检测、缺陷定位和交互式部署演示提供基础


## 后续优化方向

- 继续调优 PatchCore 的 coreset 选择、阈值策略和特征层
- 扩展到 `metal_nut`、`capsule` 等更多 MVTec 类别
- 构建 Streamlit 或 Gradio 交互式质检 demo，支持上传图片、预测和热力图展示
