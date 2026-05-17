<p align="right">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">中文</a>
</p>

# 基于 OpenCV 与深度学习的工业零件缺陷检测实验

本项目使用 MVTec AD 数据集中的 `bottle` 类别，构建一个最小可行的工业视觉质检流程。任务是二分类图像识别：

- `good = 0`
- `defect = 1`

项目覆盖数据检查、基于 metadata 的数据划分、图像预处理、ResNet18 迁移学习、评估指标输出和预测结果可视化。

## 项目价值

工业缺陷检测是制造业中非常典型的计算机视觉应用。本项目用于展示一个较完整的工程流程：

- 自动化视觉质检
- 正常/缺陷零件分类
- 可复现的数据准备
- 模型训练与评估
- 面向工程复盘的结果可视化

该项目与工业 AI、计算机视觉、制造业质量检测岗位高度相关。

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

<!-- `data/` 目录已被 Git 忽略，不应提交到 GitHub。 -->

## 关于数据划分

MVTec AD 官方设定更偏向异常检测：

- 训练集通常只有正常样本
- 测试集包含正常和缺陷样本
- 缺陷样本提供像素级 mask

本项目第一版是 supervised binary classification MVP。为了训练二分类模型，需要从缺陷测试图片中固定划分一部分用于训练和验证。

默认划分规则：

- `train/good` 中 80% 进入训练集
- `train/good` 中 20% 进入验证集
- `test/good` 全部保留为测试集
- 每个缺陷类型中 60% 进入训练集，20% 进入验证集，剩余进入测试集
- 随机种子：`42`

当前实验划分：

| Experiment Split | Good | Defect | Total |
| --- | ---: | ---: | ---: |
| train | 167 | 38 | 205 |
| val | 42 | 12 | 54 |
| test | 20 | 13 | 33 |

因此该项目应表述为“工业缺陷二分类实验”，而不是 MVTec 官方异常检测 benchmark。

## 项目结构

```text
.
├── data/
│   └── bottle/
├── src/
│   ├── check_data.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── train.py
│   └── visualize.py
├── results/
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   └── misclassified_samples.png
├── README.md
├── README.zh-CN.md
├── pyproject.toml
├── requirements.txt
├── plan.md
└── workflow_plan.md
```

## 环境安装

本项目使用 `uv`。

```bash
uv sync
```

项目默认安装 PyTorch CPU 版本。对于当前数据规模和项目展示来说，CPU 版本已经足够。

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

模型使用 ImageNet 预训练 ResNet18，并将最后一层替换为 2 分类层。

如果无法下载预训练权重，可以运行：

```bash
uv run python src/train.py --epochs 10 --batch_size 16 --no_pretrained
```

### 3. 评估模型

```bash
uv run python src/evaluate.py
```

输出：

```text
results/metrics.json
results/predictions.csv
results/confusion_matrix.png
```

评估指标：

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Per-defect-type recall

### 4. 生成预测可视化

```bash
uv run python src/visualize.py
```

输出：

```text
results/sample_predictions.png
results/misclassified_samples.png
```

## 结果展示

完成评估和可视化后，可以将以下图片用于 GitHub 展示：

```text
results/confusion_matrix.png
results/sample_predictions.png
results/misclassified_samples.png
```

## 工业应用价值

该实验模拟了基础的工业视觉质检流程：

- 自动识别产品外观异常
- 减少人工质检工作量
- 通过 Precision 和 Recall 分析误检与漏检风险
- 通过预测样例图分析模型失败案例
- 为后续异常定位和部署提供基础

## 后续优化方向

- 使用 ground-truth mask 做异常定位
- 加入 Grad-CAM 展示模型关注区域
- 尝试 PatchCore、PaDiM、AutoEncoder 等异常检测方法
- 扩展到 `metal_nut` 和 `capsule`
- 构建 Streamlit 或 Gradio 交互式演示
