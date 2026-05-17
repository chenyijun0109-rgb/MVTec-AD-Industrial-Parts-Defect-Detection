# 基于 OpenCV 与深度学习的工业零件缺陷检测实验计划

## 1. 项目定位

项目名称：基于 OpenCV 与深度学习的工业零件缺陷检测实验

项目目标：

- 使用 MVTec AD 数据集中的 `bottle` 类别构建工业零件缺陷检测实验。
- 将图片分类为两类：
  - `good`: 正常样本，标签为 `0`
  - defect: 任意缺陷类型样本，标签为 `1`
- 使用预训练 ResNet18 进行二分类训练。
- 输出分类指标与可视化结果，用于 GitHub 和求职项目展示。

项目价值：

- 对应 Valeo 岗位中的计算机视觉、缺陷检测、工业 AI 落地场景。
- 展示从数据准备、模型训练、评估分析到结果可视化的完整实验流程。
- 以最小可行版本为主，先保证项目能跑通、能解释、能展示。

## 2. 数据集选择

数据集：MVTec AD

优先类别：`bottle`

选择原因：

- 缺陷类型直观，适合在 README 和面试中展示。
- 图片结构清晰，适合二分类 MVP。
- 与工业零件外观检测场景较接近。

备选类别：

- `metal_nut`
- `capsule`

数据目录建议：

```text
data/
└── bottle/
    ├── train/
    │   └── good/
    └── test/
        ├── good/
        ├── broken_large/
        ├── broken_small/
        └── contamination/
```

注意：

- `data/` 不提交到 GitHub。
- README 中提供数据下载与放置说明。

## 3. 最小可行版本范围

第一版只做以下内容：

- 读取 `bottle` 类别图片。
- 将图片 resize 到 `224 x 224`。
- 构建二分类标签：
  - `good = 0`
  - 非 `good = 1`
- 使用 PyTorch 加载预训练 ResNet18。
- 修改最后一层为 2 分类。
- 完成训练、验证/测试、指标输出。
- 保存结果图片。
- 编写 README。

第一版暂不做：

- 像素级分割。
- 异常热力图。
- 多类别缺陷分类。
- Web Demo。
- 模型部署。
- 复杂超参数搜索。

## 4. 推荐文件结构

```text
mvtec-defect-detection/
├── data/
│   └── bottle/
├── src/
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── results/
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   └── misclassified_samples.png
├── README.md
├── requirements.txt
└── plan.md
```

## 5. 模块设计

### 5.1 `src/dataset.py`

职责：

- 扫描 MVTec AD `bottle` 目录。
- 读取训练集和测试集图片路径。
- 生成二分类标签。
- 使用 PIL / torchvision transforms 进行图像预处理。

核心逻辑：

- `train/good` 作为正常训练样本。
- `test/good` 标记为 `0`。
- `test/*` 中除 `good` 外的目录全部标记为 `1`。

可选增强：

- 对训练集正常样本做简单数据增强。
- 如果需要训练二分类模型，可从 `test` 中划分一部分 defect 样本加入训练集。

### 5.2 `src/train.py`

职责：

- 加载数据。
- 构建 ResNet18 二分类模型。
- 训练模型。
- 保存最优模型权重。

训练配置建议：

- 输入尺寸：`224 x 224`
- batch size：`16` 或 `32`
- epochs：`5-10`
- optimizer：Adam
- learning rate：`1e-4`
- loss：CrossEntropyLoss

输出：

- `results/best_model.pth`
- 控制台训练日志

### 5.3 `src/evaluate.py`

职责：

- 加载训练好的模型。
- 在测试集上预测。
- 输出分类指标。
- 保存混淆矩阵。

指标：

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

输出：

- `results/metrics.json`
- `results/confusion_matrix.png`

### 5.4 `src/visualize.py`

职责：

- 保存预测正确样本图。
- 保存预测错误样本图。
- 图片上显示真实标签与预测标签。

输出：

- `results/sample_predictions.png`
- `results/misclassified_samples.png`

## 6. 实验流程

### Step 1：准备环境

安装依赖：

```bash
pip install -r requirements.txt
```

推荐依赖：

```text
torch
torchvision
opencv-python
pillow
numpy
scikit-learn
matplotlib
tqdm
```

### Step 2：下载数据

从 MVTec AD 官方网站下载 `bottle` 类别数据。

下载后放置为：

```text
data/bottle/
```

### Step 3：训练模型

运行：

```bash
python src/train.py --data_dir data/bottle --epochs 10 --batch_size 32
```

### Step 4：评估模型

运行：

```bash
python src/evaluate.py --data_dir data/bottle --model_path results/best_model.pth
```

### Step 5：生成可视化结果

运行：

```bash
python src/visualize.py --data_dir data/bottle --model_path results/best_model.pth
```

## 7. README 展示重点

README 建议包含：

- 项目背景：工业质检、缺陷检测、人工检测痛点。
- 数据集说明：MVTec AD `bottle` 类别。
- 方法说明：OpenCV/PIL 预处理 + ResNet18 迁移学习。
- 标签定义：`good` 与 `defect` 二分类。
- 项目结构。
- 运行方法。
- 评估结果表格。
- 混淆矩阵图片。
- 预测样例图片。
- 工业落地价值总结。

工业落地价值可以强调：

- 自动识别外观缺陷，降低人工质检成本。
- 提升检测一致性，减少漏检和误检。
- 可作为产线视觉检测系统的模型原型。
- 后续可扩展到多类别缺陷识别、异常定位和边缘部署。

## 8. 风险与注意事项

### 数据划分问题

MVTec AD 原始训练集通常只有 `good` 样本，官方任务更偏异常检测。

本项目为了做最小可行二分类实验，需要明确说明：

- 这是一个 supervised binary classification MVP。
- 为了训练缺陷类别，可以从 `test` 中划分一部分缺陷样本作为训练集，其余用于测试。
- 后续版本可改进为更贴近真实工业场景的 anomaly detection 方法。

### 数据量较小

可能出现过拟合。

应对方式：

- 使用预训练模型。
- 加入简单数据增强。
- 控制训练轮数。
- 在 README 中如实说明实验性质。

### 类别不平衡

正常和缺陷数量可能不均衡。

应对方式：

- 输出 Precision、Recall、F1，而不只看 Accuracy。
- 必要时使用 class weight 或 WeightedRandomSampler。

## 9. 后续增强方向

完成 MVP 后可以继续扩展：

- 加入更多类别，如 `metal_nut`、`capsule`。
- 实现异常检测方法，如 autoencoder、PatchCore、PaDiM。
- 使用 Grad-CAM 展示模型关注区域。
- 增加 OpenCV 图像预处理对比实验。
- 做一个 Streamlit 或 Gradio 演示界面。
- 导出 ONNX，模拟工业边缘端部署。

## 10. 预计交付物

第一版交付：

- 可运行训练脚本。
- 可运行评估脚本。
- 可视化结果图片。
- 完整 README。
- 清晰的项目结构。

最终 GitHub 展示效果：

- 招聘方可以快速理解项目目标。
- 可以看到模型指标和图片结果。
- 可以看出项目与工业视觉质检岗位高度相关。
