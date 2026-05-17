<p align="right">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">中文</a>
</p>

# Industrial Parts Defect Detection with OpenCV and Deep Learning

This project builds a minimum viable industrial visual inspection pipeline using the `bottle` category from the MVTec AD dataset. The task is binary image classification:

- `good = 0`
- `defect = 1`

The project covers data checking, metadata-based splitting, image preprocessing, ResNet18 transfer learning, evaluation metrics, and visual result generation.

## Why This Project

Industrial defect detection is a common computer vision use case in manufacturing. This project is designed to demonstrate a practical workflow for:

- automated visual quality inspection
- normal/defective part classification
- reproducible data preparation
- model training and evaluation
- result visualization for engineering review

It is especially relevant to industrial AI, computer vision, and manufacturing quality inspection roles.

## Dataset

Dataset: MVTec AD

Category: `bottle`

Expected local structure:

```text
data/
└── bottle/
    ├── ground_truth/
    ├── test/
    └── train/
```

Checked data counts:

| Split | Type | Count |
| --- | --- | ---: |
| train | good | 209 |
| test | good | 20 |
| test | broken_large | 20 |
| test | broken_small | 22 |
| test | contamination | 21 |

<!-- The `data/` directory is ignored by Git and should not be committed. -->

## Important Note About Splitting

The official MVTec AD setup is mainly an anomaly detection benchmark:

- training data contains only normal images
- test data contains both normal and defective images
- defective images include pixel-level masks

This first version is a supervised binary classification MVP. To train a binary classifier, part of the defective test images is assigned to the training and validation splits through a fixed metadata file.

Default split rule:

- 80% of `train/good` goes to training
- 20% of `train/good` goes to validation
- all `test/good` images stay in the test split
- for each defect type, 60% goes to training, 20% to validation, and the rest to testing
- random seed: `42`

Current experiment split:

| Experiment Split | Good | Defect | Total |
| --- | ---: | ---: | ---: |
| train | 167 | 38 | 205 |
| val | 42 | 12 | 54 |
| test | 20 | 13 | 33 |

This should be described as a binary classification experiment, not as the official MVTec anomaly detection benchmark.

## Project Structure

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

## Environment Setup

This project uses `uv`.

```bash
uv sync
```

The default configuration installs the CPU version of PyTorch. This is enough for the current dataset and project demonstration.

## How To Run

### 1. Check Data And Create Metadata

```bash
uv run python src/check_data.py --data_dir data/bottle
```

Outputs:

```text
results/data_check.csv
results/metadata.csv
```

### 2. Train

```bash
uv run python src/train.py --epochs 10 --batch_size 16
```

Outputs:

```text
results/best_model.pth
results/train_log.csv
```

The model uses ImageNet-pretrained ResNet18 and replaces the final fully connected layer with a 2-class classifier.

If pretrained weight download is unavailable, run:

```bash
uv run python src/train.py --epochs 10 --batch_size 16 --no_pretrained
```

### 3. Evaluate

```bash
uv run python src/evaluate.py
```

Outputs:

```text
results/metrics.json
results/predictions.csv
results/confusion_matrix.png
```

Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Per-defect-type recall

### 4. Visualize Predictions

```bash
uv run python src/visualize.py
```

Outputs:

```text
results/sample_predictions.png
results/misclassified_samples.png
```

## Results

After running evaluation and visualization, the following files can be used in the GitHub README:

```text
results/confusion_matrix.png
results/sample_predictions.png
results/misclassified_samples.png
```

## Industrial Value

This project simulates a basic industrial visual inspection workflow:

- detect abnormal product appearance automatically
- reduce manual inspection workload
- evaluate false positives and missed detections with precision and recall
- inspect failure cases through prediction visualization
- provide a foundation for future anomaly localization and deployment

## Next Steps

- Use ground-truth masks for anomaly localization
- Add Grad-CAM for model attention visualization
- Try anomaly detection methods such as PatchCore, PaDiM, or AutoEncoder
- Extend the pipeline to `metal_nut` and `capsule`
- Build a Streamlit or Gradio demo
