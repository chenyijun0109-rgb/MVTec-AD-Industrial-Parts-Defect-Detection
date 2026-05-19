| Method | Learning setting | Training data used | Defect labels used for training | Supervised checkpoint used | Accuracy | Precision | Recall | F1-score | Localization IoU | Localization Dice | Pointing-hit rate | Main heatmap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 classifier + Grad-CAM | Supervised binary classification | train/good + train/defect | Yes | Yes | 0.939 | 1.000 | 0.846 | 0.917 | 0.216 | 0.321 | 0.538 | Grad-CAM |
| PatchCore anomaly detector | Unsupervised normal-only anomaly detection | train/good | No | No | 0.939 | 0.923 | 0.923 | 0.923 | 0.403 | 0.555 | 1.000 | PatchCore anomaly map |
