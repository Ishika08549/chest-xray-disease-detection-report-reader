# Model Evaluation and Comparison

## 1. Dataset and Evaluation Setup

- Task: Multi-label chest X-ray image classification.
- Labels: Normal, Opacity, Cardiomegaly, Atelectasis.
- Training images: 3,022
- Validation images: 635
- Test images: 647
- Split strategy: Report-level splitting to reduce leakage between related images.
- Evaluation library: scikit-learn.
- Models: ResNet18 V1, ResNet18 V2, ResNet50.

The labels are derived from radiology report metadata and are weak labels,
not independently confirmed radiologist annotations.

## 2. Validation-Based Threshold Tuning

Each model's classification thresholds were selected using validation data.
The test set was not used for threshold selection.

ResNet18 V2 thresholds:

| Label | Threshold |
|---|---:|
| Normal | 0.4627 |
| Opacity | 0.3932 |
| Cardiomegaly | 0.5490 |
| Atelectasis | 0.5764 |

ResNet50 thresholds:

| Label | Threshold |
|---|---:|
| Normal | 0.3549 |
| Opacity | 0.4506 |
| Cardiomegaly | 0.6368 |
| Atelectasis | 0.7515 |

## 3. Validation Model Selection

Mean per-label F1 after validation threshold tuning:

| Model | Validation Macro F1 |
|---|---:|
| ResNet18 V1 | 0.5925 |
| ResNet18 V2 | 0.5959 |
| ResNet50 | 0.6046 |

ResNet50 was selected as the validation-based candidate because it achieved
the highest validation macro F1. Its improvement over V2 was small.

These validation metrics are optimistic estimates because thresholds were
optimized on the same validation set.

## 4. Held-Out Test Results

All models must be evaluated using thresholds selected on validation data.
The held-out test set contains 647 images.

| Metric (mean across labels) | ResNet18 V2 | ResNet50 |
|---|---:|---:|
| Precision | 0.4892 | 0.5159 |
| Recall | 0.8035 | 0.7347 |
| Macro F1 | 0.5910 | 0.5877 |
| ROC-AUC | 0.8423 | 0.8361 |
| PR-AUC | 0.5825 | 0.5820 |

### Per-label test F1

| Label | ResNet18 V2 | ResNet50 |
|---|---:|---:|
| Normal | 0.8648 | 0.8628 |
| Opacity | 0.4346 | 0.4297 |
| Cardiomegaly | 0.5714 | 0.6021 |
| Atelectasis | 0.4931 | 0.4563 |

## 5. Interpretation

- ResNet18 V2 achieved a slightly higher test macro F1 and recall.
- ResNet50 achieved higher mean precision and a higher F1 for Cardiomegaly.
- Overall test performance is close; neither model is consistently better
  across all labels.
- ResNet50 remains the validation-selected candidate. The test results are
  reported transparently and are not used to retune thresholds.
- Opacity has relatively low precision, indicating a substantial number of
  false-positive predictions.

## 6. Limitations

- Labels are weak report-derived labels, not radiologist-confirmed ground truth.
- Report-derived labels may contain noise or omissions.
- Images associated with the same report are related; report-level splitting
  was used to reduce leakage.
- Thresholds were optimized for validation F1 and may not generalize.
- Results do not establish clinical diagnostic performance.
- This project is an educational research prototype and must not be used
  for diagnosis or treatment.

## 7. Evaluation Artifacts

Generated locally and excluded from Git:

- `outputs/evaluation/final_test_results.csv`
- `outputs/evaluation/final_test_confusion_matrices.csv`
- `outputs/evaluation/resnet50_final_test_results.csv`
- `outputs/evaluation/resnet50_final_test_confusion_matrices.csv`
- `outputs/evaluation/threshold_comparison_all.csv`
