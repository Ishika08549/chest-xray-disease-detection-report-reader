# Chest X-Ray Disease Detection & Radiology Report Reader

An educational deep-learning project combining chest X-ray multi-label classification, Grad-CAM visualization, radiology report extraction, and a full-stack web application.

## Project Overview

The project aims to develop an educational application that:

1. Accepts a chest X-ray image.
2. Predicts candidate chest X-ray labels with confidence scores.
3. Highlights potentially relevant image regions using Grad-CAM.
4. Extracts Findings and Impression sections from radiology reports.
5. Displays a report only when its image-report mapping is verified.

**Disclaimer:** This is an educational research prototype, not a clinically validated medical device. Predictions must not be used for diagnosis or treatment.

## Dataset

**Dataset:** Indiana University Chest X-ray Collection  
**Source:** [Open-i – National Library of Medicine](https://openi.nlm.nih.gov/faq)

Dataset preparation:

- 7,471 chest X-ray images downloaded.
- 3,955 XML radiology reports parsed.
- 7,470 image-report mappings verified.
- Findings and Impression sections extracted.
- Image integrity and report mapping audits performed.
- Report-level train, validation and test splitting used to reduce data leakage.

The selected candidate labels are:

- Normal
- Opacity
- Cardiomegaly
- Atelectasis

These labels are derived from report metadata and are weak labels, not independently verified radiologist annotations.

Raw images, XML reports, archives, processed datasets and model checkpoints are excluded from this repository.

## Dataset Split

| Split | Images |
|---|---:|
| Training | 3,022 |
| Validation | 635 |
| Test | 647 |
| **Total** | **4,304** |

Images linked to the same report are kept within the same split to reduce report-level leakage.

## Completed Development (30 October – 5 November 2026)

- Dataset download and organization.
- XML radiology report parsing.
- Image-report mapping and verification.
- Image integrity and dataset audits.
- Exploratory data analysis.
- Candidate label selection.
- Report-level dataset splitting.
- ResNet18 baseline training (V1).
- ResNet18 improved training experiment (V2).
- ResNet50 transfer-learning experiment.
- Validation evaluation and threshold tuning.
- Final test evaluation of ResNet18 V2 and ResNet50.
- Model comparison and evaluation documentation.

## Model Development

This is a multi-label classification task. More than one candidate label may be associated with an image.

Models trained:

- **ResNet18 V1:** Initial pretrained baseline.
- **ResNet18 V2:** Improved experiment using data augmentation, dropout, class-weighted binary cross-entropy, AdamW and early stopping.
- **ResNet50:** Pretrained ResNet50 transfer-learning experiment.

ResNet50 was selected as the validation-based candidate because it achieved the highest validation macro F1 among the three experiments. ResNet18 V2 is retained as the comparison baseline.

## Model Evaluation

Classification thresholds were selected using validation data and then applied to the held-out test set.

### Validation Macro F1

| Model | Validation Macro F1 |
|---|---:|
| ResNet18 V1 | 0.5925 |
| ResNet18 V2 | 0.5959 |
| ResNet50 | 0.6046 |

### Held-Out Test Results

| Metric (mean across labels) | ResNet18 V2 | ResNet50 |
|---|---:|---:|
| Precision | 0.4892 | 0.5159 |
| Recall | 0.8035 | 0.7347 |
| Macro F1 | 0.5910 | 0.5877 |
| ROC-AUC | 0.8423 | 0.8361 |
| PR-AUC | 0.5825 | 0.5820 |

### Per-Label Test F1

| Label | ResNet18 V2 | ResNet50 |
|---|---:|---:|
| Normal | 0.8648 | 0.8628 |
| Opacity | 0.4346 | 0.4297 |
| Cardiomegaly | 0.5714 | 0.6021 |
| Atelectasis | 0.4931 | 0.4563 |

The test results show that overall performance is close. ResNet18 V2 has a slightly higher test macro F1 and recall, while ResNet50 has higher mean precision and better Cardiomegaly F1.

The validation-selected ResNet50 candidate is retained for the next application development phase. Test results are reported transparently and are not used to retune thresholds.

For detailed metrics, thresholds, confusion matrices and limitations, see [Model Evaluation and Comparison](docs/model_evaluation.md).

## Implemented Image Model Features

- ResNet50-based multi-label X-ray prediction pipeline.
- Four candidate labels: Normal, Opacity, Cardiomegaly and Atelectasis.
- Validation-selected classification thresholds.
- Prediction testing on multiple sample X-ray images.
- Grad-CAM visualization using the final ResNet50 convolutional block.
- Label-specific heatmap generation and overlay saving.
- Prediction and Grad-CAM scripts: `scripts/predict_xray.py` and `scripts/gradcam_xray.py`.

Grad-CAM visualizations show image regions influencing a selected model output. They do not establish disease location or provide a medical diagnosis.

## Full-Stack Application Development

The full-stack application has been implemented and tested locally.

### Backend
- FastAPI backend with health and API endpoints.
- X-ray image upload and multi-label prediction.
- Grad-CAM visualization generation.
- Verified radiology report lookup and extraction.
- File validation, error handling, and CORS configuration.

### Frontend
- React dashboard built with Vite.
- X-ray image upload and prediction display.
- Grad-CAM visualization display.
- Radiology report reader.
- Backend API connection status and medical disclaimer.

### Testing and Validation
- Backend API tests: 6 passed.
- Frontend lint: 0 warnings and 0 errors.
- Frontend production build: successful.
- Local end-to-end testing of prediction, Grad-CAM, and report retrieval completed.

## Technology Stack

- **Language:** Python
- **Deep Learning:** PyTorch, Torchvision
- **Models:** ResNet18, ResNet50
- **Data Processing:** pandas, NumPy, Pillow
- **Report Parsing:** Python XML parser
- **Evaluation:** scikit-learn, Matplotlib
- **Backend:** FastAPI, Uvicorn
- **Frontend:** React, Vite, JavaScript, CSS
- **API Communication:** HTTP, REST endpoints
- **Version Control:** Git and GitHub

## Repository Structure

```text
chest_xray_report_reader_starter/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── main.py
│   └── tests/
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── README.md
├── scripts/
├── docs/
├── data/ (local dataset; excluded from Git)
├── outputs/ (local models; excluded from Git)
├── .gitignore
└── README.md
```

## Backend API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API home |
| GET | `/health` | Health check |
| POST | `/predict` | X-ray prediction |
| POST | `/gradcam` | Grad-CAM heatmap |
| GET | `/report/{image_id}` | Verified report retrieval |

### Run Backend

```bash
python -m uvicorn backend.app.main:app --reload
```

API documentation: `http://127.0.0.1:8000/docs`

### Run Tests

```bash
python -m pytest backend/tests/test_api.py -v
```

### Disclaimer

This is an educational research prototype. Predictions and Grad-CAM visualizations are not medical diagnoses. Retrieved reports are original dataset records, not new clinical interpretations.
