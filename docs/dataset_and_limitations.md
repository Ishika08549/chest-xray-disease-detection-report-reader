# Dataset Description and Limitations

## 1. Dataset Source

The project uses the Indiana University Chest X-ray Collection available through Open-i, provided by the National Library of Medicine.

Source: https://openi.nlm.nih.gov/faq

The downloaded dataset contains chest X-ray images and associated XML radiology reports.

Dataset redistribution must follow the applicable source license and terms. The downloaded XML sample indicates a CC BY-NC-ND 4.0 license. Dataset files are therefore not included in this Git repository.

## 2. Dataset Overview

The local dataset preparation and audit produced the following results:

| Item | Count |
|---|---:|
| Downloaded chest X-ray images | 7,471 |
| XML radiology reports parsed | 3,955 |
| Verified image-report mappings | 7,470 |
| Reports without image references | 104 |
| Images without a manifest mapping | 1 |
| Parsed reports with Findings | 3,425 |
| Parsed reports with Impression | 3,921 |
| Rows in merged image-report dataset | 7,470 |
| Unique reports in merged dataset | 3,851 |

All 3,955 XML reports were parsed successfully.

All image-report mappings included in the verified manifest passed the mapping verification checks.

## 3. Report Extraction

The XML parser extracts the following information:

- Report ID
- Report UID
- Comparison
- Indication
- Findings
- Impression
- MeSH terms
- Parsing status

Findings and Impression are extracted separately because they represent different sections of a radiology report.

## 4. Candidate Labels

Four candidate labels were selected for multi-label classification:

1. Normal
2. Opacity
3. Cardiomegaly
4. Atelectasis

These labels were derived from report metadata and MeSH terms.

They are weak labels and should not be treated as independently verified radiologist annotations.

More than one label may be associated with a report.

## 5. Model Dataset and Splitting

The model dataset contains 4,304 images associated with 2,211 reports.

Reports without any of the four selected candidate labels were excluded from this model subset. This excludes 1,640 reports and may introduce selection bias.

The data was split at the report level to reduce leakage between training, validation and test sets.

| Split | Images | Unique Reports |
|---|---:|---:|
| Training | 3,022 | 1,547 |
| Validation | 635 | 332 |
| Test | 647 | 332 |
| **Total** | **4,304** | **2,211** |

Images associated with the same report were kept in the same split.

## 6. Data Quality Checks

The preparation and audit process included:

- XML parsing success and failure checks.
- Image-report mapping verification.
- Duplicate image ID checks.
- Duplicate image-report pair checks.
- Missing image file checks.
- Image corruption checks.
- Image mode and dimensions checks.
- Label column and missing-value checks.
- Report-level split leakage checks.

The audited model images were readable and available at their recorded paths.

## 7. Important Limitations

### Weak Labels

The candidate labels are derived from report metadata and MeSH terms. They are not confirmed ground-truth labels reviewed independently by radiologists.

### Label Co-occurrence

The task is multi-label classification. Multiple candidate labels may occur together, so label counts do not have to add up to the number of images.

### Selection Bias

Reports without the four selected labels were excluded from the model dataset. The model therefore does not represent the complete original dataset.

### Class Imbalance

The selected labels have different positive sample counts. This can affect model learning and prediction performance.

### Model Performance

The model has shown stronger results for the Normal label than for some other labels. Opacity predictions have relatively low precision and can produce false positives.

### Generalization

The results are based on this prepared dataset and do not establish performance on images from other hospitals, equipment or patient populations.

## 8. Responsible Use

This project is an educational and research prototype.

- Predictions are candidate labels, not medical diagnoses.
- Grad-CAM visualizations are explanatory aids, not proof of disease.
- Extracted report text should be presented with its source context.
- The application must not generate prescriptions or recommend treatment.
- The system must not replace qualified medical professionals.

## 9. Related Documentation

For model configurations, validation thresholds, test metrics and confusion matrices, see:

[Model Evaluation and Comparison](model_evaluation.md)
