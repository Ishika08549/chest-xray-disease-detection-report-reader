import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    multilabel_confusion_matrix
)

# 1. Configuration

CSV_PATH = "data/processed/model_image_manifest_split.csv"
MODEL_PATH = "outputs/models/best_resnet18_v2.pth"

LABELS = [
    "Normal",
    "Opacity",
    "Cardiomegaly",
    "Atelectasis"
]

# Tuned thresholds obtained from validation data
THRESHOLDS = np.array([
    0.4627,
    0.3932,
    0.5490,
    0.5764
])

BATCH_SIZE = 16

OUTPUT_DIR = "outputs/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Device

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Selected device:", device)

# 3. Preprocessing

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 4. Dataset

class ChestXrayDataset(Dataset):

    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        image = transform(image)

        labels = torch.tensor(
            row[LABELS].to_numpy(dtype="float32"),
            dtype=torch.float32
        )

        return image, labels


# 5. Load test dataset

df = pd.read_csv(CSV_PATH)

test_df = df[df["split"] == "test"].copy()

test_dataset = ChestXrayDataset(test_df)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(test_dataset))

# 6. Build V2 model

def build_model():

    model = models.resnet18(weights=None)

    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, len(LABELS))
    )

    return model


# 7. Load trained checkpoint

model = build_model()

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print("V2 checkpoint loaded successfully")


# 8. Generate predictions

all_labels = []
all_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.sigmoid(outputs)

        all_labels.append(labels.numpy())

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

y_true = np.vstack(all_labels)
y_prob = np.vstack(all_probabilities)


# 9. Apply tuned thresholds

y_pred = (
    y_prob >= THRESHOLDS.reshape(1, -1)
).astype(int)


# 10. Calculate evaluation metrics

results = []

for i, label in enumerate(LABELS):

    precision = precision_score(
        y_true[:, i],
        y_pred[:, i],
        zero_division=0
    )

    recall = recall_score(
        y_true[:, i],
        y_pred[:, i],
        zero_division=0
    )

    f1 = f1_score(
        y_true[:, i],
        y_pred[:, i],
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true[:, i],
        y_prob[:, i]
    )

    pr_auc = average_precision_score(
        y_true[:, i],
        y_prob[:, i]
    )

    results.append({
        "Model": "V2",
        "Label": label,
        "Threshold": THRESHOLDS[i],
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    })


# 11. Confusion matrices

matrices = multilabel_confusion_matrix(
    y_true,
    y_pred
)

matrix_rows = []

for label, matrix in zip(LABELS, matrices):

    tn, fp, fn, tp = matrix.ravel()

    matrix_rows.append({
        "Model": "V2",
        "Label": label,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })


# 12. Classification report

print("\nFINAL TEST CLASSIFICATION REPORT")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=LABELS,
        zero_division=0
    )
)


# 13. Save results

results_df = pd.DataFrame(results)

matrices_df = pd.DataFrame(matrix_rows)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_test_results.csv"
    ),
    index=False
)

matrices_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_test_confusion_matrices.csv"
    ),
    index=False
)


# 14. Display results

print("\nFINAL TEST METRICS")

print(
    results_df.round(4).to_string(index=False)
)

print("\nFINAL TEST CONFUSION MATRICES")

print(
    matrices_df.to_string(index=False)
)

print("\nSaved final test evaluation files.")

print("\nFinal test evaluation completed successfully!")