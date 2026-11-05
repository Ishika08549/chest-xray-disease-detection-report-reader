import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    multilabel_confusion_matrix
)

CSV_PATH = "data/processed/model_image_manifest_split.csv"
CHECKPOINT_PATH = "outputs/models/best_resnet50.pth"
OUTPUT_DIR = "outputs/evaluation"

LABELS = [
    "Normal",
    "Opacity",
    "Cardiomegaly",
    "Atelectasis"
]

BATCH_SIZE = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Selected device:", device)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


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
            row[LABELS].to_numpy(dtype=np.float32),
            dtype=torch.float32
        )

        return image, labels


df = pd.read_csv(CSV_PATH)
val_df = df[df["split"] == "validation"].copy()

dataset = ChestXrayDataset(val_df)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Validation images:", len(dataset))


print("\nLoading ResNet50 checkpoint...")

model = models.resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(model.fc.in_features, len(LABELS))
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

print("ResNet50 checkpoint loaded successfully.")
print("Best training epoch:", checkpoint["epoch"])
print("Best validation loss:", checkpoint["best_val_loss"])


all_labels = []
all_probabilities = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)

        outputs = model(images)
        probabilities = torch.sigmoid(outputs)

        all_labels.append(labels.numpy())
        all_probabilities.append(probabilities.cpu().numpy())

y_true = np.vstack(all_labels)
y_prob = np.vstack(all_probabilities)

# Same 0.5 threshold for a fair initial comparison
y_pred = (y_prob >= 0.5).astype(int)

results = []

for i, label in enumerate(LABELS):

    precision = precision_score(
        y_true[:, i], y_pred[:, i], zero_division=0
    )

    recall = recall_score(
        y_true[:, i], y_pred[:, i], zero_division=0
    )

    f1 = f1_score(
        y_true[:, i], y_pred[:, i], zero_division=0
    )

    if len(np.unique(y_true[:, i])) == 2:
        roc_auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        pr_auc = average_precision_score(y_true[:, i], y_prob[:, i])
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    results.append({
        "Model": "ResNet50",
        "Label": label,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    })


print("\nRESNET50 CLASSIFICATION REPORT")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=LABELS,
        zero_division=0
    )
)

results_df = pd.DataFrame(results)

results_path = os.path.join(
    OUTPUT_DIR, "resnet50_validation_results.csv"
)

results_df.to_csv(results_path, index=False)


matrices = multilabel_confusion_matrix(y_true, y_pred)

matrix_rows = []

for label, matrix in zip(LABELS, matrices):
    tn, fp, fn, tp = matrix.ravel()

    matrix_rows.append({
        "Model": "ResNet50",
        "Label": label,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

matrices_df = pd.DataFrame(matrix_rows)

matrices_path = os.path.join(
    OUTPUT_DIR, "resnet50_validation_confusion_matrices.csv"
)

matrices_df.to_csv(matrices_path, index=False)


# Combine with the existing ResNet18 V1 and V2 results
old_results_path = os.path.join(
    OUTPUT_DIR, "model_comparison.csv"
)

if os.path.exists(old_results_path):
    old_results = pd.read_csv(old_results_path)

    combined = pd.concat(
        [old_results, results_df],
        ignore_index=True
    )

    combined.to_csv(
        os.path.join(OUTPUT_DIR, "model_comparison_all.csv"),
        index=False
    )

    print("\nCOMBINED MODEL COMPARISON")
    print(combined.round(4).to_string(index=False))

else:
    print("\nExisting V1/V2 comparison file not found.")
    print("ResNet50 results saved separately.")


print("\nResNet50 validation results saved:", results_path)
print("ResNet50 confusion matrices saved:", matrices_path)
print("\nEvaluation completed successfully.")
