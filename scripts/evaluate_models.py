
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

LABELS = [
    "Normal",
    "Opacity",
    "Cardiomegaly",
    "Atelectasis"
]

BATCH_SIZE = 16

V1_PATH = "outputs/models/best_resnet18.pth"
V2_PATH = "outputs/models/best_resnet18_v2.pth"

OUTPUT_DIR = "outputs/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Device

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Selected device:", device)

# 3. Preprocessing
# Must match the validation preprocessing used during training.

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


# 5. Validation DataLoader

df = pd.read_csv(CSV_PATH)

val_df = df[df["split"] == "validation"].copy()

val_dataset = ChestXrayDataset(val_df)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Validation images:", len(val_dataset))

# 6. Model architectures

def build_model(model_version):

    model = models.resnet18(weights=None)

    num_features = model.fc.in_features

    if model_version == "V1":

        model.fc = nn.Linear(
            num_features,
            len(LABELS)
        )

    elif model_version == "V2":

        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_features, len(LABELS))
        )

    return model


# 7. Load checkpoint

def load_model(model_version, checkpoint_path):

    model = build_model(model_version)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    print(f"{model_version} checkpoint loaded successfully")

    return model


# 8. Prediction

def get_predictions(model):

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.sigmoid(outputs)

            all_labels.append(labels.numpy())
            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probabilities)

    return y_true, y_prob


# 9. Evaluation

def evaluate_model(model_version, checkpoint_path):

    print(f"\nEvaluating {model_version}")

    model = load_model(
        model_version,
        checkpoint_path
    )

    y_true, y_prob = get_predictions(model)

    # Initial threshold for all labels
    y_pred = (y_prob >= 0.5).astype(int)

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

        if len(np.unique(y_true[:, i])) == 2:

            roc_auc = roc_auc_score(
                y_true[:, i],
                y_prob[:, i]
            )

            pr_auc = average_precision_score(
                y_true[:, i],
                y_prob[:, i]
            )

        else:
            roc_auc = np.nan
            pr_auc = np.nan

        results.append({
            "Model": model_version,
            "Label": label,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc
        })

    # Save confusion matrices
    matrices = multilabel_confusion_matrix(
        y_true,
        y_pred
    )

    matrix_rows = []

    for label, matrix in zip(LABELS, matrices):

        tn, fp, fn, tp = matrix.ravel()

        matrix_rows.append({
            "Model": model_version,
            "Label": label,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        })

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=LABELS,
            zero_division=0
        )
    )

    return results, matrix_rows


# 10. Run evaluation

v1_results, v1_matrices = evaluate_model(
    "V1",
    V1_PATH
)

v2_results, v2_matrices = evaluate_model(
    "V2",
    V2_PATH
)

# 11. Save results

all_results = v1_results + v2_results
all_matrices = v1_matrices + v2_matrices

results_df = pd.DataFrame(all_results)
matrices_df = pd.DataFrame(all_matrices)

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)

matrices_df.to_csv(
    os.path.join(OUTPUT_DIR, "confusion_matrices.csv"),
    index=False
)

print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

print("\nConfusion matrices saved.")

print("\nEvaluation completed successfully!")
