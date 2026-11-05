
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
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

CHECKPOINTS = {
    "V1": "outputs/models/best_resnet18.pth",
    "V2": "outputs/models/best_resnet18_v2.pth"
}

OUTPUT_DIR = "outputs/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Selected device:", device)

# 2. Preprocessing

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 3. Dataset

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


df = pd.read_csv(CSV_PATH)

val_df = df[df["split"] == "validation"].copy()

val_loader = DataLoader(
    ChestXrayDataset(val_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Validation images:", len(val_df))

# 4. Build model

def build_model(version):

    model = models.resnet18(weights=None)

    features = model.fc.in_features

    if version == "V1":
        model.fc = nn.Linear(features, len(LABELS))

    else:
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(features, len(LABELS))
        )

    return model


# 5. Get predictions

def predict(version, checkpoint_path):

    model = build_model(version)

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

    all_labels = []
    all_probs = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.sigmoid(outputs)

            all_labels.append(labels.numpy())
            all_probs.append(probabilities.cpu().numpy())

    return np.vstack(all_labels), np.vstack(all_probs)


# 6. Find best F1 threshold per label

all_results = []

for version, checkpoint_path in CHECKPOINTS.items():

    print(f"\nTuning thresholds for {version}")

    y_true, y_prob = predict(
        version,
        checkpoint_path
    )

    for i, label in enumerate(LABELS):

        precision, recall, thresholds = precision_recall_curve(
            y_true[:, i],
            y_prob[:, i]
        )

        f1_values = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-10)
        )

        best_index = np.argmax(f1_values)

        best_threshold = float(thresholds[best_index])

        # Original threshold
        original_pred = (y_prob[:, i] >= 0.5).astype(int)

        # Tuned threshold
        tuned_pred = (
            y_prob[:, i] >= best_threshold
        ).astype(int)

        original_f1 = f1_score(
            y_true[:, i],
            original_pred,
            zero_division=0
        )

        tuned_precision = precision_score(
            y_true[:, i],
            tuned_pred,
            zero_division=0
        )

        tuned_recall = recall_score(
            y_true[:, i],
            tuned_pred,
            zero_division=0
        )

        tuned_f1 = f1_score(
            y_true[:, i],
            tuned_pred,
            zero_division=0
        )

        all_results.append({
            "Model": version,
            "Label": label,
            "Threshold_0.5_F1": original_f1,
            "Best_Threshold": best_threshold,
            "Tuned_Precision": tuned_precision,
            "Tuned_Recall": tuned_recall,
            "Tuned_F1": tuned_f1
        })

        print(
            f"{label}: threshold={best_threshold:.4f}, "
            f"F1={tuned_f1:.4f}"
        )


# 7. Save results

results_df = pd.DataFrame(all_results)

output_path = os.path.join(
    OUTPUT_DIR,
    "threshold_comparison.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\nTHRESHOLD COMPARISON")
print(results_df.round(4).to_string(index=False))

print("\nSaved:", output_path)
print("Threshold tuning completed!")
