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
CHECKPOINT_PATH = "outputs/models/best_resnet50.pth"

LABELS = [
    "Normal",
    "Opacity",
    "Cardiomegaly",
    "Atelectasis"
]

BATCH_SIZE = 8
OUTPUT_DIR = "outputs/evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Selected device:", device)

# 2. Image preprocessing

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

# 4. Build ResNet50

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
print("Best epoch:", checkpoint["epoch"])

# 5. Generate validation predictions

all_labels = []
all_probabilities = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.sigmoid(outputs)

        all_labels.append(labels.numpy())
        all_probabilities.append(probabilities.cpu().numpy())

y_true = np.vstack(all_labels)
y_prob = np.vstack(all_probabilities)

# 6. Tune threshold separately for each label

results = []

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

    original_pred = (y_prob[:, i] >= 0.5).astype(int)
    tuned_pred = (y_prob[:, i] >= best_threshold).astype(int)

    results.append({
        "Model": "ResNet50",
        "Label": label,
        "Threshold_0.5_F1": f1_score(
            y_true[:, i],
            original_pred,
            zero_division=0
        ),
        "Best_Threshold": best_threshold,
        "Tuned_Precision": precision_score(
            y_true[:, i],
            tuned_pred,
            zero_division=0
        ),
        "Tuned_Recall": recall_score(
            y_true[:, i],
            tuned_pred,
            zero_division=0
        ),
        "Tuned_F1": f1_score(
            y_true[:, i],
            tuned_pred,
            zero_division=0
        )
    })

    print(
        f"{label}: threshold={best_threshold:.4f}, "
        f"F1={results[-1]['Tuned_F1']:.4f}"
    )

# 7. Save separate results

results_df = pd.DataFrame(results)

output_path = os.path.join(
    OUTPUT_DIR,
    "resnet50_threshold_results.csv"
)

results_df.to_csv(output_path, index=False)

print("\nRESNET50 THRESHOLD RESULTS")
print(results_df.round(4).to_string(index=False))

print("\nSaved:", output_path)
print("ResNet50 threshold tuning completed!")
