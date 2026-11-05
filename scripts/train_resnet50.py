import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights


# 1. Configuration

CSV_PATH = "data/processed/model_image_manifest_split.csv"

LABELS = [
    "Normal",
    "Opacity",
    "Cardiomegaly",
    "Atelectasis"
]

BATCH_SIZE = 8
EPOCHS = 8
LEARNING_RATE = 0.00002
PATIENCE = 3

MODEL_DIR = "outputs/models"
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR, "best_resnet50.pth"
)

HISTORY_PATH = "outputs/training_history_resnet50.csv"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# 2. Device

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Selected device:", device)


# 3. Image transformations

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(7),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 4. Dataset

class ChestXrayDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            row[LABELS].to_numpy(dtype=np.float32),
            dtype=torch.float32
        )

        return image, labels


# 5. Load split dataset

df = pd.read_csv(CSV_PATH)

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()

train_dataset = ChestXrayDataset(
    train_df, train_transform
)

val_dataset = ChestXrayDataset(
    val_df, val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nDATASET DETAILS")
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# 6. Calculate class weights from training data only

positive_counts = train_df[LABELS].sum().to_numpy(
    dtype=np.float32
)

negative_counts = len(train_df) - positive_counts

pos_weights = negative_counts / np.maximum(
    positive_counts, 1
)

pos_weights = torch.tensor(
    pos_weights,
    dtype=torch.float32,
    device=device
)

print("\nCLASS WEIGHTS")
for label, weight in zip(LABELS, pos_weights.tolist()):
    print(label, ":", round(weight, 3))


# 7. Load pretrained ResNet50

print("\nLoading pretrained ResNet50...")

model = models.resnet50(
    weights=ResNet50_Weights.DEFAULT
)

num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(num_features, len(LABELS))
)

model = model.to(device)

print("ResNet50 model loaded successfully.")


# 8. Loss and optimizer

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)


# 9. Training function

def train_one_epoch():

    model.train()
    total_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(train_dataset)


# 10. Validation function

def validate():

    model.eval()
    total_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

    return total_loss / len(val_dataset)


# 11. Training loop with early stopping

best_val_loss = float("inf")
epochs_without_improvement = 0
history = []

print("\n========== RESNET50 TRAINING STARTED ==========")

for epoch in range(1, EPOCHS + 1):

    train_loss = train_one_epoch()
    val_loss = validate()

    print(
        f"Epoch [{epoch}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Validation Loss: {val_loss:.4f}"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    pd.DataFrame(history).to_csv(
        HISTORY_PATH, index=False
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "best_val_loss": best_val_loss,
            "epoch": epoch,
            "labels": LABELS,
            "architecture": "ResNet50",
            "pos_weights": pos_weights.cpu()
        }, BEST_MODEL_PATH)

        print("Best ResNet50 checkpoint saved.")

    else:
        epochs_without_improvement += 1
        print(
            "No improvement:",
            epochs_without_improvement,
            "/",
            PATIENCE
        )

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


print("\n========== TRAINING COMPLETED ==========")
print("Best validation loss:", round(best_val_loss, 4))
print("Best checkpoint:", BEST_MODEL_PATH)
print("Training history:", HISTORY_PATH)
