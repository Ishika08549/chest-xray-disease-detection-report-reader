
import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image


# 1. Configuration

CSV_PATH = "data/processed/model_image_manifest_split.csv"

LABELS = [
    "Normal",
    "Opacity",
    "Cardiomegaly",
    "Atelectasis"
]

BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.0001

MODEL_DIR = "outputs/models"
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR, "best_resnet18.pth"
)


# 2. Reproducibility

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Selected device:", device)


# 3. Image preprocessing

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 4. Custom Dataset

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
            row[LABELS].to_numpy(dtype="float32"),
            dtype=torch.float32
        )

        return image, labels


# 5. Load dataset

df = pd.read_csv(CSV_PATH)

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "validation"]

train_dataset = ChestXrayDataset(
    train_df, image_transform
)

val_dataset = ChestXrayDataset(
    val_df, image_transform
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


# 6. Load pretrained ResNet18

weights = ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(LABELS)
)

model = model.to(device)


# 7. Loss function and optimizer

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# 8. Training function

def train_one_epoch(model, loader):

    model.train()

    total_loss = 0.0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


# 9. Validation function

def validate(model, loader):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


# 10. Training loop

best_val_loss = float("inf")

history = []

print("\nSTARTING RESNET18 TRAINING")

for epoch in range(EPOCHS):

    train_loss = train_one_epoch(
        model, train_loader
    )

    val_loss = validate(
        model, val_loader
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Validation Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "labels": LABELS,
                "best_val_loss": best_val_loss
            },
            BEST_MODEL_PATH
        )

        print("Best model saved!")


# 11. Save training history

history_df = pd.DataFrame(history)

history_df.to_csv(
    "outputs/training_history.csv",
    index=False
)

print("\nTRAINING COMPLETED")

print("Best validation loss:", best_val_loss)
print("Saved model:", BEST_MODEL_PATH)
print("Saved history: outputs/training_history.csv")
