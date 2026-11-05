
import os
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
EPOCHS = 12
LEARNING_RATE = 0.00005
PATIENCE = 3

MODEL_DIR = "outputs/models"
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR, "best_resnet18_v2.pth"
)


# 2. Reproducibility

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# 3. Device selection

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Selected device:", device)


# 4. Image transformations

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


# 5. Custom Dataset

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


# 6. Load dataset

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


# 7. Calculate class weights

positive_counts = train_df[LABELS].sum().to_numpy(
    dtype=np.float32
)

negative_counts = (
    len(train_df) - positive_counts
)

pos_weights = negative_counts / np.maximum(
    positive_counts, 1
)

pos_weights = torch.tensor(
    pos_weights,
    dtype=torch.float32,
    device=device
)

print("\nPOSITIVE CLASS WEIGHTS")
for label, weight in zip(LABELS, pos_weights.tolist()):
    print(label, ":", round(weight, 3))


# 8. Load pretrained ResNet18

model = models.resnet18(
    weights=ResNet18_Weights.DEFAULT
)

num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(num_features, len(LABELS))
)

model = model.to(device)

# Resume from the best saved V2 checkpoint

START_EPOCH = 2

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

best_val_loss = checkpoint["best_val_loss"]

print("\nRESUMING FROM SAVED CHECKPOINT")
print("Previous best validation loss:", best_val_loss)
print("Starting from epoch:", START_EPOCH + 1)


# 9. Loss and optimizer

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)


# 10. Training function

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


# 11. Validation function

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


# 12. Training loop with early stopping

epochs_without_improvement = 0

history = [
    {
        "epoch": 1,
        "train_loss": 0.9120,
        "val_loss": 0.7754
    },
    {
        "epoch": 2,
        "train_loss": 0.7581,
        "val_loss": 0.7317
    }
]

print("\nSTARTING RESNET18 V2 TRAINING")

for epoch in range(START_EPOCH, EPOCHS):

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
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "labels": LABELS,
                "best_val_loss": best_val_loss
            },
            BEST_MODEL_PATH
        )

        print("Best V2 model saved!")

    else:

        epochs_without_improvement += 1

        print(
            "No improvement:",
            epochs_without_improvement,
            "/",
            PATIENCE
        )

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered!")
        break


# 13. Save training history

history_df = pd.DataFrame(history)

history_df.to_csv(
    "outputs/training_history_v2.csv",
    index=False
)

print("\nV2 TRAINING COMPLETED")
print("Best validation loss:", best_val_loss)
print("Saved model:", BEST_MODEL_PATH)
print("Saved history: outputs/training_history_v2.csv")
