import pandas as pd
import os
import torch
import torchvision
from PIL import Image

# Load split dataset
path = "data/processed/model_image_manifest_split.csv"
df = pd.read_csv(path)

print("\nPYTORCH ENVIRONMENT")
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("MPS available:", torch.backends.mps.is_available())

print("\nDATASET CHECK")
print("Total images:", len(df))
print("Missing image paths:", df["image_path"].isna().sum())

# Verify image files
existing = df["image_path"].apply(os.path.exists)

print("Existing images:", existing.sum())
print("Missing image files:", (~existing).sum())

# Open a sample image
sample_path = df.loc[existing, "image_path"].iloc[0]

with Image.open(sample_path) as img:
    print("\nSAMPLE IMAGE")
    print("Image format:", img.format)
    print("Image size:", img.size)
    print("Image mode:", img.mode)

# Check split distribution
print("\nSPLIT DISTRIBUTION")
print(df["split"].value_counts())

print("\nLABEL DISTRIBUTION BY SPLIT")

labels = [
    "Normal",
    "Opacity",
    "Cardiomegaly",
    "Atelectasis"
]

for split in ["train", "validation", "test"]:
    print(f"\n{split.upper()}")

    report_labels = (
        df[df["split"] == split]
        .groupby("report_id")[labels]
        .first()
    )

    print(report_labels.mean().round(3))

print("\nTRAINING SETUP CHECK COMPLETED")