import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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


# 2. Image preprocessing
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 3. Custom Dataset
class ChestXrayDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = row["image_path"]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply preprocessing
        if self.transform:
            image = self.transform(image)

        # Get four labels
        labels = torch.tensor(
            row[LABELS].to_numpy(dtype="float32"),
            dtype=torch.float32
        )

        return image, labels


# 4. Load CSV
df = pd.read_csv(CSV_PATH)

# 5. Create separate datasets
train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "validation"]
test_df = df[df["split"] == "test"]

train_dataset = ChestXrayDataset(
    train_df, image_transform
)

val_dataset = ChestXrayDataset(
    val_df, image_transform
)

test_dataset = ChestXrayDataset(
    test_df, image_transform
)


# 6. Create DataLoaders
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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# 7. Test one training batch
images, labels = next(iter(train_loader))

print("\nDATALOADER VERIFICATION")

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Testing images:", len(test_dataset))

print("\nFIRST BATCH")
print("Image tensor shape:", images.shape)
print("Label tensor shape:", labels.shape)

print("\nIMAGE DETAILS")
print("Image data type:", images.dtype)
print("Image minimum:", images.min().item())
print("Image maximum:", images.max().item())

print("\nLABEL DETAILS")
print("Label data type:", labels.dtype)
print("First 5 label rows:")
print(labels[:5])

print("\nDATALOADER WORKING SUCCESSFULLY!")