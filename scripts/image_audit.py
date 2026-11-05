
import pandas as pd
from PIL import Image
from pathlib import Path
from collections import Counter

df = pd.read_csv("data/processed/image_report_dataset.csv")

dimension_count = Counter()
mode_count = Counter()
corrupted_images = []
missing_images = []

unique_paths = df["image_path"].dropna().unique()

for path in unique_paths:
    image_path = Path(path)

    if not image_path.exists():
        missing_images.append(str(image_path))
        continue

    try:
        with Image.open(image_path) as img:
            img.verify()

        with Image.open(image_path) as img:
            dimension_count[img.size] += 1
            mode_count[img.mode] += 1

    except Exception as error:
        corrupted_images.append(str(image_path))

print("\nTOTAL UNIQUE IMAGE PATHS:", len(unique_paths))

print("\nIMAGE DIMENSIONS:")
for dimension, count in dimension_count.most_common():
    print(dimension, ":", count)

print("\nIMAGE MODES:")
for mode, count in mode_count.items():
    print(mode, ":", count)

print("\nMISSING IMAGES:", len(missing_images))
print("CORRUPTED IMAGES:", len(corrupted_images))

if missing_images:
    print("\nMissing examples:", missing_images[:5])

if corrupted_images:
    print("\nCorrupted examples:", corrupted_images[:5])

print("\nIMAGE AUDIT COMPLETED")
