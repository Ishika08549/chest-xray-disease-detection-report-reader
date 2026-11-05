
import pandas as pd

CSV_PATH = "data/processed/model_image_manifest_split.csv"

LABELS = [
    "Normal",
    "Opacity",
    "Cardiomegaly",
    "Atelectasis"
]

df = pd.read_csv(CSV_PATH)

print("\nDATASET DISTRIBUTION CHECK")

for split_name in ["train", "validation", "test"]:

    split_df = df[df["split"] == split_name]

    print(f"\n{'=' * 45}")
    print(f"{split_name.upper()} DATASET")
    print("Total images:", len(split_df))

    for label in LABELS:

        positive = int(split_df[label].sum())
        negative = len(split_df) - positive

        percentage = (
            positive / len(split_df) * 100
            if len(split_df) > 0 else 0
        )

        print(f"\nLabel: {label}")
        print("Positive images:", positive)
        print("Negative images:", negative)
        print("Positive percentage:", round(percentage, 2), "%")

print("\nDISTRIBUTION CHECK COMPLETED")
