import pandas as pd
import numpy as np

# File paths
input_path = "data/processed/model_image_manifest.csv"
output_path = "data/processed/model_image_manifest_split.csv"

# Selected labels
labels = [
    "Normal",
    "Opacity",
    "Cardiomegaly",
    "Atelectasis"
]

# Load dataset
df = pd.read_csv(input_path, dtype={"report_id": str})

# Create one row per report for splitting
report_df = df.groupby("report_id")[labels].first()

# Check whether labels are consistent across images
label_consistency = df.groupby("report_id")[labels].nunique()

if (label_consistency > 1).any().any():
    raise ValueError("Some reports have inconsistent labels!")

# Shuffle unique report IDs
report_ids = report_df.index.to_numpy(copy=True)

rng = np.random.default_rng(42)
rng.shuffle(report_ids)

# Split reports: 70% train, 15% validation, 15% test
n = len(report_ids)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_ids = report_ids[:train_end]
val_ids = report_ids[train_end:val_end]
test_ids = report_ids[val_end:]

# Assign split to each image
split_mapping = {}

for report_id in train_ids:
    split_mapping[report_id] = "train"

for report_id in val_ids:
    split_mapping[report_id] = "validation"

for report_id in test_ids:
    split_mapping[report_id] = "test"

df["split"] = df["report_id"].map(split_mapping)

# Verify no report leakage
train_set = set(train_ids)
val_set = set(val_ids)
test_set = set(test_ids)

assert train_set.isdisjoint(val_set)
assert train_set.isdisjoint(test_set)
assert val_set.isdisjoint(test_set)

# Save dataset
df.to_csv(output_path, index=False)

# Display results
print("\nDATASET SPLIT SUMMARY")

for split_name in ["train", "validation", "test"]:
    split_df = df[df["split"] == split_name]

    print(f"\n{split_name.upper()}")
    print("Images:", len(split_df))
    print("Unique reports:", split_df["report_id"].nunique())

    report_labels = split_df.groupby("report_id")[labels].first()

    print("Positive report labels:")
    for label in labels:
        print(f"  {label}: {report_labels[label].sum()}")

print("\nTOTAL IMAGES:", len(df))
print("TOTAL REPORTS:", df["report_id"].nunique())
print("\nReport leakage check: PASSED")
print("Saved:", output_path)