import pandas as pd

# File paths
manifest_path = "data/manifests/image_report_manifest.csv"
labels_path = "outputs/label_analysis/report_label_flags.csv"

output_path = "data/processed/model_image_manifest.csv"

# Selected labels
selected_labels = [
    "Normal",
    "Opacity",
    "Cardiomegaly",
    "Atelectasis"
]

# Load datasets
images = pd.read_csv(manifest_path, dtype={"report_id": str})
labels = pd.read_csv(labels_path, dtype={"report_id": str})

# Keep only required columns
labels = labels[["report_id"] + selected_labels]

# Attach report labels to every associated image
model_df = images.merge(
    labels,
    on="report_id",
    how="inner",
    validate="many_to_one"
)

# Count selected positive labels for each image
model_df["selected_label_count"] = (
    model_df[selected_labels].sum(axis=1)
)

# Keep reports having at least one selected label
model_df = model_df[
    model_df["selected_label_count"] > 0
].copy()

# Save image-level dataset
model_df.to_csv(output_path, index=False)

# Display summary
print("\nIMAGE-LEVEL MODEL DATASET")
print("Total images:", len(model_df))
print("Unique reports:", model_df["report_id"].nunique())

print("\nLABEL DISTRIBUTION (IMAGE LEVEL)")

for label in selected_labels:
    count = model_df.groupby("report_id")[label].first().sum()
    print(f"{label:20}: {count} reports")

print("\nIMAGES PER REPORT")
print(model_df.groupby("report_id").size().value_counts().sort_index())

print("\nSAVED:", output_path)