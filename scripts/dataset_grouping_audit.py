
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/processed/image_report_dataset.csv")

print("TOTAL IMAGE RECORDS:", len(df))
print("UNIQUE IMAGES:", df["image_id"].nunique())
print("UNIQUE REPORTS:", df["report_id"].nunique())

print("\nMISSING REPORT IDs:", df["report_id"].isna().sum())
print("DUPLICATE IMAGE IDs:", df["image_id"].duplicated().sum())

# Number of images linked to each report
images_per_report = df.groupby("report_id")["image_id"].nunique()

print("\nIMAGES PER REPORT:")
print(images_per_report.value_counts().sort_index().to_string())

print("\nREPORTS WITH MULTIPLE IMAGES:",
      (images_per_report > 1).sum())

print("MAXIMUM IMAGES PER REPORT:",
      images_per_report.max())

# Check duplicate report-image combinations
duplicate_pairs = df.duplicated(
    subset=["report_id", "image_id"]
).sum()

print("\nDUPLICATE REPORT-IMAGE PAIRS:", duplicate_pairs)

# Save summary
summary = images_per_report.value_counts().sort_index()
summary.to_csv("outputs/eda/images_per_report_summary.csv")

print("\nSaved: outputs/eda/images_per_report_summary.csv")
print("\nGROUPING AUDIT COMPLETED")
