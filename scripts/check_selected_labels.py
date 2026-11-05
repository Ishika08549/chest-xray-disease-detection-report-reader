import pandas as pd

path = "outputs/label_analysis/report_label_flags.csv"
df = pd.read_csv(path)

selected_labels = [
    "Normal",
    "Opacity",
    "Cardiomegaly",
    "Atelectasis"
]

# Check selected label distribution
print("\nSELECTED LABEL DISTRIBUTION")

for label in selected_labels:
    count = df[label].sum()
    percentage = (count / len(df)) * 100
    print(f"{label:20} {count:5} ({percentage:.2f}%)")

# Count selected labels per report
df["selected_label_count"] = df[selected_labels].sum(axis=1)

print("\nREPORT OVERLAP")
print("No selected labels:",
      (df["selected_label_count"] == 0).sum())

print("Exactly one selected label:",
      (df["selected_label_count"] == 1).sum())

print("Multiple selected labels:",
      (df["selected_label_count"] > 1).sum())

# Find which labels occur together
print("\nLABEL OVERLAP MATRIX")
print(df[selected_labels].astype(int).T.dot(
    df[selected_labels].astype(int)
))

# Save results
df.to_csv(
    "outputs/label_analysis/selected_label_analysis.csv",
    index=False
)

print("\nSaved: outputs/label_analysis/selected_label_analysis.csv")