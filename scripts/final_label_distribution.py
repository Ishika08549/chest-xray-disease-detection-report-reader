import pandas as pd

path = "outputs/label_analysis/report_label_flags.csv"
df = pd.read_csv(path)

disease_labels = [
    "Normal",
    "Cardiomegaly",
    "Atelectasis",
    "Pleural_Effusion",
    "Pneumonia",
    "Pneumothorax",
    "Emphysema",
    "Opacity"
]

print("TOTAL UNIQUE REPORTS:", df["report_id"].nunique())

print("\nDISEASE LABEL DISTRIBUTION")

for label in disease_labels:
    count = df[label].sum()
    percentage = (count / len(df)) * 100

    print(f"{label:22} {count:5} reports ({percentage:.2f}%)")

selected = df[disease_labels].sum(axis=1)

print("\nREPORTS WITH NO SELECTED LABEL:", (selected == 0).sum())
print("REPORTS WITH ONE SELECTED LABEL:", (selected == 1).sum())
print("REPORTS WITH MULTIPLE SELECTED LABELS:", (selected > 1).sum())