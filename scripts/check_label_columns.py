
import pandas as pd

path = "outputs/label_analysis/report_label_flags.csv"
df = pd.read_csv(path)

print("\nALL COLUMN NAMES:")
for col in df.columns:
    print(repr(col))

print("\nDATASET SHAPE:", df.shape)

print("\nFIRST 5 ROWS:")
print(df.head().to_string())

print("\nDATA TYPES:")
print(df.dtypes.to_string())
