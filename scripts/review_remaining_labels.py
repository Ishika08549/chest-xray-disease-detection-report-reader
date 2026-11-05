
import pandas as pd

path = "outputs/label_analysis/label_text_comparison.csv"
df = pd.read_csv(path).fillna("")

categories = [
    "Pleural Effusion",
    "Pneumothorax",
    "Emphysema",
    "Opacity"
]

for category in categories:
    print("\n" + "=" * 75)
    print("CATEGORY:", category)
    print("=" * 75)

    sample = df[df["candidate_category"] == category]

    for _, row in sample.iterrows():
        print("\nReport ID:", row["report_id"])
        print("MeSH:", row["mesh_terms"])
        print("Keyword match:", row["text_keyword_match"])
        print("Findings:", row["findings"][:400])
        print("Impression:", row["impression"][:300])

print("\nREMAINING LABEL REVIEW COMPLETED")
