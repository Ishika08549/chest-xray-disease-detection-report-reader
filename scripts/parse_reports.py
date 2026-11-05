
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = PROJECT_ROOT / "data" / "raw" / "reports"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "image_report_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_section_text(root, label):
    """Extract text from a specific AbstractText section."""
    values = []

    for element in root.iter():
        if element.tag.split("}")[-1] == "AbstractText":
            section_label = element.attrib.get("Label", "").strip().upper()

            if section_label == label.upper():
                text = " ".join(element.itertext()).strip()

                if text:
                    values.append(text)

    return " ".join(values)


def get_mesh_terms(root):
    """Extract MeSH terms from an XML report."""
    terms = []

    for element in root.iter():
        if element.tag.split("}")[-1] in ("major", "minor"):
            term = " ".join(element.itertext()).strip()

            if term:
                terms.append(term)

    return sorted(set(terms))


def parse_report(xml_path):
    """Read one XML report and return structured information."""
    try:
        root = ET.parse(xml_path).getroot()

        report_id = xml_path.stem

        uid = root.find(".//uId")
        report_uid = uid.attrib.get("id", "") if uid is not None else ""

        return {
            "report_id": report_id,
            "report_uid": report_uid,
            "comparison": get_section_text(root, "COMPARISON"),
            "indication": get_section_text(root, "INDICATION"),
            "findings": get_section_text(root, "FINDINGS"),
            "impression": get_section_text(root, "IMPRESSION"),
            "mesh_terms": "|".join(get_mesh_terms(root)),
            "parse_status": "success",
        }

    except (ET.ParseError, OSError) as error:
        return {
            "report_id": xml_path.stem,
            "report_uid": "",
            "comparison": "",
            "indication": "",
            "findings": "",
            "impression": "",
            "mesh_terms": "",
            "parse_status": f"error: {error}",
        }


def main():
    print("Starting XML report extraction...")

    xml_files = sorted(REPORTS_DIR.rglob("*.xml"))

    print(f"XML files found: {len(xml_files)}")

    records = [parse_report(path) for path in xml_files]

    reports_df = pd.DataFrame(records)

    reports_output = OUTPUT_DIR / "radiology_reports.csv"
    reports_df.to_csv(reports_output, index=False)

    print("\nReport extraction complete.")
    print(f"Reports processed: {len(reports_df)}")
    print(
        "Successful reports:",
        reports_df["parse_status"].eq("success").sum()
    )
    print(
        "Failed reports:",
        reports_df["parse_status"].ne("success").sum()
    )
    print(
        "Reports with findings:",
        reports_df["findings"].fillna("").str.strip().ne("").sum()
    )
    print(
        "Reports with impression:",
        reports_df["impression"].fillna("").str.strip().ne("").sum()
    )

    manifest_df = pd.read_csv(MANIFEST_PATH,
    dtype={"report_id": "string"})
    
    reports_df["report_id"] = reports_df["report_id"].astype("string")

    dataset_df = manifest_df.merge(
        reports_df,
        on="report_id",
        how="left",
        validate="many_to_one"
    )

    dataset_output = OUTPUT_DIR / "image_report_dataset.csv"
    dataset_df.to_csv(dataset_output, index=False)

    print("\nImage-report dataset created.")
    print(f"Dataset rows: {len(dataset_df)}")
    print(f"Unique images: {dataset_df['image_id'].nunique()}")
    print(f"Unique reports: {dataset_df['report_id'].nunique()}")
    print(
        "Rows missing impressions:",
        dataset_df["impression"].isna().sum()
    )

    print(f"\nSaved files:\n{reports_output}\n{dataset_output}")

    print("\nSample extracted report:")
    print(
        reports_df[
            ["report_id", "findings", "impression", "mesh_terms"]
        ].head(3).to_string(index=False)
    )


if __name__ == "__main__":
    main()
