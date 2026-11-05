from pathlib import Path

import pandas as pd


# Project configuration

PROJECT_ROOT = Path(__file__).resolve().parents[3]

REPORT_DATASET_PATH = (
    PROJECT_ROOT / "data/processed/image_report_dataset.csv"
)


# Load report dataset once and reuse it

_REPORT_DATA = None


def get_report_data():

    global _REPORT_DATA

    if _REPORT_DATA is None:

        if not REPORT_DATASET_PATH.exists():
            raise FileNotFoundError(
                f"Report dataset not found: {REPORT_DATASET_PATH}"
            )

        _REPORT_DATA = pd.read_csv(
            REPORT_DATASET_PATH,
            dtype={"image_id": str, "report_id": str}
        )

    return _REPORT_DATA


# Retrieve report using image ID

def get_report_by_image_id(image_id):

    if not image_id or not image_id.strip():
        raise ValueError("Image ID cannot be empty.")

    df = get_report_data()

    matched_rows = df[
        df["image_id"] == image_id.strip()
    ]

    if matched_rows.empty:
        raise LookupError(
            f"No report found for image ID: {image_id}"
        )

    if len(matched_rows) != 1:
        raise ValueError(
            "Multiple records found for this image ID."
        )

    record = matched_rows.iloc[0]

    mapping_verified = record["mapping_verified"]

    if pd.isna(mapping_verified) or not bool(mapping_verified):
        raise PermissionError(
            "Report mapping is not verified."
        )

    if record["parse_status"] != "success":
        raise ValueError(
            "The associated radiology report was not parsed successfully."
        )

    def clean_value(value):
        if pd.isna(value):
            return None
        return str(value).strip()

    return {
        "image_id": clean_value(record["image_id"]),
        "report_id": clean_value(record["report_id"]),
        "report_uid": clean_value(record["report_uid"]),
        "indication": clean_value(record["indication"]),
        "comparison": clean_value(record["comparison"]),
        "findings": clean_value(record["findings"]),
        "impression": clean_value(record["impression"]),
        "mesh_terms": clean_value(record["mesh_terms"]),
        "mapping_verified": True,
        "parse_status": "success"
    }
