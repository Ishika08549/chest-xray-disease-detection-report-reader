from pathlib import Path
import csv
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images"
REPORT_DIR = PROJECT_ROOT / "data" / "raw" / "reports"
OUTPUT_DIR = PROJECT_ROOT / "data" / "manifests"

OUTPUT_FILE = OUTPUT_DIR / "image_report_manifest.csv"


def get_image_references(xml_file):
    """Extract image IDs referenced by an XML report."""
    try:
        root = ET.parse(xml_file).getroot()
    except ET.ParseError as error:
        print(f"Could not parse {xml_file.name}: {error}")
        return []

    image_ids = []

    for element in root.iter():
        tag_name = element.tag.split("}")[-1]

        if tag_name.lower() == "parentimage":
            image_id = element.attrib.get("id")

            if image_id:
                image_id = Path(image_id).name
                image_id = image_id.removesuffix(".png")
                image_ids.append(image_id)

    return sorted(set(image_ids))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(REPORT_DIR.rglob("*.xml"))
    rows = []
    missing_images = 0
    reports_without_references = 0

    for xml_file in xml_files:
        image_ids = get_image_references(xml_file)

        if not image_ids:
            reports_without_references += 1

        for image_id in image_ids:
            image_path = IMAGE_DIR / f"{image_id}.png"
            image_exists = image_path.is_file()

            if not image_exists:
                missing_images += 1

            rows.append({
                "report_id": xml_file.stem,
                "image_id": image_id,
                "image_path": (
                    str(image_path.relative_to(PROJECT_ROOT))
                    if image_exists else ""
                ),
                "report_path": str(
                    xml_file.relative_to(PROJECT_ROOT)
                ),
                "mapping_verified": image_exists,
            })

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        fieldnames = [
            "report_id",
            "image_id",
            "image_path",
            "report_path",
            "mapping_verified",
        ]

        writer = csv.DictWriter(
            csv_file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nManifest generation complete")
    print(f"XML reports found: {len(xml_files)}")
    print(f"Image-report references: {len(rows)}")
    print(f"References without matching image files: {missing_images}")
    print(f"Reports without image references: {reports_without_references}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()