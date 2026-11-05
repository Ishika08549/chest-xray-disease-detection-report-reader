from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app


client = TestClient(app)


def create_test_image():
    image = Image.new("RGB", (224, 224), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "API is running" in response.json()["message"]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_valid_report_endpoint():
    response = client.get(
        "/report/CXR1_1_IM-0001-3001"
    )

    assert response.status_code == 200

    data = response.json()["report"]

    assert data["image_id"] == "CXR1_1_IM-0001-3001"
    assert data["mapping_verified"] is True
    assert data["parse_status"] == "success"
    assert data["findings"] is not None
    assert data["impression"] is not None


def test_invalid_report_endpoint():
    response = client.get(
        "/report/INVALID_IMAGE_123"
    )

    assert response.status_code == 404


def test_predict_rejects_wrong_file_type():
    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                b"not an image",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400


def test_gradcam_rejects_invalid_label():
    response = client.post(
        "/gradcam",
        data={"selected_label": "InvalidLabel"},
        files={
            "file": (
                "test.png",
                create_test_image(),
                "image/png"
            )
        }
    )

    assert response.status_code == 400
