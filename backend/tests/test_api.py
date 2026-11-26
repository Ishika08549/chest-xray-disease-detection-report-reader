from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from backend.app import main as main_module


client = TestClient(main_module.app)


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
    response = client.get("/report/CXR1_1_IM-0001-3001")

    assert response.status_code == 200

    data = response.json()["report"]

    assert data["image_id"] == "CXR1_1_IM-0001-3001"
    assert data["mapping_verified"] is True
    assert data["parse_status"] == "success"
    assert data["findings"] is not None
    assert data["impression"] is not None



def test_report_explanation_endpoint(monkeypatch):
    expected = {
        "image_id": "CXR1_1_IM-0001-3001",
        "original_report": {
            "findings": "Original findings",
            "impression": "Original impression"
        },
        "patient_friendly_explanation": {
            "summary": "Educational summary",
            "sections_explained": [],
            "medical_terms": [],
            "patient_guidance": "Discuss the original report with a qualified doctor.",
            "prescription": None,
            "limitations": ["Educational information, not a diagnosis."]
        },
        "retrieved_sources": [
            {
                "topic": "example",
                "source": "Example source",
                "url": "https://example.org",
                "similarity": 0.8
            }
        ]
    }

    def fake_explain_report(report):
        assert report["image_id"] == "CXR1_1_IM-0001-3001"
        return expected

    monkeypatch.setattr(main_module, "explain_report", fake_explain_report)

    response = client.post("/report/CXR1_1_IM-0001-3001/explain")

    assert response.status_code == 200
    data = response.json()
    assert data["image_id"] == expected["image_id"]
    assert data["patient_friendly_explanation"]["prescription"] is None
    assert len(data["retrieved_sources"]) == 1
    assert "not a diagnosis" in data["disclaimer"].lower()


def test_report_explanation_handles_lm_studio_unavailable(monkeypatch):
    def fake_explain_report(report):
        raise RuntimeError("LM Studio unavailable")

    monkeypatch.setattr(main_module, "explain_report", fake_explain_report)

    response = client.post("/report/CXR1_1_IM-0001-3001/explain")

    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"].lower()


def test_invalid_report_endpoint():
    response = client.get("/report/INVALID_IMAGE_123")
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


def test_predict_success(monkeypatch):
    expected_predictions = [
        {
            "label": "Normal",
            "probability": 0.9757,
            "probability_percent": 97.57,
            "candidate_detected": True
        }
    ]

    def fake_predict_image(image_bytes):
        return expected_predictions

    monkeypatch.setattr(
        main_module,
        "predict_image",
        fake_predict_image
    )

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.png",
                create_test_image(),
                "image/png"
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == "test.png"
    assert data["predictions"] == expected_predictions
    assert "not a medical diagnosis" in data["disclaimer"]


def test_gradcam_success(monkeypatch):
    expected_image = create_test_image().getvalue()

    def fake_generate_gradcam(image_bytes, selected_label):
        assert selected_label == "Normal"
        return expected_image, 0.9757

    monkeypatch.setattr(
        main_module,
        "generate_gradcam_image",
        fake_generate_gradcam
    )

    response = client.post(
        "/gradcam",
        data={"selected_label": "Normal"},
        files={
            "file": (
                "test.png",
                create_test_image(),
                "image/png"
            )
        }
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-gradcam-probability"] == "0.9757"
    assert response.content == expected_image


def test_predict_rejects_invalid_image_content():
    response = client.post(
        "/predict",
        files={
            "file": (
                "invalid.png",
                b"this is not a real image",
                "image/png"
            )
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a valid image."


def test_predict_rejects_file_over_10_mb():
    large_file = b"x" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/predict",
        files={
            "file": (
                "large.png",
                large_file,
                "image/png"
            )
        }
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Image size must be below 10 MB."


def test_gradcam_rejects_file_over_10_mb():
    large_file = b"x" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/gradcam",
        data={"selected_label": "Normal"},
        files={
            "file": (
                "large.png",
                large_file,
                "image/png"
            )
        }
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Image size must be below 10 MB."
