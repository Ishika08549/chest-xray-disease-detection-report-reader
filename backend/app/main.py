from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import UnidentifiedImageError

from backend.app.services.model_service import (
    LABELS,
    predict_image,
    generate_gradcam_image
)

from backend.app.services.report_service import (
    get_report_by_image_id
)


app = FastAPI(
    title="Chest X-Ray Disease Detection API",
    description="Educational chest X-ray prediction and radiology report reader",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def home():
    return {
        "message": "Chest X-Ray Disease Detection API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Backend is working properly"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    allowed_types = ["image/png", "image/jpeg", "image/jpg"]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PNG or JPEG image."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image size must be below 10 MB."
        )

    try:
        predictions = predict_image(image_bytes)

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image."
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again."
        )

    return {
        "filename": file.filename,
        "predictions": predictions,
        "disclaimer": "Educational research output only. This is not a medical diagnosis."
    }


@app.post("/gradcam")
async def gradcam(
    file: UploadFile = File(...),
    selected_label: str = Form(...)
):

    allowed_types = ["image/png", "image/jpeg", "image/jpg"]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PNG or JPEG image."
        )

    if selected_label not in LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"Choose one of these labels: {LABELS}"
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image size must be below 10 MB."
        )

    try:
        heatmap_bytes, probability = generate_gradcam_image(
            image_bytes,
            selected_label
        )

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image."
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Grad-CAM generation failed."
        )

    return StreamingResponse(
        BytesIO(heatmap_bytes),
        media_type="image/png",
        headers={
            "Content-Disposition": "inline; filename=gradcam.png",
            "X-GradCAM-Probability": str(round(probability, 4)),
            "X-GradCAM-Disclaimer": "Visualization only; not a diagnosis"
        }
    )


@app.get("/report/{image_id}")
def get_radiology_report(image_id: str):

    try:
        report = get_report_by_image_id(image_id)

        return {
            "report": report,
            "disclaimer": (
                "Educational research use only. "
                "This report is retrieved from the associated dataset "
                "and is not a new clinical interpretation."
            )
        }

    except LookupError:
        raise HTTPException(
            status_code=404,
            detail="No report found for this image ID."
        )

    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail="Report mapping is not verified."
        )

    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error)
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve radiology report."
        )
