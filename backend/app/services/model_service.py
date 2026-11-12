from pathlib import Path
import os
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms


# Project configuration

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(PROJECT_ROOT / "outputs/models/best_resnet50.pth")
    )
)

LABELS = [
    "Normal",
    "Opacity",
    "Cardiomegaly",
    "Atelectasis"
]

THRESHOLDS = np.array([
    0.3548726737,
    0.4505716562,
    0.6367863417,
    0.7514647245
])

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# Image preprocessing

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Model builder

def build_model():

    model = models.resnet50(weights=None)

    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, len(LABELS))
    )

    return model


# Load model once and reuse it

_MODEL = None


def get_model():

    global _MODEL

    if _MODEL is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {MODEL_PATH}"
            )

        model = build_model()

        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model = model.to(DEVICE)
        model.eval()

        _MODEL = model

    return _MODEL


# Prediction function

def predict_image(image_bytes):

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    model = get_model()

    with torch.inference_mode():

        outputs = model(image_tensor)

        probabilities = torch.sigmoid(
            outputs
        ).cpu().numpy()[0]

    predictions = (
        probabilities >= THRESHOLDS
    ).astype(int)

    results = []

    for label, probability, prediction in zip(
        LABELS,
        probabilities,
        predictions
    ):

        results.append({
            "label": label,
            "probability": round(float(probability), 4),
            "probability_percent": round(
                float(probability * 100), 2
            ),
            "candidate_detected": bool(prediction)
        })

    return results


# Grad-CAM generation function

def generate_gradcam_image(image_bytes, selected_label):

    if selected_label not in LABELS:
        raise ValueError(
            f"Choose one of these labels: {LABELS}"
        )

    from matplotlib import colormaps

    original_image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    display_image = original_image.resize((224, 224))

    input_tensor = transform(
        original_image
    ).unsqueeze(0).to(DEVICE)

    model = get_model()

    activations = []
    gradients = []

    target_layer = model.layer4[-1]

    def forward_hook(module, inputs, output):

        activations.append(output)

        output.register_hook(
            lambda grad: gradients.append(grad)
        )

    hook = target_layer.register_forward_hook(
        forward_hook
    )

    try:

        model.zero_grad(set_to_none=True)

        outputs = model(input_tensor)

        probabilities = torch.sigmoid(outputs)[0]

        label_index = LABELS.index(selected_label)

        target_score = outputs[0, label_index]

        target_score.backward()

        activation = activations[0].detach()
        gradient = gradients[0].detach()

        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = torch.sum(
            weights * activation,
            dim=1
        )

        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(1),
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )[0, 0]

        cam = cam.cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        probability = float(
            probabilities[label_index].detach().cpu().item()
        )

    finally:
        hook.remove()

    heatmap = colormaps["jet"](cam)[:, :, :3]

    heatmap = (
        heatmap * 255
    ).astype(np.uint8)

    heatmap_image = Image.fromarray(heatmap)

    overlay = Image.blend(
        display_image,
        heatmap_image,
        alpha=0.4
    )

    output_buffer = BytesIO()

    overlay.save(
        output_buffer,
        format="PNG"
    )

    output_buffer.seek(0)

    return output_buffer.getvalue(), probability
