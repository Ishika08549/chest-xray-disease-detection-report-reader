import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms, models


# 1. Configuration

MODEL_PATH = "outputs/models/best_resnet50.pth"

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

# 2. Device

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# 3. Image preprocessing

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 4. Build model

def build_model():

    model = models.resnet50(weights=None)

    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, len(LABELS))
    )

    return model


# 5. Load checkpoint

def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
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

    return model


# 6. Predict X-ray

def predict_xray(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    model = load_model()

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.sigmoid(
            outputs
        ).cpu().numpy()[0]

    predictions = (
        probabilities >= THRESHOLDS
    ).astype(int)

    print("\n========== CHEST X-RAY PREDICTION ==========")

    print("Image:", image_path)
    print("Model: ResNet50")
    print("\nCandidate prediction results:")

    for label, probability, prediction in zip(
        LABELS,
        probabilities,
        predictions
    ):

        status = (
            "Candidate detected"
            if prediction == 1
            else "Below threshold"
        )

        print(
            f"{label:15} "
            f"Probability: {probability * 100:6.2f}% "
            f"| {status}"
        )

    print("\nNote:")
    print(
        "These are experimental model predictions, "
        "not confirmed medical diagnoses."
    )

    return probabilities, predictions


# 7. Command-line interface

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Chest X-ray candidate prediction using ResNet50"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to chest X-ray image"
    )

    args = parser.parse_args()

    predict_xray(args.image)
