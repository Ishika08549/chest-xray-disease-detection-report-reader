import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

OUTPUT_DIR = Path("outputs/gradcam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Preprocessing

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 3. Build ResNet50

def build_model():

    model = models.resnet50(weights=None)

    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, len(LABELS))
    )

    return model


# 4. Load checkpoint

def load_model():

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


# 5. Generate Grad-CAM

def generate_gradcam(image_path, selected_label):

    if not Path(image_path).exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if selected_label not in LABELS:
        raise ValueError(
            f"Choose one of these labels: {LABELS}"
        )

    original_image = Image.open(
        image_path
    ).convert("RGB").resize((224, 224))

    input_tensor = transform(
        Image.open(image_path).convert("RGB")
    ).unsqueeze(0).to(DEVICE)

    model = load_model()

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

        probabilities = torch.sigmoid(
            outputs
        )[0]

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

        probability = probabilities[
            label_index
        ].item()

    finally:
        hook.remove()

    # 6. Create heatmap overlay

    heatmap = plt.get_cmap("jet")(
        cam
    )[:, :, :3]

    heatmap = (
        heatmap * 255
    ).astype(np.uint8)

    heatmap_image = Image.fromarray(heatmap)

    overlay = Image.blend(
        original_image,
        heatmap_image,
        alpha=0.4
    )

    output_path = OUTPUT_DIR / (
        f"{Path(image_path).stem}_{selected_label}_gradcam.png"
    )

    overlay.save(output_path)

    print("\n========== GRAD-CAM RESULTS ==========")
    print("Image:", image_path)
    print("Selected label:", selected_label)
    print(f"Probability: {probability * 100:.2f}%")
    print("Heatmap saved:", output_path)
    print("\nGrad-CAM generated successfully.")


# 7. Command-line interface

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM for chest X-ray"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to chest X-ray image"
    )

    parser.add_argument(
        "--label",
        required=True,
        choices=LABELS,
        help="Label for Grad-CAM visualization"
    )

    args = parser.parse_args()

    generate_gradcam(
        args.image,
        args.label
    )
