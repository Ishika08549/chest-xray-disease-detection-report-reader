import torch
import torch.nn as nn
from torchvision import models

# 1. Select device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Selected device:", device)


# 2. Load pretrained ResNet18
weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)


# 3. Replace final layer
# Four output neurons for four candidate labels
num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 4)


# 4. Move model to device
model = model.to(device)


# 5. Test model with sample input
sample_input = torch.randn(2, 3, 224, 224).to(device)

model.eval()

with torch.no_grad():
    output = model(sample_input)

print("\nRESNET18 MODEL DETAILS")
print("Architecture: ResNet18")
print("Pretrained weights: ImageNet")
print("Input shape:", sample_input.shape)
print("Output shape:", output.shape)
print("Output values:", output)

# Convert logits into probabilities
probabilities = torch.sigmoid(output)

print("\nPREDICTED PROBABILITIES")
print(probabilities)

print("\nRESNET18 MODEL WORKING SUCCESSFULLY!")