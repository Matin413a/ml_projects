from pathlib import Path

from PIL import Image
import torch
from torchvision.models import resnet18, ResNet18_Weights


# Load pretrained model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()

# Get the preprocessing pipeline
preprocess = weights.transforms()

# Use a path relative to this script so it works from any working directory
image_path = Path(__file__).resolve().parent / "cat.jpg"
if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

# Load image
image = Image.open(image_path)

# Preprocess image
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

# Make prediction
with torch.no_grad():
    output = model(input_batch)

# Convert model output to probabilities
probabilities = torch.nn.functional.softmax(output, dim=1)[0]

# Get the class with the highest probability
class_id = probabilities.argmax().item()
score = probabilities[class_id].item()

# Get class name
class_name = weights.meta["categories"][class_id]

print(f"Prediction: {class_name}")
print(f"Confidence: {score:.4f}")