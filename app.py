
import gradio as gr
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

device = torch.device("cpu")

def load_model():
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 1)
    )
    checkpoint = torch.load(
        "deepfake_detector.pth",
        map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image):
    if image is None:
        return "Please upload an image", None

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor).squeeze()
        prob = torch.sigmoid(output).item()

    fake_prob = prob
    real_prob = 1 - prob

    if fake_prob > 0.5:
        label = "FAKE — AI Generated"
        confidence = fake_prob * 100
        color = "red"
    else:
        label = "REAL — Authentic"
        confidence = real_prob * 100
        color = "green"

    result = f"{label}\nConfidence: {confidence:.1f}%"

    return (
        result,
        {
            "REAL": float(real_prob),
            "FAKE (AI Generated)": float(fake_prob)
        }
    )

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload any image"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(label="Confidence Scores")
    ],
    title="Deepfake / AI Image Detector",
    description=(
        "Upload any image to find out if it is real or "
        "AI-generated. Built with EfficientNet B0 trained "
        "on 120,000 images."
    ),
    examples=[],
    theme=gr.themes.Soft()
)

interface.launch()
