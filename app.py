import gradio as gr
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Use CPU for inference
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

    # Supports checkpoints containing model_state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
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
        return "Please upload an image.", {
            "REAL": 0.0,
            "FAKE (AI Generated)": 0.0
        }

    # Convert Gradio image input to RGB PIL image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        image = Image.open(image).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor).reshape(-1)[0]
        fake_prob = torch.sigmoid(output).item()

    real_prob = 1.0 - fake_prob

    if fake_prob >= 0.5:
        label = "FAKE — AI Generated"
        confidence = fake_prob * 100
    else:
        label = "REAL — Authentic"
        confidence = real_prob * 100

    result = f"{label}\nConfidence: {confidence:.1f}%"

    scores = {
        "REAL": float(real_prob),
        "FAKE (AI Generated)": float(fake_prob)
    }

    return result, scores


# CSS hides the standard Gradio footer and branding
custom_css = """
footer {
    display: none !important;
}

#footer {
    display: none !important;
}

.gradio-container > footer {
    display: none !important;
}
"""


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="pil",
        label="Upload any image"
    ),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(label="Confidence Scores")
    ],
    title="Deepfake / AI Image Detector",
    description=(
        "Upload an image to estimate whether it is authentic or AI-generated. "
        "Built with EfficientNet-B0."
    ),
    examples=[],
    theme=gr.themes.Soft(),
    css=custom_css
)


if __name__ == "__main__":
    interface.launch()

