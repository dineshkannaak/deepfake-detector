import os
import sys
import asyncio
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

# Optional Hugging Face Spaces GPU decorator.
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(function):
            return function

# Helps avoid an asyncio issue on Windows.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# Use CUDA when available; otherwise use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training code established: label 0 = FAKE and label 1 = REAL.
# During evaluation, sigmoid(output) > 0.40 was treated as class 1 (REAL).
REAL_THRESHOLD = 0.40


class FrequencyBranch(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),

            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(4096, out_dim),
            nn.GELU(),
        )

    @staticmethod
    def to_freq_map(x):
        x_fft = torch.fft.fft2(x, norm="ortho")
        x_mag = torch.abs(x_fft) + 1e-8
        x_log = torch.log(x_mag)
        mu = x_log.mean(dim=(-2, -1), keepdim=True)
        std = x_log.std(dim=(-2, -1), keepdim=True) + 1e-8
        return (x_log - mu) / std

    def forward(self, x):
        return self.cnn(self.to_freq_map(x))


class GANBranch(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(256 * 4, out_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)


class DeepfakeDetector(nn.Module):
    """Exact architecture used by the supplied training code."""

    def __init__(self):
        super().__init__()

        efficientnet_base = models.efficientnet_b2(weights=None)
        self.backbone = efficientnet_base.features
        self.spatial_dim = efficientnet_base.classifier[1].in_features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.freq = FrequencyBranch(out_dim=256)
        self.freq_dim = 256

        self.gan = GANBranch(out_dim=128)
        self.gan_dim = 128

        combined_dim = self.spatial_dim + self.freq_dim + self.gan_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.40),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        spatial = self.avgpool(self.backbone(x)).flatten(1)
        frequency = self.freq(x)
        gan_features = self.gan(x)
        combined = torch.cat([spatial, frequency, gan_features], dim=1)
        return self.head(combined).squeeze(1)



def find_checkpoint():
    candidates = [
        "deepfake_detector.pth",
        "deepfake_detector_fp16.pth",
        "deepfake_detector.pht",
    ]
    for filename in candidates:
        if Path(filename).exists():
            return filename
    raise FileNotFoundError(
        "No checkpoint found. Put deepfake_detector.pth or "
        "deepfake_detector_fp16.pth in the same folder as this script."
    )



def load_model():
    checkpoint_path = find_checkpoint()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain a 'model_state_dict' key.")

    model = DeepfakeDetector()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Inference device: {device}")
    print("Label mapping: 0 = FAKE, 1 = REAL")
    print(f"REAL threshold: {REAL_THRESHOLD}")
    return model


model = load_model()


normalization = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

# Deterministic views reduce sensitivity to a single portrait crop or flip.
tta_transforms = [
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalization,
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        normalization,
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalization,
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.10),
        transforms.ToTensor(),
        normalization,
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(contrast=0.10),
        transforms.ToTensor(),
        normalization,
    ]),
]


@spaces.GPU
def predict(image):
    if image is None:
        return "Please upload an image.", {
            "REAL": 0.0,
            "FAKE (AI Generated)": 0.0,
        }

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")

    # The training code uses sigmoid(output) as class-1 probability.
    # Because class 1 is REAL, this is real_prob—not fake_prob.
    real_probabilities = []

    with torch.inference_mode():
        for transform in tta_transforms:
            tensor = transform(image).unsqueeze(0).to(device)
            logit = model(tensor).reshape(-1)[0]
            real_probability = torch.sigmoid(logit).item()
            real_probabilities.append(real_probability)

    real_prob = float(np.mean(real_probabilities))
    fake_prob = 1.0 - real_prob

    # Match the training/evaluation rule: class 1 (REAL) when real_prob > 0.40.
    if real_prob > REAL_THRESHOLD:
        label = "REAL — Authentic"
        confidence = real_prob * 100.0
    else:
        label = "FAKE — AI Generated"
        confidence = fake_prob * 100.0

    result = (
        f"{label}\n"
        f"Confidence: {confidence:.1f}%\n"
        f"Real probability: {real_prob * 100:.1f}%\n"
        f"Fake probability: {fake_prob * 100:.1f}%"
    )

    scores = {
        "REAL": real_prob,
        "FAKE (AI Generated)": fake_prob,
    }
    return result, scores


custom_css = """
footer,
#footer,
.gradio-container > footer {
    display: none !important;
}
"""


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a portrait or other image"),
    outputs=[
        gr.Textbox(label="Prediction", lines=4),
        gr.Label(label="Confidence Scores"),
    ],
    title="Deepfake / AI Image Detector",
    description=(
        "Uses the trained spatial, frequency, and GAN branches. "
        "The score mapping is 0 = FAKE and 1 = REAL, matching training. "
        "Results are estimates, not proof of authenticity."
    ),
    examples=[],
    theme=gr.themes.Soft(),
    css=custom_css,
)


if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
