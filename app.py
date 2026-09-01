import os
import sys
import asyncio
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image, ImageOps
from torchvision import transforms
from threading import Lock

# Pillow rejects oversized decompression-bomb images before full decoding.
MAX_IMAGE_PIXELS = 25_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# Optional Hugging Face Spaces GPU decorator.
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(function):
            return function

# Helps avoid an asyncio issue during Gradio shutdown.
def _configure_asyncio():
    """Prevent Python 3.10/3.11 from reporting a duplicate self-pipe close."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Some Gradio/asyncio combinations attempt to close the selector self-pipe
    # twice. The second close sees fd=-1 and raises from BaseEventLoop.__del__.
    # Replace only that cleanup method with an idempotent equivalent.
    try:
        from asyncio import selector_events

        if not getattr(selector_events, "_deepfake_safe_pipe_cleanup", False):
            def safe_close_self_pipe(loop):
                for socket_name in ("_ssock", "_csock"):
                    sock = getattr(loop, socket_name, None)
                    if sock is None:
                        continue
                    if socket_name == "_ssock":
                        try:
                            fd = sock.fileno()
                            if fd >= 0:
                                loop._remove_reader(fd)
                        except (OSError, KeyError, ValueError):
                            pass
                    try:
                        sock.close()
                    except (OSError, ValueError):
                        pass
                    setattr(loop, socket_name, None)

            selector_events.BaseSelectorEventLoop._close_self_pipe = safe_close_self_pipe
            selector_events._deepfake_safe_pipe_cleanup = True
    except (AttributeError, ImportError):
        pass


_configure_asyncio()


# Use CUDA when available; otherwise use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training code established: label 0 = FAKE and label 1 = REAL.
# Checkpoint mapping: class 0 = FAKE and class 1 = REAL.
# Fake-priority mode raises the REAL cutoff so borderline cases are flagged FAKE.
# This improves fake recall at the cost of more real images being flagged.
FAKE_PRIORITY_MODE = os.environ.get("FAKE_PRIORITY_MODE", "1").lower() not in {"0", "false", "no"}
DEFAULT_REAL_THRESHOLD = 0.60 if FAKE_PRIORITY_MODE else 0.40
REAL_THRESHOLD = float(os.environ.get("REAL_THRESHOLD", str(DEFAULT_REAL_THRESHOLD)))
MAX_INPUT_SIDE = 2048
PREDICTION_LOCK = Lock()


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


class DeepfakeDetector(nn.Module):
    """Architecture matched to the attached checkpoint: backbone + frequency + head."""

    def __init__(self):
        super().__init__()

        efficientnet_base = models.efficientnet_b2(weights=None)
        self.backbone = efficientnet_base.features
        self.spatial_dim = efficientnet_base.classifier[1].in_features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.freq = FrequencyBranch(out_dim=256)
        self.freq_dim = 256

        # The attached checkpoint has no GAN branch: 1408 + 256 = 1664.
        combined_dim = self.spatial_dim + self.freq_dim
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
        combined = torch.cat([spatial, frequency], dim=1)
        return self.head(combined).squeeze(1)



def find_checkpoint():
    """Find the checkpoint independently of the process working directory."""
    configured = os.environ.get("CHECKPOINT_PATH")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())

    script_dir = Path(__file__).resolve().parent
    candidates.extend([
        script_dir / "deepfake_detector.pth",
        script_dir / "deepfake_detector .pth",
        script_dir / "deepfake_detector_fp16.pth",
        Path("deepfake_detector.pth"),
        Path("deepfake_detector .pth"),
        Path("deepfake_detector_fp16.pth"),
    ])

    for path in candidates:
        if path.is_file():
            return str(path.resolve())

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "No checkpoint found. Set CHECKPOINT_PATH or place "
        "deepfake_detector.pth beside app.py. Searched:\n" + searched
    )



def load_model():
    checkpoint_path = find_checkpoint()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # Accept both a wrapped checkpoint and a raw state_dict.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = {
            key: value for key, value in checkpoint.items()
            if isinstance(key, str) and torch.is_tensor(value)
        }
        if not state_dict:
            raise ValueError(
                "Checkpoint is neither a wrapped checkpoint nor a raw model state_dict."
            )
    else:
        raise ValueError("Unsupported checkpoint format.")

    # Support checkpoints saved through torch.nn.DataParallel.
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    model = DeepfakeDetector()
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Inference device: {device}")
    print("Label mapping: 0 = FAKE, 1 = REAL")
    print(f"REAL threshold: {REAL_THRESHOLD}")
    print("Decision policy: fake-priority; verify thresholds on labeled data")
    print("Architecture: EfficientNet-B2 backbone + frequency branch + head")
    return model


# Lazy loading avoids loading the model for --help or simple module imports.
model = None
MODEL_LOAD_LOCK = Lock()


def get_model():
    global model
    if model is None:
        with MODEL_LOAD_LOCK:
            if model is None:
                model = load_model()
    return model


normalization = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

# Exact deterministic preprocessing used during evaluation.
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalization,
])


def _predict_impl(image):
    if image is None:
        return "Please upload an image.", {
            "REAL": 0.0,
            "FAKE (AI Generated)": 0.0,
        }

    if isinstance(image, np.ndarray):
        if image.ndim >= 2 and image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
            raise ValueError("Image is too large to process safely.")
        image = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("Image is too large to process safely.")
    else:
        # Read image headers first, then copy the decoded image only after the
        # pixel-count guard has passed.
        with Image.open(image) as opened:
            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image is too large to process safely.")
            image = opened.copy()

    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) > MAX_INPUT_SIDE:
        image.thumbnail((MAX_INPUT_SIDE, MAX_INPUT_SIDE), Image.Resampling.LANCZOS)

    # The training code uses sigmoid(output) as class-1 probability.
    # Because class 1 is REAL, this is real_prob—not fake_prob.
    with PREDICTION_LOCK, torch.inference_mode():
        # Match the training/evaluation pipeline exactly: one 224x224 view,
        # one model forward pass, sigmoid output as class-1 probability.
        tensor = base_transform(image).unsqueeze(0).to(device)
        logit = get_model()(tensor).reshape(-1)[0]
        real_prob = float(torch.sigmoid(logit).item())

    fake_prob = 1.0 - real_prob

    # In fake-priority mode, REAL_THRESHOLD defaults to 0.60; otherwise it is 0.40.
    if real_prob > REAL_THRESHOLD:
        label = "REAL — Likely Authentic"
        confidence = real_prob * 100.0
    else:
        label = "FAKE — Likely AI Generated"
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



@spaces.GPU
def predict(image):
    """Public Gradio entry point with safe request-level error handling."""
    try:
        return _predict_impl(image)
    except Exception as exc:
        # Prevent backend exceptions from surfacing as aborted browser streams.
        print(f"Prediction error: {type(exc).__name__}: {exc}")
        return (
            "Unable to process this image. Please upload a valid JPG, PNG, or WEBP image.",
            {
                "REAL": 0.0,
                "FAKE (AI Generated)": 0.0,
            },
        )


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
        "Uses the trained EfficientNet-B2 spatial and frequency branches. "
        "The score mapping is 0 = FAKE and 1 = REAL, matching training. "
        "Fake-priority mode favors catching fake images and may flag some real images. "
        "Results are estimates, not proof of authenticity."
    ),
    examples=[],
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(f"Usage: python {Path(__file__).name}")
        print("Starts the Gradio deepfake detector server.")
        raise SystemExit(0)

    share_enabled = os.environ.get("GRADIO_SHARE", "0").lower() in {
        "1", "true", "yes"
    }

    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=share_enabled,
        theme=gr.themes.Soft(),
        css=custom_css,
    )
