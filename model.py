"""Hybrid ALeaf-STENet model loading and inference helpers."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_CLASSES = [
    "Alternaria leaf spot",
    "Brown spot",
    "Frogeye leaf spot",
    "Grey spot",
    "Health",
    "Mosaic",
    "Powdery mildew",
    "Rust",
    "Scab",
]


class MissingModelDependency(RuntimeError):
    """Raised when torch/timm are not available."""


@dataclass
class ModelBundle:
    model: Any
    classes: list[str]
    device: Any
    val_acc: float | None
    missing_keys: list[str]
    unexpected_keys: list[str]


def _import_torch_stack():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import timm
    except ModuleNotFoundError as exc:
        raise MissingModelDependency(
            "PyTorch and timm are required for hybrid model inference. "
            "Install the packages from requirements.txt before running predictions."
        ) from exc

    return torch, nn, F, timm


def _build_hybrid_model(num_classes: int):
    torch, nn, F, timm = _import_torch_stack()
    effnet_name = os.getenv("ALEAF_EFFNET_NAME", "tf_efficientnetv2_s")
    swin_name = os.getenv("ALEAF_SWIN_NAME", "swin_tiny_patch4_window7_224")

    class ChannelAttention(nn.Module):
        def __init__(self, channels: int, reduction: int = 16):
            super().__init__()
            hidden = max(channels // reduction, 1)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(channels, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, channels),
            )

        def forward(self, x):
            avg = self.fc(self.avg_pool(x))
            max_value = self.fc(self.max_pool(x))
            scale = torch.sigmoid(avg + max_value).view(x.size(0), x.size(1), 1, 1)
            return x * scale

    class SpatialAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

        def forward(self, x):
            avg = torch.mean(x, dim=1, keepdim=True)
            max_value, _ = torch.max(x, dim=1, keepdim=True)
            scale = torch.sigmoid(self.conv(torch.cat([avg, max_value], dim=1)))
            return x * scale

    class CBAM(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.channel = ChannelAttention(channels)
            self.spatial = SpatialAttention()

        def forward(self, x):
            return self.spatial(self.channel(x))

    class HybridALeafSTENet(nn.Module):
        def __init__(self, classes_count: int):
            super().__init__()
            self.effnet_backbone = timm.create_model(
                effnet_name,
                pretrained=False,
                num_classes=0,
                global_pool="",
            )
            self.cbam = CBAM(1280)
            self.swin = timm.create_model(
                swin_name,
                pretrained=False,
                num_classes=0,
                global_pool="avg",
            )
            self.fusion = nn.Sequential(
                nn.Linear(2048, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )
            self.cls_head = nn.Linear(512, classes_count)
            self.sev_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            eff_features = self.effnet_backbone.forward_features(x)
            if eff_features.ndim == 4:
                eff_features = self.cbam(eff_features)
                eff_features = F.adaptive_avg_pool2d(eff_features, 1).flatten(1)

            swin_features = self.swin(x)
            if isinstance(swin_features, (tuple, list)):
                swin_features = swin_features[0]
            if swin_features.ndim == 4:
                if swin_features.shape[-1] == 768:
                    swin_features = swin_features.mean(dim=(1, 2))
                else:
                    swin_features = swin_features.mean(dim=(2, 3))

            features = torch.cat([eff_features, swin_features], dim=1)
            fused = self.fusion(features)
            class_logits = self.cls_head(fused)
            severity_raw = self.sev_head(fused)
            return class_logits, severity_raw

    return HybridALeafSTENet(num_classes)


def load_hybrid_model(model_path: str | os.PathLike[str]) -> ModelBundle:
    """Load the trained ALeaf-STENet checkpoint."""

    torch, _, _, _ = _import_torch_stack()
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {path}. Put aleaf_best.pth beside app.py "
            "or set ALEAF_MODEL_PATH to the checkpoint path."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    classes = list(checkpoint.get("classes", DEFAULT_CLASSES))
    model = _build_hybrid_model(len(classes)).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    try:
        model.load_state_dict(state_dict, strict=True)
        missing_keys: list[str] = []
        unexpected_keys: list[str] = []
    except RuntimeError:
        load_result = model.load_state_dict(state_dict, strict=False)
        missing_keys = list(load_result.missing_keys)
        unexpected_keys = list(load_result.unexpected_keys)

    model.eval()
    return ModelBundle(
        model=model,
        classes=classes,
        device=device,
        val_acc=checkpoint.get("val_acc"),
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )


def preprocess_image(image: Image.Image):
    """Convert a PIL image into the model's 224x224 normalized tensor."""

    torch, _, _, _ = _import_torch_stack()
    image = image.convert("RGB").resize((224, 224))
    array = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0)


def normalize_severity(raw_value: float) -> float:
    """Normalize severity head output into 0-100 percentage."""

    mode = os.getenv("ALEAF_SEVERITY_MODE", "auto").strip().lower()

    if mode == "sigmoid":
        value = 100.0 / (1.0 + math.exp(-raw_value))
    elif mode == "fraction":
        value = raw_value * 100.0
    elif mode == "percent":
        value = raw_value
    else:
        if 0.0 <= raw_value <= 1.0:
            value = raw_value * 100.0
        elif -8.0 <= raw_value <= 8.0:
            value = 100.0 / (1.0 + math.exp(-raw_value))
        else:
            value = raw_value

    return round(max(0.0, min(100.0, value)), 1)


def predict_leaf(bundle: ModelBundle, image: Image.Image) -> dict:
    """Run classification and severity inference on one leaf image."""

    torch, _, _, _ = _import_torch_stack()
    tensor = preprocess_image(image).to(bundle.device)

    with torch.no_grad():
        class_logits, severity_raw = bundle.model(tensor)
        probabilities = torch.softmax(class_logits, dim=1)[0]
        confidence, class_index = torch.max(probabilities, dim=0)

    class_idx = int(class_index.item())
    class_name = bundle.classes[class_idx]
    top_values, top_indices = torch.topk(probabilities, k=min(3, len(bundle.classes)))
    severity_raw_value = float(severity_raw.detach().flatten()[0].item())

    return {
        "class_index": class_idx,
        "class_name": class_name,
        "confidence": round(float(confidence.item()) * 100.0, 1),
        "severity": normalize_severity(severity_raw_value),
        "raw_severity": severity_raw_value,
        "top_predictions": [
            {
                "class_name": bundle.classes[int(index.item())],
                "confidence": round(float(value.item()) * 100.0, 1),
            }
            for value, index in zip(top_values, top_indices)
        ],
        "device": str(bundle.device),
        "val_acc": bundle.val_acc,
    }


def _hotspots_from_cam(cam: np.ndarray, max_points: int = 3) -> list[dict]:
    blocks = []
    height, width = cam.shape
    block_h = max(12, height // 8)
    block_w = max(12, width // 8)
    for y in range(0, height, block_h):
        for x in range(0, width, block_w):
            patch = cam[y : y + block_h, x : x + block_w]
            if patch.size:
                blocks.append((float(patch.mean()), x + block_w / 2, y + block_h / 2))

    blocks.sort(reverse=True, key=lambda item: item[0])
    hotspots: list[dict] = []
    for value, x, y in blocks[:max_points]:
        if value < 0.18:
            continue
        hotspots.append(
            {
                "x": round((x / width) * 100, 1),
                "y": round((y / height) * 100, 1),
                "radius": round(12 + value * 16, 1),
                "intensity": round(min(0.98, max(0.45, value)), 2),
            }
        )
    return hotspots


def generate_gradcam_plus_plus(bundle: ModelBundle, image: Image.Image, class_index: int | None = None) -> tuple[Image.Image, list[dict]]:
    """Generate a Grad-CAM++ style overlay from the CBAM output activations."""

    torch, _, F, _ = _import_torch_stack()
    model = bundle.model
    model.eval()

    activations: dict[str, Any] = {}
    gradients: dict[str, Any] = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    handle_forward = model.cbam.register_forward_hook(forward_hook)
    handle_backward = model.cbam.register_full_backward_hook(backward_hook)

    try:
        tensor = preprocess_image(image).to(bundle.device)
        model.zero_grad(set_to_none=True)
        class_logits, _severity_raw = model(tensor)
        target_index = int(class_index if class_index is not None else class_logits.argmax(dim=1).item())
        target_score = class_logits[:, target_index].sum()
        target_score.backward()

        if "value" not in activations or "value" not in gradients:
            raise RuntimeError("Unable to capture Grad-CAM activations from the CBAM layer.")

        activation = activations["value"].detach()[0]
        gradient = gradients["value"].detach()[0]

        grad_2 = gradient.pow(2)
        grad_3 = grad_2 * gradient
        activation_sum = activation.view(activation.shape[0], -1).sum(dim=1).view(-1, 1, 1)
        denominator = 2.0 * grad_2 + activation_sum * grad_3
        denominator = torch.where(denominator != 0.0, denominator, torch.ones_like(denominator))
        alpha = grad_2 / (denominator + 1e-7)
        weights = (alpha * torch.relu(gradient)).sum(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * activation).sum(dim=0))

        if float(cam.max().item()) > 0:
            cam = cam / cam.max()

        width, height = image.size
        cam = F.interpolate(
            cam[None, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam_np = cam.detach().cpu().numpy()

        base = np.asarray(image.convert("RGB")).astype("float32") / 255.0
        heat = np.zeros_like(base)
        heat[..., 0] = np.clip(1.8 * cam_np, 0, 1)
        heat[..., 1] = np.clip(1.8 * cam_np - 0.35, 0, 1)
        heat[..., 2] = np.clip(0.45 - cam_np, 0, 0.45)
        alpha_map = (0.22 + 0.58 * cam_np)[..., None]
        alpha_map = np.clip(alpha_map, 0.18, 0.78)
        overlay = np.clip(base * (1.0 - alpha_map) + heat * alpha_map, 0, 1)

        return Image.fromarray((overlay * 255).astype("uint8")), _hotspots_from_cam(cam_np)
    finally:
        handle_forward.remove()
        handle_backward.remove()
        model.zero_grad(set_to_none=True)
