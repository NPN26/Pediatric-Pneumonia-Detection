"""Model definition and inference helpers for pediatric pneumonia detection."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class GhostConv(nn.Module):
    def __init__(self, inputs: int, out_channels: int) -> None:
        super().__init__()
        ghost_channels = (out_channels // 2) // 2 * 2
        self.primary_channels = ghost_channels // 2
        self.ghost_channels = ghost_channels // 2
        self.out_channels = out_channels

        self.primary = nn.Sequential(
            nn.Conv2d(inputs, self.primary_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.primary_channels),
            nn.ReLU(),
        )

        self.cheap = nn.Sequential(
            nn.Conv2d(
                self.primary_channels,
                self.ghost_channels,
                kernel_size=3,
                padding=1,
                groups=self.primary_channels,
                bias=False,
            ),
            nn.BatchNorm2d(self.ghost_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary(x)
        x2 = self.cheap(x1)
        out = torch.cat([x1, x2], dim=1)

        if out.size(1) < self.out_channels:
            padding = torch.zeros(
                out.size(0),
                self.out_channels - out.size(1),
                out.size(2),
                out.size(3),
                device=out.device,
            )
            out = torch.cat([out, padding], dim=1)

        return out


class FE_Module(nn.Module):
    def __init__(self, inputs: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, out_channels, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)

        self.remainder = int(out_channels * 0.2)
        ghost_channels = out_channels - self.remainder
        if ghost_channels % 2 != 0:
            ghost_channels -= 1
            self.remainder += 1

        self.ghost = GhostConv(ghost_channels, ghost_channels)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = F.relu(self.conv1(x))

        x_remainder = x[:, : self.remainder]
        x_ghost = x[:, self.remainder :]

        x_ghost = self.ghost(x_ghost)

        x = torch.cat([x_remainder, x_ghost], dim=1)
        x = self.bn(self.conv2(x))

        if identity.shape == x.shape:
            x += identity

        return x


class GDConv(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        half_channels = channels // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        self.dilated = nn.Sequential(
            nn.Conv2d(
                half_channels,
                half_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=half_channels,
            ),
            nn.BatchNorm2d(half_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        half = x.shape[1] // 2
        x1 = x[:, :half]
        x2 = x[:, half:]

        x1 = self.dilated(x1)
        return torch.cat([x1, x2], dim=1)


class MF_Module(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool5 = nn.MaxPool2d(5, stride=1, padding=2)
        self.pool9 = nn.MaxPool2d(9, stride=1, padding=4)
        self.pool13 = nn.MaxPool2d(13, stride=1, padding=6)

        self.branches = nn.ModuleList()
        for _ in range(3):
            branch = nn.ModuleList(
                [
                    GDConv(channels, dilation=1),
                    GDConv(channels, dilation=5),
                    GDConv(channels, dilation=9),
                ]
            )
            self.branches.append(branch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [x]

        p5 = self.pool5(x)
        for conv in self.branches[0]:
            outputs.append(conv(p5))

        p9 = self.pool9(x)
        for conv in self.branches[1]:
            outputs.append(conv(p9))

        p13 = self.pool13(x)
        for conv in self.branches[2]:
            outputs.append(conv(p13))

        return torch.cat(outputs, dim=1)


class Pnemonia_Model(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()

        self.start = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.fe1 = FE_Module(64, 128)
        self.fe2 = FE_Module(128, 256)
        self.mf = MF_Module(256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(2560, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.start(x)
        x = self.fe1(x)
        x = self.fe2(x)
        x = self.mf(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def preprocess_image(image: Image.Image) -> torch.Tensor:
    rgb_image = image.convert("RGB")
    return INFERENCE_TRANSFORM(rgb_image).unsqueeze(0)


def load_model(weights_path: str = "best_model.pth", device: str = "cpu") -> Pnemonia_Model:
    model = Pnemonia_Model(num_classes=2)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(model: nn.Module, image: Image.Image, device: str = "cpu") -> Dict[str, object]:
    input_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    pred_idx = int(torch.argmax(probabilities).item())
    label = CLASS_NAMES[pred_idx]
    confidence = float(probabilities[pred_idx].item())

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[0]: float(probabilities[0].item()),
            CLASS_NAMES[1]: float(probabilities[1].item()),
        },
    }


def predict_with_gradcam(
    model: nn.Module, image: Image.Image, device: str = "cpu"
) -> Dict[str, object]:
    from gradcam import generate_gradcam_overlay

    input_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    pred_idx = int(torch.argmax(probabilities).item())
    label = CLASS_NAMES[pred_idx]
    confidence = float(probabilities[pred_idx].item())

    _cam, overlay = generate_gradcam_overlay(
        model=model,
        input_tensor=input_tensor,
        original_image=image,
        class_idx=pred_idx,
    )

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[0]: float(probabilities[0].item()),
            CLASS_NAMES[1]: float(probabilities[1].item()),
        },
        "gradcam_overlay": overlay,
    }
