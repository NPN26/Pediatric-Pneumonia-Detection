"""Grad-CAM generation utilities for Streamlit inference."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_module: torch.nn.Module) -> None:
        self.model = model
        self.target_module = target_module
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._forward_handle = self.target_module.register_forward_hook(self._save_activations)
        self._backward_handle = self.target_module.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inputs, output) -> None:
        self.activations = output

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def remove_hooks(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        target_score = logits[:, class_idx]
        target_score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1)
        cam = F.relu(cam)

        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam.squeeze(0).detach().cpu().numpy()


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def overlay_cam_on_image(
    base_image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    base_rgb = base_image.convert("RGB")
    width, height = base_rgb.size

    cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float()
    cam_up = F.interpolate(cam_tensor, size=(height, width), mode="bilinear", align_corners=False)
    cam_up_np = cam_up.squeeze().cpu().numpy()

    heatmap = (_jet_colormap(cam_up_np) * 255.0).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap, mode="RGB")

    return Image.blend(base_rgb, heatmap_img, alpha=alpha)


def generate_gradcam_overlay(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: Image.Image,
    class_idx: Optional[int] = None,
) -> Tuple[np.ndarray, Image.Image]:
    gradcam = GradCAM(model, model.mf)
    try:
        cam = gradcam.generate(input_tensor=input_tensor, class_idx=class_idx)
        overlay = overlay_cam_on_image(original_image, cam)
    finally:
        gradcam.remove_hooks()
    return cam, overlay
