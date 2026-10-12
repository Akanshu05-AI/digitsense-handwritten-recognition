"""
DigitSense: Explainable AI (XAI) Module using Grad-CAM
Implements Gradient-weighted Class Activation Mapping (Grad-CAM) for Convolutional Neural Networks.
Provides visual explanations of model decisions by highlighting salient pixel regions.
"""

from typing import Tuple, Dict, Any, Optional, Union, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from config import DEVICE
from src.cnn_model import DigitSenseCNN


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for DigitSenseCNN.

    Reference:
    Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017).
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
    ICCV 2017.
    """

    def __init__(self, model: DigitSenseCNN, target_layer: Optional[nn.Module] = None, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()

        # Default target layer to final conv layer (conv5)
        self.target_layer = target_layer if target_layer is not None else self.model.conv5

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hooks: List[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target convolutional layer."""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] is the gradient w.r.t the layer's output
            self.gradients = grad_output[0]

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        # Use full backward hook for clean gradient capture
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        """Remove hooks to prevent memory leaks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Compute Grad-CAM heatmap for a single 1x1x28x28 input tensor.

        Args:
            input_tensor: PyTorch tensor with shape (1, 1, 28, 28)
            target_class: Optional target class index. If None, uses top predicted class.

        Returns:
            heatmap (np.ndarray): 28x28 float array in [0.0, 1.0]
            predicted_class (int): Index of the analyzed class
            confidence (float): Softmax probability of target class
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(False)

        # Ensure target layer requires gradients for backpropagation
        for param in self.target_layer.parameters():
            param.requires_grad = True

        # Forward pass
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        confidence = float(probs[0, target_class].item())

        # Zero gradients and backpropagate from chosen logit
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations during Grad-CAM backward pass.")

        # Global Average Pooling of gradients across spatial dimensions (H, W)
        # gradients shape: (1, C, H, W)
        # pooled_grads shape: (C,)
        pooled_grads = torch.mean(self.gradients[0], dim=[1, 2])

        # Weight the feature maps by the pooled gradients
        activations = self.activations[0]  # (C, H, W)
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=self.device)

        for i, w in enumerate(pooled_grads):
            cam += w * activations[i]

        # Apply ReLU to retain only features with positive influence on the target score
        cam = F.relu(cam)

        # Move to CPU numpy
        cam_np = cam.detach().cpu().numpy()

        # Normalize to [0.0, 1.0]
        cam_max = np.max(cam_np)
        if cam_max > 1e-8:
            cam_np = cam_np / cam_max
        else:
            cam_np = np.zeros_like(cam_np)

        # Upsample heatmap to 28x28 input dimensions using bilinear interpolation
        heatmap_28 = cv2.resize(cam_np, (28, 28), interpolation=cv2.INTER_LINEAR)
        # Ensure values stay strictly in [0.0, 1.0]
        heatmap_28 = np.clip(heatmap_28, 0.0, 1.0)

        return heatmap_28, target_class, confidence


def overlay_heatmap_on_crop(
    crop_28x28: np.ndarray,
    heatmap_28x28: np.ndarray,
    alpha: float = 0.55,
    colormap: int = cv2.COLORMAP_JET
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate visual overlays from a Grad-CAM heatmap and original 28x28 crop.

    Args:
        crop_28x28: Grayscale character crop (28x28 uint8 or float in [0, 255] or [0, 1])
        heatmap_28x28: 28x28 float array in [0.0, 1.0]
        alpha: Weight of the heatmap in the overlay blend (0.0 to 1.0)
        colormap: OpenCV colormap constant (e.g. cv2.COLORMAP_JET or cv2.COLORMAP_VIRIDIS)

    Returns:
        blended_rgb (np.ndarray): 28x28x3 RGB uint8 image with heatmap blended over stroke
        colored_heatmap_rgb (np.ndarray): 28x28x3 RGB uint8 image showing raw colormapped heatmap
    """
    # Normalize crop to uint8 [0, 255]
    if crop_28x28.dtype != np.uint8:
        if np.max(crop_28x28) <= 1.0:
            base_gray = (crop_28x28 * 255).astype(np.uint8)
        else:
            base_gray = np.clip(crop_28x28, 0, 255).astype(np.uint8)
    else:
        base_gray = crop_28x28.copy()

    # Convert base grayscale to RGB
    base_rgb = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2RGB)

    # Convert heatmap to uint8 [0, 255]
    heat_u8 = (np.clip(heatmap_28x28, 0.0, 1.0) * 255).astype(np.uint8)

    # Apply colormap (OpenCV returns BGR)
    colored_heat_bgr = cv2.applyColorMap(heat_u8, colormap)
    colored_heat_rgb = cv2.cvtColor(colored_heat_bgr, cv2.COLOR_BGR2RGB)

    # Blend original stroke and heatmap
    blended = cv2.addWeighted(colored_heat_rgb, alpha, base_rgb, 1.0 - alpha, 0)

    return blended, colored_heat_rgb


def analyze_attention_regions(heatmap_28x28: np.ndarray) -> Dict[str, Any]:
    """
    Analyze the spatial distribution of Grad-CAM attention across anatomical zones
    of the 28x28 character canvas (Top, Center, Bottom, Left, Right).

    Returns:
        Dict containing zone activation percentages and dominant attention summary.
    """
    h, w = heatmap_28x28.shape

    top_zone = float(np.mean(heatmap_28x28[:h//3, :]))
    center_y_zone = float(np.mean(heatmap_28x28[h//3: 2*(h//3), :]))
    bottom_zone = float(np.mean(heatmap_28x28[2*(h//3):, :]))

    left_zone = float(np.mean(heatmap_28x28[:, :w//3]))
    center_x_zone = float(np.mean(heatmap_28x28[:, w//3: 2*(w//3)]))
    right_zone = float(np.mean(heatmap_28x28[:, 2*(w//3):]))

    # Locate the peak activation coordinate (y, x)
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap_28x28), heatmap_28x28.shape)

    # Semantic description of the dominant region
    descriptions = []
    if top_zone > 0.4:
        descriptions.append("upper ascender / top loop / roof bar")
    if center_y_zone > 0.4:
        descriptions.append("central crossbar / junction / stroke intersection")
    if bottom_zone > 0.4:
        descriptions.append("base loop / baseline / descender")
    if not descriptions:
        descriptions.append("distributed stroke curvature")

    summary = ", ".join(descriptions)

    return {
        "vertical_zones": {
            "top": round(top_zone, 3),
            "middle": round(center_y_zone, 3),
            "bottom": round(bottom_zone, 3)
        },
        "horizontal_zones": {
            "left": round(left_zone, 3),
            "center": round(center_x_zone, 3),
            "right": round(right_zone, 3)
        },
        "peak_pixel": (int(peak_x), int(peak_y)),
        "dominant_features": summary
    }


def explain_character_prediction(
    model: DigitSenseCNN,
    crop_28x28: np.ndarray,
    target_class: Optional[int] = None,
    class_names: Optional[List[str]] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    High-level convenience function: Computes Grad-CAM heatmap, overlays, and
    attention explanation for a 28x28 character crop.
    """
    from src.preprocessing import to_model_tensor

    input_tensor = to_model_tensor(crop_28x28, device=device)
    cam_engine = GradCAM(model, device=device)

    try:
        heatmap, pred_idx, confidence = cam_engine.generate_cam(input_tensor, target_class=target_class)
        blended, colormapped = overlay_heatmap_on_crop(crop_28x28, heatmap)
        attention_info = analyze_attention_regions(heatmap)

        char_name = class_names[pred_idx] if class_names and pred_idx < len(class_names) else str(pred_idx)

        return {
            "target_class_index": pred_idx,
            "target_class_name": char_name,
            "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.1f}%",
            "heatmap": heatmap,
            "blended_overlay": blended,
            "colormapped_heatmap": colormapped,
            "attention_analysis": attention_info
        }
    finally:
        cam_engine.remove_hooks()
