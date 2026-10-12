"""
Tests for Explainable AI (Grad-CAM) module.
Verifies forward/backward hook capture, gradient propagation, and heatmap normalization.
"""

import pytest
import numpy as np
import torch

from src.cnn_model import DigitSenseCNN
from src.explainability import (
    GradCAM,
    overlay_heatmap_on_crop,
    analyze_attention_regions,
    explain_character_prediction
)


@pytest.fixture
def dummy_cnn():
    """Create a lightweight initialized DigitSenseCNN instance."""
    torch.manual_seed(42)
    model = DigitSenseCNN(num_classes=10)
    model.eval()
    return model


def test_gradcam_generation_shape_and_range(dummy_cnn):
    cam = GradCAM(dummy_cnn)
    x = torch.randn(1, 1, 28, 28)

    heatmap, pred_idx, conf = cam.generate_cam(x)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (28, 28)
    assert 0 <= pred_idx < 10
    assert 0.0 <= conf <= 1.0
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0

    cam.remove_hooks()


def test_gradcam_specific_target_class(dummy_cnn):
    cam = GradCAM(dummy_cnn)
    x = torch.randn(1, 1, 28, 28)

    target_cls = 7
    heatmap, pred_idx, conf = cam.generate_cam(x, target_class=target_cls)

    assert pred_idx == target_cls
    assert heatmap.shape == (28, 28)

    cam.remove_hooks()


def test_overlay_heatmap_on_crop():
    crop = np.zeros((28, 28), dtype=np.uint8)
    crop[10:18, 12:16] = 255  # vertical stroke
    heatmap = np.zeros((28, 28), dtype=np.float32)
    heatmap[10:18, 12:16] = 1.0

    blended, colormapped = overlay_heatmap_on_crop(crop, heatmap)

    assert blended.shape == (28, 28, 3)
    assert blended.dtype == np.uint8
    assert colormapped.shape == (28, 28, 3)
    assert colormapped.dtype == np.uint8


def test_analyze_attention_regions():
    heatmap = np.zeros((28, 28), dtype=np.float32)
    # Put strong activation at top
    heatmap[:9, :] = 0.9

    analysis = analyze_attention_regions(heatmap)

    assert "vertical_zones" in analysis
    assert "horizontal_zones" in analysis
    assert "peak_pixel" in analysis
    assert "dominant_features" in analysis
    assert analysis["vertical_zones"]["top"] > analysis["vertical_zones"]["bottom"]


def test_explain_character_prediction_pipeline(dummy_cnn):
    crop = np.zeros((28, 28), dtype=np.uint8)
    crop[7:21, 13:15] = 255  # digit 1 stroke

    result = explain_character_prediction(
        model=dummy_cnn,
        crop_28x28=crop,
        class_names=[str(i) for i in range(10)]
    )

    assert "heatmap" in result
    assert "blended_overlay" in result
    assert "colormapped_heatmap" in result
    assert "attention_analysis" in result
    assert result["heatmap"].shape == (28, 28)
