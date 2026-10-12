"""
Unit Tests for Image Preprocessing Module
Covers loading, grayscale conversion, polarity detection, thresholding,
resizing, centering, and tensor normalization.
"""

import pytest
import numpy as np
import cv2
from PIL import Image
import torch

from src.preprocessing import (
    load_image,
    to_grayscale,
    detect_polarity,
    normalize_polarity,
    binarize,
    denoise,
    crop_to_bounding_box,
    resize_and_pad,
    center_by_center_of_mass,
    preprocess_image,
    to_model_tensor,
    to_classical_features
)


def test_load_image_numpy():
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    loaded = load_image(dummy)
    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (100, 100, 3)


def test_load_image_pil():
    pil_img = Image.new("RGB", (64, 64), color="white")
    loaded = load_image(pil_img)
    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (64, 64, 3)


def test_load_image_invalid():
    with pytest.raises(ValueError):
        load_image(None)

    with pytest.raises(ValueError):
        load_image(np.array([]))

    with pytest.raises(FileNotFoundError):
        load_image("non_existent_image_path_12345.png")


def test_to_grayscale():
    # 3-channel BGR
    bgr = np.ones((50, 50, 3), dtype=np.uint8) * 200
    gray = to_grayscale(bgr)
    assert gray.ndim == 2
    assert gray.shape == (50, 50)

    # 4-channel RGBA (alpha blended with white bg)
    rgba = np.zeros((50, 50, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255  # fully opaque black
    gray_alpha = to_grayscale(rgba)
    assert gray_alpha.ndim == 2
    assert gray_alpha.shape == (50, 50)
    assert gray_alpha[25, 25] == 0


def test_polarity_detection_and_normalization():
    # White background (255) with dark stroke (0) in center
    img_light_bg = np.ones((100, 100), dtype=np.uint8) * 255
    img_light_bg[40:60, 40:60] = 0
    assert detect_polarity(img_light_bg) == "light_bg"

    # Normalized polarity should invert so stroke is white (255) on black (0)
    normalized = normalize_polarity(img_light_bg, target="white_on_black")
    assert normalized[0, 0] == 0       # background is dark
    assert normalized[50, 50] == 255   # stroke is bright

    # Dark background (0) with white stroke (255)
    img_dark_bg = np.zeros((100, 100), dtype=np.uint8)
    img_dark_bg[40:60, 40:60] = 255
    assert detect_polarity(img_dark_bg) == "dark_bg"
    normalized_dark = normalize_polarity(img_dark_bg, target="white_on_black")
    assert normalized_dark[0, 0] == 0
    assert normalized_dark[50, 50] == 255


def test_binarize_methods():
    gray = np.zeros((50, 50), dtype=np.uint8)
    gray[20:30, 20:30] = 200

    bin_otsu = binarize(gray, method="otsu")
    assert bin_otsu.shape == (50, 50)
    assert set(np.unique(bin_otsu)).issubset({0, 255})

    bin_sauvola = binarize(gray, method="sauvola", sauvola_window=15, sauvola_k=0.2)
    assert bin_sauvola.shape == (50, 50)
    assert bin_sauvola[25, 25] == 255
    assert bin_sauvola[0, 0] == 0

    bin_fixed = binarize(gray, method="fixed", fixed_thresh=100)
    assert bin_fixed[25, 25] == 255
    assert bin_fixed[0, 0] == 0


def test_normalize_illumination():
    from src.preprocessing import normalize_illumination
    # Create image with uneven gradient illumination (bright on left, dark on right)
    gradient = np.tile(np.linspace(50, 220, 100, dtype=np.uint8), (100, 1)).T
    # Put a dark dot stroke in it
    gradient[45:55, 45:55] = 10

    normalized = normalize_illumination(gradient, blur_ksize=31)
    assert normalized.shape == (100, 100)
    assert normalized.dtype == np.uint8
    # Contrast is preserved and range is valid
    assert normalized.min() < normalized.max()


def test_crop_to_bounding_box():
    canvas = np.zeros((100, 100), dtype=np.uint8)
    # White box between rows 30..60, cols 40..70
    canvas[30:60, 40:70] = 255

    cropped, bbox = crop_to_bounding_box(canvas, pad=2)
    assert bbox is not None
    x, y, w, h = bbox
    assert x == 40 and y == 30 and w == 30 and h == 30
    assert cropped.shape[0] == 30 + 4  # h + 2*pad
    assert cropped.shape[1] == 30 + 4  # w + 2*pad


def test_resize_and_pad():
    char_crop = np.ones((40, 20), dtype=np.uint8) * 255  # 2:1 aspect ratio
    out = resize_and_pad(char_crop, target_size=(28, 28), fit_box_size=20)
    assert out.shape == (28, 28)
    assert out.dtype == np.uint8
    # Canvas outer border should be black padding
    assert out[0, 0] == 0
    assert out[27, 27] == 0
    # Center should contain the character stroke
    assert np.max(out) == 255


def test_to_model_tensor():
    dummy_28 = np.ones((28, 28), dtype=np.uint8) * 255
    tensor = to_model_tensor(dummy_28, device=torch.device("cpu"))
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 1, 28, 28)
    assert tensor.dtype == torch.float32
    assert torch.isclose(tensor.max(), torch.tensor(1.0))
    assert torch.isclose(tensor.min(), torch.tensor(1.0))


def test_to_classical_features():
    dummy_28 = np.ones((28, 28), dtype=np.uint8) * 128
    feat = to_classical_features(dummy_28)
    assert isinstance(feat, np.ndarray)
    assert feat.shape == (28, 28)
    assert feat.dtype == np.float32
    assert 0.49 <= feat[0, 0] <= 0.51


def test_full_preprocessing_pipeline():
    # Synthetic dark digit on white canvas
    img = np.ones((80, 80, 3), dtype=np.uint8) * 255
    cv2.circle(img, (40, 40), 15, (0, 0, 0), -1)

    norm_28, preview = preprocess_image(img)
    assert norm_28.shape == (28, 28)
    assert "original" in preview
    assert "binary" in preview
    assert "normalized_28x28" in preview
    assert preview["polarity_detected"] == "light_bg"
