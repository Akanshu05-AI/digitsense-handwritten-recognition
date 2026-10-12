"""
DigitSense: Image Preprocessing Module
Implements robust loading, color conversion, polarity detection, noise reduction,
Otsu/adaptive binarization, aspect-ratio-preserving resize, and centering.
"""

from typing import Union, Tuple, Dict, Any, Optional
import io
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch

from config import IMAGE_SIZE, GAUSSIAN_BLUR_KERNEL


def load_image(image_input: Union[str, Path, bytes, np.ndarray, Image.Image, Any]) -> np.ndarray:
    """
    Load and validate an image from various input types.
    Returns:
        np.ndarray: BGR or Grayscale uint8 image.
    Raises:
        ValueError: If input format is invalid or image cannot be decoded.
    """
    if image_input is None:
        raise ValueError("Image input cannot be None.")

    # Streamlit UploadedFile or file-like object with .read()
    if hasattr(image_input, "read"):
        file_bytes = image_input.read()
        if len(file_bytes) == 0:
            raise ValueError("Uploaded file buffer is empty.")
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Failed to decode image from file buffer.")
        return img

    # Raw bytes
    if isinstance(image_input, (bytes, bytearray)):
        if len(image_input) == 0:
            raise ValueError("Byte buffer is empty.")
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Failed to decode image from raw bytes.")
        return img

    # File path
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image from path: {path}")
        return img

    # PIL Image
    if isinstance(image_input, Image.Image):
        return np.array(image_input)

    # NumPy Array
    if isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ValueError("NumPy array is empty.")
        return image_input.copy()

    raise ValueError(f"Unsupported image input type: {type(image_input)}")


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert an image (BGR, RGB, RGBA, BGRA, or Gray) to single-channel uint8 grayscale.
    """
    if img is None or img.size == 0:
        raise ValueError("Cannot convert empty image to grayscale.")

    if len(img.shape) == 2:
        return img.astype(np.uint8)

    if len(img.shape) == 3:
        channels = img.shape[2]
        if channels == 1:
            return img[:, :, 0].astype(np.uint8)
        elif channels == 3:
            # Handle standard BGR / RGB
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            # Handle alpha channel (common in drawing canvases and PNGs)
            # If transparent background is present, blend against white background
            alpha = img[:, :, 3] / 255.0
            bgr = img[:, :, :3]
            white_bg = np.ones_like(bgr, dtype=np.float32) * 255.0
            blended = (alpha[:, :, None] * bgr.astype(np.float32) + (1.0 - alpha[:, :, None]) * white_bg).astype(np.uint8)
            return cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Unexpected image shape for grayscale conversion: {img.shape}")


def enhance_contrast(gray_img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Enhance local contrast for low-contrast webcam snapshots, faint handwriting,
    and textured surfaces using Contrast Limited Adaptive Histogram Equalization (CLAHE).
    """
    if gray_img is None or gray_img.size == 0:
        return gray_img

    p_low, p_high = np.percentile(gray_img, (2, 98))
    if p_high > p_low + 10:
        stretched = np.clip((gray_img.astype(np.float32) - p_low) / (p_high - p_low) * 255.0, 0, 255).astype(np.uint8)
    else:
        stretched = gray_img

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(stretched)


def detect_polarity(gray_img: np.ndarray) -> str:
    """
    Detect whether the background is light (dark text on white) or dark (white text on black).
    Sample the perimeter border pixels to estimate the dominant background intensity.
    Returns:
        'light_bg': White/light background with dark strokes.
        'dark_bg': Black/dark background with light strokes.
    """
    h, w = gray_img.shape
    border_pixels = np.concatenate([
        gray_img[0, :],         # top row
        gray_img[-1, :],        # bottom row
        gray_img[:, 0],         # left col
        gray_img[:, -1]         # right col
    ])
    median_border = np.median(border_pixels)
    if median_border > 127:
        return "light_bg"
    return "dark_bg"


def normalize_polarity(gray_img: np.ndarray, target: str = "white_on_black") -> np.ndarray:
    """
    Ensure the strokes match the standard MNIST/EMNIST representation:
    target="white_on_black" (bright character strokes = 255, dark background = 0).
    """
    polarity = detect_polarity(gray_img)
    if target == "white_on_black":
        if polarity == "light_bg":
            # Invert: white background (255) becomes 0, dark stroke becomes bright
            return cv2.bitwise_not(gray_img)
        return gray_img
    else:  # black_on_white
        if polarity == "dark_bg":
            return cv2.bitwise_not(gray_img)
        return gray_img


def denoise(gray_img: np.ndarray, method: str = "gaussian", ksize: int = 3) -> np.ndarray:
    """
    Apply noise reduction to grayscale image.
    Options: 'gaussian', 'median', 'bilateral', 'none'
    """
    if method == "none" or ksize <= 1:
        return gray_img.copy()

    ksize = ksize if ksize % 2 == 1 else ksize + 1  # ensure odd kernel

    if method == "gaussian":
        return cv2.GaussianBlur(gray_img, (ksize, ksize), 0)
    elif method == "median":
        return cv2.medianBlur(gray_img, ksize)
    elif method == "bilateral":
        return cv2.bilateralFilter(gray_img, d=ksize, sigmaColor=75, sigmaSpace=75)
    return gray_img


def normalize_illumination(gray_img: np.ndarray, blur_ksize: int = 51) -> np.ndarray:
    """
    Remove uneven illumination gradients, flash glare, and shadows
    via background division (flat-field correction).
    Estimates background illumination and normalizes local contrast.
    """
    if gray_img is None or gray_img.size == 0:
        return gray_img

    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    # Estimate background illumination using large Gaussian filter
    bg = cv2.GaussianBlur(gray_img, (ksize, ksize), 0).astype(np.float32)
    bg = np.maximum(bg, 1.0)  # Avoid division by zero

    # Background division normalized to constant mean intensity
    mean_val = float(np.mean(bg))
    norm = (gray_img.astype(np.float32) / bg) * mean_val
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    return norm


def binarize_sauvola(
    gray_img: np.ndarray,
    window_size: int = 25,
    k: float = 0.2,
    r: float = 128.0,
    is_white_on_black: bool = True
) -> np.ndarray:
    """
    Sauvola's local adaptive thresholding.
    Optimal for degraded documents, uneven lighting, and variable stroke thickness.

    Formula:
        T(x, y) = m(x, y) * [1 + k * (s(x, y) / R - 1)]

    Args:
        gray_img: Grayscale uint8 image.
        window_size: Local neighborhood window size (odd integer, e.g. 25).
        k: Dynamic range threshold multiplier (typical range: 0.1 to 0.3, default: 0.2).
        r: Dynamic range of standard deviation (default: 128.0 for 8-bit images).
        is_white_on_black: True if foreground stroke is bright on dark background.

    Returns:
        binary (np.ndarray): Binary image with foreground strokes = 255 and background = 0.
    """
    w = window_size if window_size % 2 == 1 else window_size + 1

    # Map to classic dark-ink on light-paper representation for canonical Sauvola formula
    if is_white_on_black:
        src = 255.0 - gray_img.astype(np.float32)
    else:
        src = gray_img.astype(np.float32)

    # Compute local mean m and variance var via fast boxFilter
    m = cv2.boxFilter(src, ddepth=-1, ksize=(w, w), borderType=cv2.BORDER_REFLECT)
    m2 = cv2.boxFilter(src * src, ddepth=-1, ksize=(w, w), borderType=cv2.BORDER_REFLECT)
    var = np.maximum(0.0, m2 - m * m)
    s = np.sqrt(var)

    # Sauvola threshold: ink is strictly darker than surface T
    thresh = m * (1.0 + k * (s / r - 1.0))

    # Output foreground strokes as 255, background as 0
    binary = np.where(src <= thresh, 255, 0).astype(np.uint8)
    return binary


def binarize(
    gray_img: np.ndarray,
    method: str = "otsu",
    fixed_thresh: int = 127,
    sauvola_window: int = 25,
    sauvola_k: float = 0.2
) -> np.ndarray:
    """
    Binarize grayscale image to binary (values 0 and 255).
    Assumes polarity is white strokes on black background.
    Methods: 'otsu', 'adaptive', 'sauvola', 'fixed'
    """
    if method == "otsu":
        # Otsu's thresholding automatically computes optimal threshold
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return binary
    elif method == "sauvola":
        return binarize_sauvola(gray_img, window_size=sauvola_window, k=sauvola_k, is_white_on_black=True)
    elif method == "fixed":
        _, binary = cv2.threshold(gray_img, fixed_thresh, 255, cv2.THRESH_BINARY)
        return binary
    raise ValueError(f"Unknown thresholding method: {method}")


def crop_to_bounding_box(
    binary_img: np.ndarray,
    pad: int = 4
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """
    Crop tightly around foreground white strokes (pixel value > 0).
    Returns cropped image and (x, y, w, h) bounding box.
    If image contains no foreground, returns original image and None.
    """
    coords = cv2.findNonZero(binary_img)
    if coords is None:
        return binary_img.copy(), None

    x, y, w, h = cv2.boundingRect(coords)
    img_h, img_w = binary_img.shape

    # Apply padding while respecting image bounds
    x_start = max(0, x - pad)
    y_start = max(0, y - pad)
    x_end = min(img_w, x + w + pad)
    y_end = min(img_h, y + h + pad)

    cropped = binary_img[y_start:y_end, x_start:x_end]
    return cropped, (x, y, w, h)


def center_by_center_of_mass(img_28x28: np.ndarray) -> np.ndarray:
    """
    Center an isolated 28x28 digit image by its center of mass,
    matching the exact preprocessing used in the original MNIST dataset.
    """
    moments = cv2.moments(img_28x28)
    if moments["m00"] == 0:
        return img_28x28  # Empty image

    # Center of mass coordinates
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    # Target center is (14, 14) for 28x28
    shift_x = np.round(14.0 - cx)
    shift_y = np.round(14.0 - cy)

    # Translation matrix
    m = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    centered = cv2.warpAffine(img_28x28, m, (28, 28), borderValue=0)
    return centered


def resize_and_pad(
    img: np.ndarray,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    fit_box_size: int = 20,
    use_center_of_mass: bool = True
) -> np.ndarray:
    """
    Resize character preserving aspect ratio into a fit_box (default 20x20),
    and center onto target canvas (default 28x28).
    This strictly matches Yann LeCun's MNIST standard pipeline.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros(target_size, dtype=np.uint8)

    # Compute scaling factor to fit into fit_box_size x fit_box_size
    scale = fit_box_size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Rescale using linear or area interpolation
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

    # Create empty black target canvas
    canvas = np.zeros(target_size, dtype=np.uint8)

    # Calculate initial top-left placing in the center of 28x28
    pad_y = (target_size[0] - new_h) // 2
    pad_x = (target_size[1] - new_w) // 2

    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    if use_center_of_mass:
        canvas = center_by_center_of_mass(canvas)

    return canvas


def preprocess_image(
    image_input: Union[str, Path, bytes, np.ndarray, Image.Image, Any],
    denoise_method: str = "gaussian",
    denoise_ksize: int = 3,
    threshold_method: str = "otsu",
    fixed_thresh: int = 127,
    sauvola_window: int = 25,
    sauvola_k: float = 0.2,
    normalize_light: bool = False,
    enhance_contrast_flag: bool = False,
    pad: int = 4
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Full end-to-end preprocessing pipeline for a single character input.
    Returns:
        normalized_28x28 (np.ndarray): 28x28 uint8 array (strokes=255, bg=0).
        pipeline_preview (dict): dictionary of intermediate representations for inspection.
    """
    # 1. Load image
    raw = load_image(image_input)

    # 2. Convert to grayscale
    gray = to_grayscale(raw)

    # Optional CLAHE contrast enhancement for low contrast/webcam captures
    if enhance_contrast_flag:
        gray = enhance_contrast(gray)

    # Optional Illumination Normalization for camera/shadow photos
    if normalize_light:
        gray = normalize_illumination(gray)

    # 3. Detect and correct polarity (normalize to white strokes on black background)
    polarity = detect_polarity(gray)
    normalized_pol = normalize_polarity(gray, target="white_on_black")

    # 4. Noise reduction
    denoised = denoise(normalized_pol, method=denoise_method, ksize=denoise_ksize)

    # 5. Binarize
    binary = binarize(
        denoised,
        method=threshold_method,
        fixed_thresh=fixed_thresh,
        sauvola_window=sauvola_window,
        sauvola_k=sauvola_k
    )

    # 6. Crop foreground
    cropped, bbox = crop_to_bounding_box(binary, pad=pad)

    # 7. Aspect-ratio-preserving resize and center on 28x28 canvas
    normalized_28x28 = resize_and_pad(cropped, target_size=IMAGE_SIZE, fit_box_size=20, use_center_of_mass=True)

    preview = {
        "original": raw,
        "grayscale": gray,
        "polarity_detected": polarity,
        "polarity_corrected": normalized_pol,
        "denoised": denoised,
        "binary": binary,
        "cropped": cropped,
        "bbox": bbox,
        "normalized_28x28": normalized_28x28
    }

    return normalized_28x28, preview


def to_model_tensor(img_28x28: np.ndarray, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Convert a 28x28 uint8 numpy array [0, 255] into a PyTorch model tensor
    of shape (1, 1, 28, 28), float32, normalized to range [0.0, 1.0].
    """
    if img_28x28.shape != (28, 28):
        raise ValueError(f"Expected image of shape (28, 28), got {img_28x28.shape}")

    tensor = torch.from_numpy(img_28x28).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
    return tensor.to(device)


def to_classical_features(img_28x28: np.ndarray) -> np.ndarray:
    """
    Convert 28x28 uint8 array to float32 [0.0, 1.0] for classical feature extraction.
    """
    if img_28x28.shape != (28, 28):
        raise ValueError(f"Expected image of shape (28, 28), got {img_28x28.shape}")
    return img_28x28.astype(np.float32) / 255.0
