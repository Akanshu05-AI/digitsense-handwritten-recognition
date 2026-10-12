"""
DigitSense: Multi-Character Segmentation Module
Extracts individual character bounding boxes, filters noise, orders them in reading
order (left-to-right), applies LeCun-style normalization, and creates annotations.
"""

from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np

from config import SEGMENTATION_MIN_AREA, SEGMENTATION_MAX_AREA_RATIO, SEGMENTATION_PADDING, IMAGE_SIZE
from src.preprocessing import (
    load_image,
    to_grayscale,
    normalize_illumination,
    normalize_polarity,
    denoise,
    binarize,
    resize_and_pad,
    enhance_contrast
)


def compute_vertical_projection_profile(binary_patch: np.ndarray) -> np.ndarray:
    """
    Compute vertical projection profile V(x) = sum_y (binary_patch(y, x) > 0).
    Represents stroke density per horizontal column.
    """
    return np.sum(binary_patch > 0, axis=0)


def split_touching_box(
    binary_img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    aspect_ratio_thresh: float = 1.35,
    min_width: int = 28
) -> List[Tuple[int, int, int, int]]:
    """
    Detect and split touching handwritten characters using Vertical Projection Profile (VPP).
    Identifies the stroke density minimum (valley) in the central ligature region.

    Args:
        binary_img: Full binarized image (white strokes = 255, background = 0).
        bbox: Candidate bounding box (x, y, w, h).
        aspect_ratio_thresh: Ratio (w / h) above which a box is evaluated for ligature split.
        min_width: Minimum pixel width to consider splitting.

    Returns:
        List of 1 or more bounding boxes.
    """
    x, y, w, h = bbox
    # If aspect ratio is normal, do not split
    if w < min_width or (w / max(1, h)) < aspect_ratio_thresh:
        return [bbox]

    patch = binary_img[y:y+h, x:x+w]
    if patch.size == 0:
        return [bbox]

    vpp = compute_vertical_projection_profile(patch).astype(np.float32)

    # Smooth the profile with a 1D uniform filter to remove stroke speckle
    kernel_size = 5
    if len(vpp) > kernel_size:
        kernel = np.ones(kernel_size) / kernel_size
        vpp_smooth = np.convolve(vpp, kernel, mode="same")
    else:
        vpp_smooth = vpp

    # Search in central zone: between 25% and 75% of box width
    search_start = max(6, int(w * 0.25))
    search_end = min(w - 6, int(w * 0.75))

    if search_start >= search_end:
        return [bbox]

    central_vpp = vpp_smooth[search_start:search_end]
    min_idx_relative = int(np.argmin(central_vpp))
    split_col = search_start + min_idx_relative

    valley_val = vpp_smooth[split_col]
    left_peak = float(np.max(vpp_smooth[:split_col])) if split_col > 0 else 1.0
    right_peak = float(np.max(vpp_smooth[split_col:])) if split_col < w else 1.0
    min_flanking_peak = min(left_peak, right_peak)

    # Valley depth criterion:
    # 1. Valley must be significantly lower than flanking peaks (<= 75% of min flanking peak), OR
    # 2. Valley must have 0 or very few stroke pixels (<= 2 pixels)
    is_valid_valley = (
        (valley_val <= 0.75 * min_flanking_peak) or
        (valley_val <= 2.0)
    )

    # Ensure both sides have enough width to be valid characters
    width_ok = (split_col >= 8) and ((w - split_col) >= 8)

    if is_valid_valley and width_ok:
        left_sub = patch[:, :split_col]
        right_sub = patch[:, split_col:]

        coords_l = cv2.findNonZero(left_sub)
        coords_r = cv2.findNonZero(right_sub)

        if coords_l is not None and coords_r is not None:
            lx, ly, lw, lh = cv2.boundingRect(coords_l)
            rx, ry, rw, rh = cv2.boundingRect(coords_r)
            box_left = (x + lx, y + ly, lw, lh)
            box_right = (x + split_col + rx, y + ry, rw, rh)
            # Recursively check right half in case of 3 touching characters
            right_splits = split_touching_box(binary_img, box_right, aspect_ratio_thresh, min_width)
            return [box_left] + right_splits

    return [bbox]


def compute_box_overlap(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> Dict[str, Any]:
    """
    Compute intersection, horizontal overlap, and vertical overlap metrics between two bounding boxes.
    """
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b

    x_left = max(xa, xb)
    x_right = min(xa + wa, xb + wb)
    x_overlap = max(0, x_right - x_left)

    y_top = max(ya, yb)
    y_bottom = min(ya + ha, yb + hb)
    y_overlap = max(0, y_bottom - y_top)

    intersection = x_overlap * y_overlap
    area_a = wa * ha
    area_b = wb * hb

    return {
        "x_overlap": x_overlap,
        "y_overlap": y_overlap,
        "intersection": intersection,
        "min_w": min(wa, wb),
        "min_h": min(ha, hb),
        "max_w": max(wa, wb),
        "max_h": max(ha, hb),
        "min_area": min(area_a, area_b),
        "max_area": max(area_a, area_b)
    }


def merge_two_boxes(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """
    Compute the union bounding box enclosing both box_a and box_b.
    """
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b
    x_new = min(xa, xb)
    y_new = min(ya, yb)
    w_new = max(xa + wa, xb + wb) - x_new
    h_new = max(ya + ha, yb + hb) - y_new
    return (x_new, y_new, w_new, h_new)


def merge_overlapping_boxes(
    boxes: List[Tuple[int, int, int, int]],
    max_merged_aspect_ratio: float = 1.45,
    x_gap_thresh: int = 4
) -> List[Tuple[int, int, int, int]]:
    """
    Merge fragmented strokes, multi-stroke characters, accents, and interior islands
    into cohesive character bounding boxes.

    Criteria for merging:
    1. Containment / Interior Island: One box is inside or substantially overlapping another.
    2. Vertical Alignment / Column Sharing: High horizontal overlap (e.g., dots of 'i', 'j',
       crossbars, vertical stroke fragments) where y_gap is small and merged aspect ratio is valid.
    3. Partial Stroke Fragment: A small stroke fragment (h < 55% of character height) horizontally
       adjacent with vertical overlap (e.g. base of 'L' with stem, crossbar of 'T').
    4. Twin Thin Legs: Broken stencil legs of letters like 'A', 'H', 'M', 'N' with small gap.
    """
    if len(boxes) <= 1:
        return boxes

    merged = list(boxes)
    changed = True

    while changed:
        changed = False
        n = len(merged)
        i = 0
        while i < n:
            j = i + 1
            while j < n:
                box_a = merged[i]
                box_b = merged[j]

                ov = compute_box_overlap(box_a, box_b)
                xa, ya, wa, ha = box_a
                xb, yb, wb, hb = box_b

                # Condition 1: Containment / heavy intersection
                is_contained = (
                    (ov["intersection"] >= 0.70 * ov["min_area"]) or
                    (xa <= xb + 2 and xa + wa >= xb + wb - 2 and ya <= yb + 2 and ya + ha >= yb + hb - 2) or
                    (xb <= xa + 2 and xb + wb >= xa + wa - 2 and yb <= ya + 2 and yb + hb >= ya + ha - 2)
                )

                candidate_merged = merge_two_boxes(box_a, box_b)
                _, _, mw, mh = candidate_merged
                merged_aspect = mw / max(1, mh)

                # Condition 2: High horizontal overlap (dots, diacritics, column fragments)
                h_overlap_ratio = ov["x_overlap"] / max(1, ov["min_w"])
                y_gap = max(0, max(ya, yb) - min(ya + ha, yb + hb))
                is_vertically_aligned = (
                    (h_overlap_ratio > 0.40) and
                    (y_gap <= 0.40 * max(ha, hb)) and
                    (merged_aspect <= max_merged_aspect_ratio)
                )

                # Condition 3: Disconnected partial stroke fragment merging with main stem
                x_gap = max(0, max(xa, xb) - min(xa + wa, xb + wb))
                is_sub_fragment = (min(ha, hb) < 0.55 * max(ha, hb))
                is_adjacent_fragment = (
                    (x_gap <= max(x_gap_thresh, int(0.12 * max(ha, hb)))) and
                    (ov["y_overlap"] > 0) and
                    (merged_aspect <= max_merged_aspect_ratio) and
                    is_sub_fragment
                )

                # Condition 4: Twin thin vertical legs (e.g. broken stencil legs of 'A', 'H')
                is_twin_thin_legs = (
                    (wa < 0.32 * ha) and (wb < 0.32 * hb) and
                    (x_gap <= max(3, int(0.15 * max(ha, hb)))) and
                    (ov["y_overlap"] >= 0.65 * min(ha, hb)) and
                    (merged_aspect <= 1.15)
                )

                if is_contained or is_vertically_aligned or is_adjacent_fragment or is_twin_thin_legs:
                    merged[i] = candidate_merged
                    merged.pop(j)
                    changed = True
                    n -= 1
                    break
                else:
                    j += 1
            if changed:
                break
            i += 1

    return merged


def filter_noise_boxes(
    boxes: List[Tuple[int, int, int, int]],
    min_relative_height: float = 0.35,
    min_relative_area: float = 0.20
) -> List[Tuple[int, int, int, int]]:
    """
    Scale-adaptive noise rejection.
    Discards stray sensor noise, lint, and fabric weave specks whose dimensions
    are significantly below the median character height and area.
    """
    if len(boxes) <= 2:
        return boxes

    heights = [h for _, _, _, h in boxes]
    areas = [w * h for _, _, w, h in boxes]
    med_h = float(np.median(heights))
    med_area = float(np.median(areas))

    filtered = []
    for b in boxes:
        _, _, w, h = b
        area = w * h
        # Keep if height is at least 35% of median OR area is at least 20% of median
        if (h >= min_relative_height * med_h) or (area >= min_relative_area * med_area):
            filtered.append(b)

    return filtered if filtered else boxes


def segment_characters(
    image_input: Any,
    min_area: int = SEGMENTATION_MIN_AREA,
    max_area_ratio: float = SEGMENTATION_MAX_AREA_RATIO,
    padding: int = SEGMENTATION_PADDING,
    denoise_method: str = "gaussian",
    denoise_ksize: int = 3,
    threshold_method: str = "otsu",
    fixed_thresh: int = 127,
    sauvola_window: int = 25,
    sauvola_k: float = 0.2,
    normalize_light: bool = False,
    split_ligatures: bool = True,
    morph_close: bool = True,
    morph_ksize: int = 3,
    merge_fragments: bool = True,
    adaptive_noise_filter: bool = True,
    enhance_contrast_flag: bool = False
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], np.ndarray, Dict[str, Any]]:
    """
    Segment handwritten characters from an input image.

    Workflow:
    1. Preprocesses image to binary with white strokes on black background.
    2. Applies optional CLAHE contrast enhancement and illumination correction.
    3. Bridges micro-gaps and stroke discontinuities using morphological closing.
    4. Uses cv2.findContours with RETR_EXTERNAL to detect candidate character regions.
    5. Filters noise (scale-adaptive min_area) and full-image canvas borders.
    6. Merges overlapping or fragmented strokes (accents, dots, broken stems).
    7. Discards stray background sensor/fabric texture speckles (scale-adaptive noise filter).
    8. Checks and splits touching characters (ligatures) via Vertical Projection Profile.
    9. Sorts bounding boxes in reading order (top-to-bottom, left-to-right).
    10. Crops each candidate region and normalizes to 28x28 canvas (LeCun style).
    11. Produces an annotated RGB visualization image showing numbered bounding boxes.

    Returns:
        crops (List[np.ndarray]): list of 28x28 normalized character arrays.
        bboxes (List[Tuple[int, int, int, int]]): list of (x, y, w, h) bounding boxes.
        annotated_image (np.ndarray): RGB visualization with drawn boxes and labels.
        metadata (Dict[str, Any]): segmentation statistics, flags, and warnings.
    """
    # Load and prepare image
    raw = load_image(image_input)
    if len(raw.shape) == 2:
        vis_image = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    elif raw.shape[2] == 4:
        # Convert RGBA to RGB with white background
        alpha = raw[:, :, 3] / 255.0
        rgb = raw[:, :, :3]
        white_bg = np.ones_like(rgb, dtype=np.float32) * 255.0
        vis_image = (alpha[:, :, None] * rgb.astype(np.float32) + (1.0 - alpha[:, :, None]) * white_bg).astype(np.uint8)
    else:
        vis_image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB) if len(raw.shape) == 3 else raw.copy()

    img_h, img_w = vis_image.shape[:2]
    total_image_area = img_h * img_w

    # Convert to grayscale and normalize polarity (white stroke on black background)
    gray = to_grayscale(raw)
    if enhance_contrast_flag:
        gray = enhance_contrast(gray)
    if normalize_light:
        gray = normalize_illumination(gray)

    white_on_black = normalize_polarity(gray, target="white_on_black")
    denoised = denoise(white_on_black, method=denoise_method, ksize=denoise_ksize)
    binary = binarize(
        denoised,
        method=threshold_method,
        fixed_thresh=fixed_thresh,
        sauvola_window=sauvola_window,
        sauvola_k=sauvola_k
    )

    # Optional morphological closing to bridge stroke gaps, dotted fonts, and intra-character breaks
    if morph_close and morph_ksize >= 3:
        ksize = morph_ksize if morph_ksize % 2 == 1 else morph_ksize + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        contour_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    else:
        contour_binary = binary

    # Find external contours
    contours, _ = cv2.findContours(contour_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_boxes = []
    warnings = []

    # Scale-adaptive minimum area threshold
    effective_min_area = max(min_area, int(total_image_area * 0.0001))

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box_area = w * h

        # Filter out noise dots
        if box_area < effective_min_area:
            continue

        # Filter out image frame borders covering practically the whole canvas
        if box_area / total_image_area > max_area_ratio:
            continue

        # Ignore thin border slices touching all 4 outer edges
        if (w >= img_w - 2) and (h >= img_h - 2):
            continue

        candidate_boxes.append((x, y, w, h))

    # Merge multi-stroke fragments, interior loops, and vertically aligned stroke parts
    if merge_fragments and len(candidate_boxes) > 1:
        num_before = len(candidate_boxes)
        candidate_boxes = merge_overlapping_boxes(candidate_boxes)
        if len(candidate_boxes) < num_before:
            warnings.append(
                f"Consolidated {num_before - len(candidate_boxes)} multi-stroke/fragmented character regions."
            )

    # Scale-adaptive noise rejection to eliminate small fabric weave/lint/sensor specks
    if adaptive_noise_filter and len(candidate_boxes) > 2:
        num_before = len(candidate_boxes)
        candidate_boxes = filter_noise_boxes(candidate_boxes)
        if len(candidate_boxes) < num_before:
            warnings.append(
                f"Filtered {num_before - len(candidate_boxes)} background noise/texture artifacts."
            )

    # Apply vertical projection profile ligature splitting if touching characters are present
    if split_ligatures and candidate_boxes:
        refined_boxes = []
        for box in candidate_boxes:
            splits = split_touching_box(binary, box)
            if len(splits) > 1:
                warnings.append(f"Detected and split touching character ligature into {len(splits)} sub-characters.")
            refined_boxes.extend(splits)
        candidate_boxes = refined_boxes

    if not candidate_boxes:
        # Fallback: if no valid contours were found, check if whole image has some strokes
        coords = cv2.findNonZero(binary)
        if coords is not None and len(coords) > 10:
            x, y, w, h = cv2.boundingRect(coords)
            if w * h >= effective_min_area:
                candidate_boxes.append((x, y, w, h))
                warnings.append("No distinct contours found; used entire foreground bounding box.")
            else:
                warnings.append("No significant handwriting strokes detected (content area too small).")
        else:
            warnings.append("No handwriting strokes detected in image.")

    # Sort bounding boxes in reading order: top-to-bottom, left-to-right
    sorted_boxes = sort_bounding_boxes(candidate_boxes, line_threshold=0.5)

    crops = []
    final_boxes = []

    for idx, (x, y, w, h) in enumerate(sorted_boxes):
        # Heuristic for touching characters remaining: width is still disproportionately larger than height
        if w > 2.0 * h and w > 40:
            warnings.append(
                f"Region #{idx+1} (x={x}, y={y}, w={w}, h={h}) has wide aspect ratio; possible touching characters."
            )

        # Crop from binary image with padding
        x_pad_min = max(0, x - padding)
        y_pad_min = max(0, y - padding)
        x_pad_max = min(img_w, x + w + padding)
        y_pad_max = min(img_h, y + h + padding)

        crop_bin = binary[y_pad_min:y_pad_max, x_pad_min:x_pad_max]

        # Normalize to 28x28 LeCun style
        norm_28 = resize_and_pad(crop_bin, target_size=IMAGE_SIZE, fit_box_size=20, use_center_of_mass=True)

        crops.append(norm_28)
        final_boxes.append((x, y, w, h))

        # Annotate image with high-contrast bounding box and badge
        # Bounding box in vibrant emerald green
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 200, 115), 2)

        # Badge pill for index number
        badge_text = str(idx + 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(badge_text, font, font_scale, thickness)

        badge_x = x
        badge_y = max(text_h + 4, y - 4)
        cv2.rectangle(
            vis_image,
            (badge_x, badge_y - text_h - 4),
            (badge_x + text_w + 6, badge_y + 2),
            (0, 200, 115),
            -1
        )
        cv2.putText(
            vis_image,
            badge_text,
            (badge_x + 3, badge_y - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    metadata = {
        "num_characters_detected": len(crops),
        "image_dimensions": (img_w, img_h),
        "warnings": warnings,
        "is_multi_character": len(crops) > 1,
        "is_empty": len(crops) == 0
    }

    return crops, final_boxes, vis_image, metadata


def sort_bounding_boxes(
    boxes: List[Tuple[int, int, int, int]],
    line_threshold: float = 0.5
) -> List[Tuple[int, int, int, int]]:
    """
    Sort bounding boxes primarily left-to-right.
    If multiple lines of writing exist, groups by horizontal lines first (top-to-bottom),
    then sorts each line left-to-right.
    """
    if not boxes:
        return []

    if len(boxes) == 1:
        return boxes

    # Calculate median box height to determine line-clustering distance
    median_height = float(np.median([b[3] for b in boxes]))
    y_threshold = median_height * line_threshold

    # Sort primarily by y-coordinate
    sorted_by_y = sorted(boxes, key=lambda b: b[1])

    # Cluster into lines
    lines: List[List[Tuple[int, int, int, int]]] = []
    current_line: List[Tuple[int, int, int, int]] = [sorted_by_y[0]]

    for box in sorted_by_y[1:]:
        curr_line_center_y = sum(b[1] + b[3] / 2.0 for b in current_line) / len(current_line)
        box_center_y = box[1] + box[3] / 2.0

        line_min_y = min(b[1] for b in current_line)
        line_max_y = max(b[1] + b[3] for b in current_line)
        v_overlap = max(0, min(box[1] + box[3], line_max_y) - max(box[1], line_min_y))

        is_same_line = (
            (v_overlap >= 0.25 * box[3]) or
            (abs(box_center_y - curr_line_center_y) < y_threshold)
        )

        if is_same_line:
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]
    lines.append(current_line)

    # Sort each line left-to-right by x coordinate
    ordered_boxes = []
    for line in lines:
        line_sorted = sorted(line, key=lambda b: b[0])
        ordered_boxes.extend(line_sorted)

    return ordered_boxes
