"""
Unit tests for touching character ligature splitting.
Tests vertical projection profiling and ligature valley detection.
"""

import numpy as np
import pytest

from src.segmentation import (
    compute_vertical_projection_profile,
    split_touching_box,
    segment_characters
)


def test_vertical_projection_profile():
    patch = np.zeros((20, 30), dtype=np.uint8)
    # Column 5 has 10 stroke pixels
    patch[:10, 5] = 255
    # Column 15 has 4 stroke pixels
    patch[:4, 15] = 255

    vpp = compute_vertical_projection_profile(patch)
    assert len(vpp) == 30
    assert vpp[5] == 10
    assert vpp[15] == 4
    assert vpp[0] == 0


def test_split_touching_characters_synthetic():
    # Construct an image with two digits touching with a 1-pixel bridge
    # Left digit: box width ~20, height ~30
    # Right digit: box width ~20, height ~30
    # Ligature at x=25
    img = np.zeros((50, 70), dtype=np.uint8)

    # Left character body (col 10 to 22)
    img[10:40, 12:15] = 255
    img[10:13, 12:22] = 255
    img[25:28, 12:22] = 255

    # Right character body (col 28 to 40)
    img[10:40, 35:38] = 255
    img[10:13, 28:38] = 255
    img[37:40, 28:38] = 255

    # Thin ligature connection bridging left and right at col 24-25, 1-pixel row
    img[25, 22:28] = 255

    bbox = (10, 10, 35, 30)  # aspect ratio 35 / 30 > 1.15

    splits = split_touching_box(img, bbox, aspect_ratio_thresh=1.1, min_width=20)

    # Should split into 2 bounding boxes
    assert len(splits) == 2
    b1, b2 = splits
    # b1 is to the left of b2
    assert b1[0] < b2[0]


def test_do_not_split_normal_character():
    # Normal aspect ratio character (e.g. digit 0 or 8)
    img = np.zeros((50, 50), dtype=np.uint8)
    img[10:40, 15:35] = 255
    bbox = (15, 10, 20, 30)  # w=20, h=30 -> w/h = 0.67

    splits = split_touching_box(img, bbox, aspect_ratio_thresh=1.35)
    assert len(splits) == 1
    assert splits[0] == bbox
