"""
Unit Tests for Multi-Character Segmentation Module
Covers single character, multi-character left-to-right ordering,
noise dot rejection, and bounding box grouping.
"""

import pytest
import numpy as np
import cv2

from src.segmentation import (
    segment_characters,
    sort_bounding_boxes,
    merge_overlapping_boxes,
    filter_noise_boxes
)


def test_segment_empty_image():
    # Blank white image (no handwriting strokes)
    blank = np.ones((100, 100, 3), dtype=np.uint8) * 255
    crops, bboxes, vis, meta = segment_characters(blank)
    assert meta["is_empty"] is True
    assert len(crops) == 0
    assert len(bboxes) == 0
    assert len(meta["warnings"]) > 0


def test_segment_single_character():
    # White image with a single dark circle in center
    img = np.ones((120, 120, 3), dtype=np.uint8) * 255
    cv2.circle(img, (60, 60), 20, (20, 20, 20), -1)

    crops, bboxes, vis, meta = segment_characters(img)
    assert len(crops) == 1
    assert len(bboxes) == 1
    assert crops[0].shape == (28, 28)
    assert meta["is_multi_character"] is False


def test_segment_multi_character_left_to_right():
    # Synthetic multi-character image with three digits: '4', '7', '9'
    canvas = np.ones((100, 300, 3), dtype=np.uint8) * 255

    # Draw three distinct blocks left to right
    # Character 1 at x = 40
    cv2.rectangle(canvas, (40, 30), (70, 70), (0, 0, 0), -1)
    # Character 2 at x = 130
    cv2.rectangle(canvas, (130, 30), (160, 70), (0, 0, 0), -1)
    # Character 3 at x = 220
    cv2.rectangle(canvas, (220, 30), (250, 70), (0, 0, 0), -1)

    crops, bboxes, vis, meta = segment_characters(canvas)
    assert len(crops) == 3
    assert len(bboxes) == 3
    assert meta["is_multi_character"] is True

    # Validate strict left-to-right ordering
    x_coords = [b[0] for b in bboxes]
    assert x_coords[0] < x_coords[1] < x_coords[2]


def test_segment_noise_filtering():
    # Canvas with one real character and several tiny single-pixel or 2x2 noise specks
    canvas = np.ones((100, 100, 3), dtype=np.uint8) * 255

    # Real character
    cv2.rectangle(canvas, (35, 35), (65, 65), (0, 0, 0), -1)

    # Noise specks (area < 25)
    canvas[10:12, 10:12] = 0
    canvas[80:82, 80:82] = 0

    crops, bboxes, vis, meta = segment_characters(canvas, min_area=30)
    # Only the real character should be extracted, noise rejected
    assert len(crops) == 1
    assert len(bboxes) == 1


def test_sort_bounding_boxes_multiline():
    # Two lines of text: line 1 (y~20), line 2 (y~80)
    # Unsorted input
    boxes = [
        (100, 80, 20, 30),  # Line 2, right
        (20, 20, 20, 30),   # Line 1, left
        (80, 20, 20, 30),   # Line 1, right
        (30, 80, 20, 30)    # Line 2, left
    ]
    sorted_boxes = sort_bounding_boxes(boxes, line_threshold=0.5)

    assert sorted_boxes[0] == (20, 20, 20, 30)
    assert sorted_boxes[1] == (80, 20, 20, 30)
    assert sorted_boxes[2] == (30, 80, 20, 30)
    assert sorted_boxes[3] == (100, 80, 20, 30)


def test_merge_overlapping_boxes_fragments():
    # Test merging of an L character split into vertical stem and horizontal foot
    stem = (100, 50, 12, 50)  # x=100, y=50, w=12, h=50
    foot = (100, 90, 30, 10)  # x=100, y=90, w=30, h=10 (foot under stem)
    boxes = [stem, foot]

    merged = merge_overlapping_boxes(boxes)
    assert len(merged) == 1
    # Merged bounding box should encompass both: x=100, y=50, w=30, h=50
    assert merged[0] == (100, 50, 30, 50)


def test_merge_overlapping_boxes_twin_legs():
    # Test merging of broken stencil legs of 'A' (two thin vertical legs side-by-side)
    leg_left = (50, 40, 6, 45)   # thin vertical stroke (aspect 6/45 = 0.13)
    leg_right = (58, 40, 6, 45)  # thin vertical stroke separated by 2px gap
    boxes = [leg_left, leg_right]

    merged = merge_overlapping_boxes(boxes)
    assert len(merged) == 1
    assert merged[0] == (50, 40, 14, 45)


def test_merge_overlapping_boxes_preserves_adjacent_distinct_characters():
    # Test that two distinct full-height adjacent characters (e.g. 'I' and 'Y') are NOT merged
    char_i = (100, 50, 12, 50)  # 'I': thin full-height letter
    char_y = (116, 50, 45, 50)  # 'Y': wide full-height letter, gap = 4px
    boxes = [char_i, char_y]

    merged = merge_overlapping_boxes(boxes)
    assert len(merged) == 2
    assert char_i in merged
    assert char_y in merged


def test_filter_noise_boxes_scale_adaptive():
    # When multiple full characters exist (e.g. height ~50, area ~2000),
    # stray sensor/fabric speckles (height ~8, area ~40) must be discarded.
    real_chars = [
        (50, 50, 40, 50),
        (100, 50, 40, 50),
        (150, 50, 40, 50)
    ]
    noise_dot1 = (10, 20, 6, 6)   # area 36, height 6
    noise_dot2 = (220, 80, 8, 8)  # area 64, height 8
    all_boxes = real_chars + [noise_dot1, noise_dot2]

    filtered = filter_noise_boxes(all_boxes)
    assert len(filtered) == 3
    assert noise_dot1 not in filtered
    assert noise_dot2 not in filtered


def test_segment_characters_broken_strokes_consolidation():
    # Canvas with broken letter (e.g. two strokes forming an L with a 2px gap)
    canvas = np.zeros((120, 120, 3), dtype=np.uint8)
    # Vertical bar of L
    canvas[20:70, 30:38] = 255
    # Horizontal bar of L with small gap
    canvas[66:72, 38:60] = 255

    crops, bboxes, vis, meta = segment_characters(canvas, merge_fragments=True, morph_close=True)
    # Should consolidate into 1 character bounding box
    assert len(bboxes) == 1
    assert len(crops) == 1

