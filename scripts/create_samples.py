"""
DigitSense: Sample Image Generator
Creates realistic synthetic handwriting demo samples for testing
single-character and multi-character recognition.
"""

from pathlib import Path
import numpy as np
import cv2

OUTPUT_DIR = Path("data/samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_styled_char(canvas, char, x, y, scale=2.0, thickness=4):
    """Draw a character using Hershey script or complex font."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, char, (x, y), font, scale, (30, 30, 30), thickness, cv2.LINE_AA)


def create_single_sample(char, filename):
    canvas = np.ones((160, 160, 3), dtype=np.uint8) * 255
    # Slight background texture/noise
    noise = np.random.randint(-4, 4, canvas.shape, dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    (tw, th), _ = cv2.getTextSize(char, cv2.FONT_HERSHEY_SIMPLEX, 2.8, 5)
    cx = (160 - tw) // 2
    cy = (160 + th) // 2
    cv2.putText(canvas, char, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2.8, (25, 25, 25), 6, cv2.LINE_AA)
    cv2.imwrite(str(OUTPUT_DIR / filename), canvas)


def create_multi_sample(text, filename):
    canvas = np.ones((160, 100 * len(text) + 60, 3), dtype=np.uint8) * 255
    noise = np.random.randint(-4, 4, canvas.shape, dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    font_scale = 2.4
    thickness = 5

    x_cursor = 40
    for ch in text:
        (tw, th), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cy = (160 + th) // 2 + np.random.randint(-4, 5)
        cv2.putText(canvas, ch, (x_cursor, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (25, 25, 25), thickness, cv2.LINE_AA)
        x_cursor += tw + 35

    cv2.imwrite(str(OUTPUT_DIR / filename), canvas)


def main():
    print("Generating demo handwriting samples in data/samples/...")
    # Single characters
    create_single_sample("3", "digit_3.png")
    create_single_sample("7", "digit_7.png")
    create_single_sample("8", "digit_8.png")
    create_single_sample("A", "letter_A.png")
    create_single_sample("B", "letter_B.png")

    # Multi-characters
    create_multi_sample("42", "multi_42.png")
    create_multi_sample("2026", "multi_2026.png")
    create_multi_sample("AI", "multi_AI.png")
    create_multi_sample("905", "multi_905.png")
    print("Demo samples successfully generated in data/samples/.")


if __name__ == "__main__":
    main()
