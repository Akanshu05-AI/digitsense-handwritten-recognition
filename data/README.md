# DigitSense Datasets Directory

This directory stores raw and processed dataset files for handwritten digit and character recognition.

## Structure
- `raw/`: Downloaded original benchmark datasets:
  - `MNIST/`: Modified National Institute of Standards and Technology dataset containing 70,000 28x28 grayscale images of handwritten digits (0–9).
    - 60,000 training samples
    - 10,000 held-out test samples
  - `EMNIST/`: Extended MNIST dataset (Balanced Split).
    - Contains 131,600 total samples across 47 balanced classes (0-9, A-Z, plus 11 lowercase letters with distinctive shapes: a, b, d, e, f, g, h, n, q, r, t).
    - 112,800 training samples (2,400 per class)
    - 18,800 test samples (400 per class)
- `processed/`: Intermediate processed arrays and cached subsets.
- `samples/`: Test images for demo and interactive UI experimentation.

## Orientation Correction
EMNIST raw binary images from NIST SD19 are stored transposed relative to MNIST. DigitSense automatically applies an orientation transform (`Image.TRANSPOSE`) during loading to ensure all characters are upright.

## License and Attribution
- MNIST is provided by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.
- EMNIST is provided by Gregory Cohen, Saeed Afshar, Jonathan Tapson, and André van Schaik (Western Sydney University).
