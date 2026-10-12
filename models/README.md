# DigitSense Models Directory

This directory stores trained model checkpoints, configurations, and weights.

## Checkpoint Files
- `checkpoints/cnn_mnist.pt`:
  - Architecture: `DigitSenseCNN` (3-stage Conv2D + BatchNorm2D + MaxPool2D + Dropout + Dense Head).
  - Classes: 10 digit classes ('0' through '9').
  - Training Data: MNIST Train split with spatial data augmentation.
- `checkpoints/cnn_emnist_balanced.pt`:
  - Architecture: `DigitSenseCNN` (47 output logits).
  - Classes: 47 classes (0–9 digits, A–Z uppercase, and a, b, d, e, f, g, h, n, q, r, t lowercase).
  - Training Data: EMNIST Balanced Train split.
- `checkpoints/classical_mnist.joblib`:
  - Architecture: Histogram of Oriented Gradients (HOG) + Calibrated Linear Support Vector Classifier (Platt scaling).
  - Classes: 10 digit classes ('0' through '9').
- `checkpoints/classical_emnist.joblib`:
  - Architecture: HOG + Calibrated LinearSVC.
  - Classes: 47 classes.

## Checkpoint Structure
PyTorch checkpoints contain:
- `state_dict`: Model weights and batch norm statistics.
- `num_classes`: Output dimension.
- `class_names`: List of class labels in index order.
- `metrics`: Best validation loss, accuracy, and full training loss history.
- `extra_metadata`: Hyperparameters, batch size, learning rate, and device.
