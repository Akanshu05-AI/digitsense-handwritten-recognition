"""
DigitSense: Deep Learning CNN Architecture Module
Implements a deep, regularized Convolutional Neural Network with Batch Normalization
and Dropout for handwritten digit and character recognition.
"""

from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEVICE, MNIST_CLASSES, EMNIST_BALANCED_CLASSES


class DigitSenseCNN(nn.Module):
    """
    Multi-stage Convolutional Neural Network for Handwritten Character Recognition.
    Accepts single-channel 28x28 grayscale images and outputs unnormalized class logits.
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        super(DigitSenseCNN, self).__init__()
        self.num_classes = num_classes

        # Stage 1: Initial feature extraction (28x28 -> 14x14)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.20)

        # Stage 2: Mid-level shape and loop extraction (14x14 -> 7x7)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.25)

        # Stage 3: High-level topological representations (7x7 -> 3x3)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(dropout_rate)

        # Classifier Head: Flatten -> Dense 256 -> Logits
        self.flatten_dim = 128 * 3 * 3
        self.fc1 = nn.Linear(self.flatten_dim, 256, bias=False)
        self.bn_fc = nn.BatchNorm1d(256)
        self.drop_fc = nn.Dropout(0.40)
        self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        # Dense Head
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        logits = self.fc_out(x)
        return logits

    def count_parameters(self) -> int:
        """Total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def save_cnn_checkpoint(
    model: DigitSenseCNN,
    file_path: Union[str, Path],
    dataset_name: str,
    class_names: List[str],
    metrics: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model weights, configuration, class mapping, and training metadata to disk.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "num_classes": model.num_classes,
        "dataset_name": dataset_name,
        "class_names": class_names,
        "metrics": metrics or {},
        "extra_metadata": extra_metadata or {},
        "model_architecture": "DigitSenseCNN (3-stage Conv-BN-Dropout)"
    }
    torch.save(checkpoint, str(path))


def load_cnn_checkpoint(
    file_path: Union[str, Path],
    device: Optional[torch.device] = None
) -> Tuple[DigitSenseCNN, Dict[str, Any]]:
    """
    Load a trained CNN model and its training metadata from a checkpoint file.
    Raises:
        FileNotFoundError: If checkpoint does not exist.
        RuntimeError: If checkpoint format is invalid.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CNN model checkpoint not found at: {path}")

    target_device = device or DEVICE
    checkpoint = torch.load(str(path), map_location=target_device, weights_only=False)

    num_classes = checkpoint.get("num_classes", len(checkpoint.get("class_names", [])))
    model = DigitSenseCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(target_device)
    model.eval()

    metadata = {
        "dataset_name": checkpoint.get("dataset_name", "unknown"),
        "class_names": checkpoint.get("class_names", []),
        "num_classes": num_classes,
        "metrics": checkpoint.get("metrics", {}),
        "extra_metadata": checkpoint.get("extra_metadata", {}),
        "model_architecture": checkpoint.get("model_architecture", "DigitSenseCNN"),
        "parameter_count": model.count_parameters()
    }

    return model, metadata
