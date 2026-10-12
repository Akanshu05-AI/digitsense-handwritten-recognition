"""
DigitSense: Dataset Loading and Management Module
Handles MNIST and EMNIST datasets, reproducible train/val/test splits,
EMNIST image orientation correction, and PyTorch / NumPy data conversion.
"""

from typing import Tuple, List, Dict, Optional, Any
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from PIL import Image

from config import (
    RAW_DATA_DIR,
    MNIST_CLASSES,
    EMNIST_BALANCED_CLASSES,
    EMNIST_LETTERS_CLASSES,
    RANDOM_SEED,
    CNN_BATCH_SIZE
)


def get_emnist_orientation_transform():
    """
    EMNIST raw images in the original NIST format are transposed.
    Using Image.TRANSPOSE correctly rotates and flips them to standard upright orientation.
    """
    return transforms.Lambda(lambda img: img.transpose(Image.TRANSPOSE))


def get_transforms(augment: bool = False, is_emnist: bool = False):
    """
    Compose torchvision transforms for train/test datasets.
    """
    transform_list = []
    if is_emnist:
        transform_list.append(get_emnist_orientation_transform())

    if augment:
        transform_list.extend([
            transforms.RandomRotation(degrees=10, fill=0),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05), fill=0)
        ])

    transform_list.append(transforms.ToTensor())  # scales to [0.0, 1.0]

    return transforms.Compose(transform_list)


def load_mnist_datasets(
    data_dir: Path = RAW_DATA_DIR,
    val_ratio: float = 0.1,
    seed: int = RANDOM_SEED
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load MNIST datasets with train, validation, and test splits.
    Train split has data augmentation; val and test splits do not.
    """
    train_transform = get_transforms(augment=True, is_emnist=False)
    eval_transform = get_transforms(augment=False, is_emnist=False)

    full_train_ds = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=train_transform)
    test_ds = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=eval_transform)

    # Deterministic train / val split
    val_size = int(len(full_train_ds) * val_ratio)
    train_size = len(full_train_ds) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_train_ds, [train_size, val_size], generator=generator)

    return train_subset, val_subset, test_ds


def load_emnist_datasets(
    split: str = "balanced",
    data_dir: Path = RAW_DATA_DIR,
    val_ratio: float = 0.1,
    seed: int = RANDOM_SEED
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load EMNIST datasets (e.g. 'balanced' or 'letters') with train, validation, and test splits.
    Applies orientation correction so characters are upright like MNIST.
    """
    train_transform = get_transforms(augment=True, is_emnist=True)
    eval_transform = get_transforms(augment=False, is_emnist=True)

    full_train_ds = datasets.EMNIST(root=str(data_dir), split=split, train=True, download=True, transform=train_transform)
    test_ds = datasets.EMNIST(root=str(data_dir), split=split, train=False, download=True, transform=eval_transform)

    val_size = int(len(full_train_ds) * val_ratio)
    train_size = len(full_train_ds) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_train_ds, [train_size, val_size], generator=generator)

    return train_subset, val_subset, test_ds


def get_dataloaders(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    batch_size: int = CNN_BATCH_SIZE,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and evaluation.
    num_workers=0 ensures safe execution on Windows without multiprocess spawn issues.
    """
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def extract_numpy_dataset(
    dataset: Dataset,
    max_samples: Optional[int] = None,
    seed: int = RANDOM_SEED
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract images and labels as NumPy arrays for classical machine learning models (HOG + SVM).
    Returns:
        X (np.ndarray): Shape (N, 28, 28), float32 in [0.0, 1.0].
        y (np.ndarray): Shape (N,), int64.
    """
    total_samples = len(dataset)
    indices = np.arange(total_samples)

    if max_samples and max_samples < total_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, size=max_samples, replace=False)

    images = []
    labels = []

    for idx in indices:
        item = dataset[idx]
        img, label = item[0], item[1]

        if isinstance(img, torch.Tensor):
            img_np = img.squeeze().numpy()
        elif isinstance(img, Image.Image):
            img_np = np.array(img).astype(np.float32) / 255.0
        elif isinstance(img, np.ndarray):
            img_np = img.astype(np.float32) / 255.0 if img.max() > 1.0 else img.astype(np.float32)
        else:
            img_np = np.array(img, dtype=np.float32)

        images.append(img_np)
        labels.append(int(label))

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)


def get_class_names(dataset_name: str = "mnist") -> List[str]:
    """
    Retrieve human-readable class names for the specified dataset.
    """
    dataset_name = dataset_name.lower().strip()
    if dataset_name == "mnist":
        return MNIST_CLASSES
    elif dataset_name in ["emnist", "emnist_balanced"]:
        return EMNIST_BALANCED_CLASSES
    elif dataset_name == "emnist_letters":
        return EMNIST_LETTERS_CLASSES
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")


def label_to_char(label_idx: int, dataset_name: str = "mnist") -> str:
    """
    Convert an integer class index into its character symbol.
    """
    classes = get_class_names(dataset_name)
    if 0 <= label_idx < len(classes):
        return classes[label_idx]
    return f"?[{label_idx}]"
