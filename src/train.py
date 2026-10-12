"""
DigitSense: Model Training Module
Provides reproducible training pipelines for both PyTorch CNN and
Classical HOG + LinearSVC models with validation tracking and early stopping.
"""

from typing import Dict, Any, Optional
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    DEVICE,
    RANDOM_SEED,
    CNN_BATCH_SIZE,
    CNN_LEARNING_RATE,
    CNN_EPOCHS,
    CNN_WEIGHT_DECAY,
    CNN_PATIENCE,
    CHECKPOINT_CNN_MNIST,
    CHECKPOINT_CNN_EMNIST,
    CHECKPOINT_CLASSICAL_MNIST,
    CHECKPOINT_CLASSICAL_EMNIST
)
from src.datasets import (
    load_mnist_datasets,
    load_emnist_datasets,
    get_dataloaders,
    extract_numpy_dataset,
    get_class_names
)
from src.cnn_model import DigitSenseCNN, save_cnn_checkpoint
from src.classical_model import ClassicalHOGClassifier


def set_seed(seed: int = RANDOM_SEED):
    """Ensure reproducibility across PyTorch and NumPy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train_cnn(
    dataset_name: str = "mnist",
    epochs: int = CNN_EPOCHS,
    batch_size: int = CNN_BATCH_SIZE,
    learning_rate: float = CNN_LEARNING_RATE,
    weight_decay: float = CNN_WEIGHT_DECAY,
    patience: int = CNN_PATIENCE,
    checkpoint_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train DigitSenseCNN on MNIST or EMNIST Balanced dataset.
    Includes validation monitoring and early stopping.
    """
    set_seed(RANDOM_SEED)
    dev = device or DEVICE
    start_time = time.perf_counter()

    if dataset_name.lower() == "mnist":
        train_ds, val_ds, test_ds = load_mnist_datasets()
        class_names = get_class_names("mnist")
        save_path = checkpoint_path or CHECKPOINT_CNN_MNIST
    elif dataset_name.lower() in ["emnist", "emnist_balanced"]:
        train_ds, val_ds, test_ds = load_emnist_datasets(split="balanced")
        class_names = get_class_names("emnist_balanced")
        save_path = checkpoint_path or CHECKPOINT_CNN_EMNIST
    else:
        raise ValueError(f"Unsupported dataset for CNN training: {dataset_name}")

    num_classes = len(class_names)
    train_loader, val_loader, _ = get_dataloaders(train_ds, val_ds, test_ds, batch_size=batch_size)

    if verbose:
        print(f"\n==========================================")
        print(f"Training CNN on {dataset_name.upper()} ({num_classes} classes)")
        print(f"Device: {dev} | Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
        print(f"Batch Size: {batch_size} | Initial LR: {learning_rate}")
        print(f"==========================================\n")

    model = DigitSenseCNN(num_classes=num_classes).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, verbose=verbose)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        # Training Phase
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(dev), labels.to(dev)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        epoch_train_loss = running_train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(dev), labels.to(dev)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = running_val_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_time = time.perf_counter() - epoch_start

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(epoch_val_loss)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["lr"].append(current_lr)

        if verbose:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
                f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}% | "
                f"Time: {epoch_time:.1f}s"
            )

        # Early stopping & Best model checkpointing
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            epochs_no_improve = 0

            # Save best checkpoint
            save_cnn_checkpoint(
                model=model,
                file_path=save_path,
                dataset_name=dataset_name,
                class_names=class_names,
                metrics={
                    "best_epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                    "history": history
                },
                extra_metadata={
                    "total_epochs": epoch,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "device": str(dev)
                }
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"\n[Early Stopping triggered after {epoch} epochs. Best Val Acc: {best_val_acc*100:.2f}%]")
                break

    total_training_time = time.perf_counter() - start_time
    if verbose:
        print(f"CNN Training Complete in {total_training_time:.1f} seconds. Checkpoint saved to: {save_path}")

    return {
        "dataset_name": dataset_name,
        "checkpoint_path": str(save_path),
        "total_training_time_sec": total_training_time,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "history": history
    }


def train_classical(
    dataset_name: str = "mnist",
    max_train_samples: int = 15000,
    checkpoint_path: Optional[Path] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train Classical HOG + Calibrated LinearSVC Baseline.
    """
    set_seed(RANDOM_SEED)
    start_time = time.perf_counter()

    if dataset_name.lower() == "mnist":
        train_ds, val_ds, test_ds = load_mnist_datasets()
        class_names = get_class_names("mnist")
        save_path = checkpoint_path or CHECKPOINT_CLASSICAL_MNIST
    elif dataset_name.lower() in ["emnist", "emnist_balanced"]:
        train_ds, val_ds, test_ds = load_emnist_datasets(split="balanced")
        class_names = get_class_names("emnist_balanced")
        save_path = checkpoint_path or CHECKPOINT_CLASSICAL_EMNIST
    else:
        raise ValueError(f"Unsupported dataset for Classical training: {dataset_name}")

    if verbose:
        print(f"\n==========================================")
        print(f"Training Classical Baseline (HOG + LinearSVC) on {dataset_name.upper()}")
        print(f"Extracting up to {max_train_samples} samples from training set...")
        print(f"==========================================\n")

    X_train, y_train = extract_numpy_dataset(train_ds, max_samples=max_train_samples, seed=RANDOM_SEED)

    clf = ClassicalHOGClassifier(random_state=RANDOM_SEED)
    clf.fit(X_train, y_train, dataset_name=dataset_name)
    clf.save(save_path)

    total_time = time.perf_counter() - start_time
    if verbose:
        print(f"Classical Training Complete in {total_time:.1f}s. Saved to: {save_path}")

    return {
        "dataset_name": dataset_name,
        "checkpoint_path": str(save_path),
        "total_training_time_sec": total_time,
        "metadata": clf.metadata
    }
