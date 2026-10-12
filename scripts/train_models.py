"""
DigitSense: Model Training Script
Trains Convolutional Neural Networks and Classical ML Baselines on MNIST and EMNIST.
Usage:
    python scripts/train_models.py [--model cnn|classical|all] [--dataset mnist|emnist|all] [--epochs N]
"""

import argparse
import sys
from pathlib import Path
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import (
    CNN_EPOCHS,
    CNN_BATCH_SIZE,
    CNN_LEARNING_RATE,
    DEVICE,
    DEVICE_NAME
)
from src.train import train_cnn, train_classical


def main():
    parser = argparse.ArgumentParser(description="Train DigitSense recognition models.")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["cnn", "classical", "all"],
        help="Model architecture to train"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["mnist", "emnist", "all"],
        help="Dataset to train on"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=CNN_EPOCHS,
        help="Epoch count for CNN training"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=CNN_BATCH_SIZE,
        help="Batch size for CNN training"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=15000,
        help="Max training samples for Classical baseline"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=str(DEVICE),
        help="Execution device ('cuda' or 'cpu')"
    )
    args = parser.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    print("\n=======================================================")
    print("DigitSense Model Training Pipeline")
    print(f"Target Device: {dev} ({DEVICE_NAME})")
    print(f"Models: {args.model.upper()} | Datasets: {args.dataset.upper()}")
    print("=======================================================\n")

    datasets_to_train = ["mnist", "emnist"] if args.dataset == "all" else [args.dataset]
    models_to_train = ["classical", "cnn"] if args.model == "all" else [args.model]

    for ds in datasets_to_train:
        for m in models_to_train:
            if m == "classical":
                sample_count = args.samples if ds == "mnist" else min(args.samples + 5000, 20000)
                print(f"\n--- Training Classical Baseline on {ds.upper()} ({sample_count} samples) ---")
                train_classical(dataset_name=ds, max_train_samples=sample_count, verbose=True)

            elif m == "cnn":
                # For quick student training, use 10 epochs for MNIST, 12 for EMNIST
                ep = args.epochs if ds == "mnist" else min(args.epochs, 12)
                print(f"\n--- Training CNN on {ds.upper()} ({ep} epochs) ---")
                train_cnn(
                    dataset_name=ds,
                    epochs=ep,
                    batch_size=args.batch_size,
                    device=dev,
                    verbose=True
                )

    print("\n✓ Model training pipeline finished successfully!")
    print("Trained checkpoints are stored in models/checkpoints/\n")


if __name__ == "__main__":
    main()
