"""
DigitSense: Dataset Download Script
Downloads and verifies MNIST and EMNIST Balanced datasets.
Usage:
    python scripts/download_data.py [--dataset mnist|emnist|all]
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.datasets import load_mnist_datasets, load_emnist_datasets


def main():
    parser = argparse.ArgumentParser(description="Download and verify MNIST and EMNIST datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["mnist", "emnist", "all"],
        help="Target dataset to download ('mnist', 'emnist', or 'all')"
    )
    args = parser.parse_args()

    print("\n=======================================================")
    print("DigitSense Dataset Downloader")
    print("=======================================================\n")

    if args.dataset in ["mnist", "all"]:
        print("Checking/Downloading MNIST dataset (10 digit classes)...")
        train_ds, val_ds, test_ds = load_mnist_datasets()
        print(f"✓ MNIST successfully verified!")
        print(f"  - Training samples:   {len(train_ds):,}")
        print(f"  - Validation samples: {len(val_ds):,}")
        print(f"  - Test samples:       {len(test_ds):,}")
        print(f"  - Total samples:      {len(train_ds) + len(val_ds) + len(test_ds):,}\n")

    if args.dataset in ["emnist", "all"]:
        print("Checking/Downloading EMNIST Balanced dataset (47 classes)...")
        train_ds, val_ds, test_ds = load_emnist_datasets(split="balanced")
        print(f"✓ EMNIST Balanced successfully verified!")
        print(f"  - Training samples:   {len(train_ds):,}")
        print(f"  - Validation samples: {len(val_ds):,}")
        print(f"  - Test samples:       {len(test_ds):,}")
        print(f"  - Total samples:      {len(train_ds) + len(val_ds) + len(test_ds):,}\n")

    print("All requested datasets are ready in data/raw/\n")


if __name__ == "__main__":
    main()
