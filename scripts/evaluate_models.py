"""
DigitSense: Model Evaluation Script
Evaluates trained models on held-out test sets, generates confusion matrices,
saves JSON metrics, and outputs comparison summary tables.
Usage:
    python scripts/evaluate_models.py [--dataset mnist|emnist|all]
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import (
    CHECKPOINT_CNN_MNIST,
    CHECKPOINT_CNN_EMNIST,
    CHECKPOINT_CLASSICAL_MNIST,
    CHECKPOINT_CLASSICAL_EMNIST,
    DEVICE
)
from src.evaluate import evaluate_cnn, evaluate_classical


def main():
    parser = argparse.ArgumentParser(description="Evaluate DigitSense trained models.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["mnist", "emnist", "all"],
        help="Dataset test set to evaluate"
    )
    args = parser.parse_args()

    print("\n=======================================================")
    print("DigitSense Model Evaluation Pipeline")
    print("=======================================================\n")

    summary_rows = []

    # MNIST Evaluation
    if args.dataset in ["mnist", "all"]:
        if CHECKPOINT_CLASSICAL_MNIST.exists():
            print("Evaluating Classical Baseline on MNIST test set...")
            res_c = evaluate_classical(CHECKPOINT_CLASSICAL_MNIST, dataset_name="mnist")
            summary_rows.append(res_c)
        else:
            print(f"Skipping Classical MNIST: Checkpoint not found at {CHECKPOINT_CLASSICAL_MNIST}")

        if CHECKPOINT_CNN_MNIST.exists():
            print("Evaluating CNN on MNIST test set...")
            res_cnn = evaluate_cnn(CHECKPOINT_CNN_MNIST, dataset_name="mnist", device=DEVICE)
            summary_rows.append(res_cnn)
        else:
            print(f"Skipping CNN MNIST: Checkpoint not found at {CHECKPOINT_CNN_MNIST}")

    # EMNIST Evaluation
    if args.dataset in ["emnist", "all"]:
        if CHECKPOINT_CLASSICAL_EMNIST.exists():
            print("Evaluating Classical Baseline on EMNIST Balanced test set...")
            res_c = evaluate_classical(CHECKPOINT_CLASSICAL_EMNIST, dataset_name="emnist_balanced")
            summary_rows.append(res_c)
        else:
            print(f"Skipping Classical EMNIST: Checkpoint not found at {CHECKPOINT_CLASSICAL_EMNIST}")

        if CHECKPOINT_CNN_EMNIST.exists():
            print("Evaluating CNN on EMNIST Balanced test set...")
            res_cnn = evaluate_cnn(CHECKPOINT_CNN_EMNIST, dataset_name="emnist_balanced", device=DEVICE)
            summary_rows.append(res_cnn)
        else:
            print(f"Skipping CNN EMNIST: Checkpoint not found at {CHECKPOINT_CNN_EMNIST}")

    if summary_rows:
        print("\n" + "=" * 95)
        print(f"{'Model':<35} | {'Dataset':<10} | {'Accuracy':<10} | {'Macro F1':<10} | {'Latency':<12} | {'Size'}")
        print("-" * 95)
        for r in summary_rows:
            print(
                f"{r['model_name']:<35} | "
                f"{r['dataset_name']:<10} | "
                f"{r['accuracy']*100:>8.2f}% | "
                f"{r['f1_macro']*100:>8.2f}% | "
                f"{r['avg_inference_latency_ms']:>8.2f} ms | "
                f"{r['model_size_mb']:>5.2f} MB"
            )
        print("=" * 95 + "\n")
        print("Evaluation reports and confusion matrices saved to reports/results/\n")
    else:
        print("\nNo trained checkpoints were found to evaluate.")
        print("Run `python scripts/train_models.py` first to train models.\n")


if __name__ == "__main__":
    main()
