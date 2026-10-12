"""
DigitSense: Model Evaluation Module
Evaluates trained models on held-out test sets, computing accuracy,
precision, recall, F1, confusion matrices, latency, and per-class reports.
"""

from typing import Dict, Any, Optional, List, Tuple
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from config import (
    DEVICE,
    RESULTS_DIR,
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
from src.cnn_model import load_cnn_checkpoint
from src.classical_model import ClassicalHOGClassifier
from src.confidence import compute_softmax_probabilities
from src.visualization import plot_confusion_matrix, plot_misclassified_gallery


def evaluate_cnn(
    checkpoint_path: Path,
    dataset_name: str = "mnist",
    save_results: bool = True,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a trained CNN model on held-out test set.
    """
    dev = device or DEVICE
    model, metadata = load_cnn_checkpoint(checkpoint_path, device=dev)
    class_names = metadata["class_names"]

    if dataset_name.lower() == "mnist":
        _, _, test_ds = load_mnist_datasets()
    elif dataset_name.lower() in ["emnist", "emnist_balanced"]:
        _, _, test_ds = load_emnist_datasets(split="balanced")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    _, _, test_loader = get_dataloaders(test_ds, test_ds, test_ds, batch_size=128)

    y_true = []
    y_pred = []
    y_probs = []
    misclassified = []

    model.eval()
    inference_start = time.perf_counter()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(dev)
            outputs = model(images)
            probs = compute_softmax_probabilities(outputs)
            preds = np.argmax(probs, axis=1)

            labels_np = labels.numpy()
            y_true.extend(labels_np)
            y_pred.extend(preds)
            y_probs.extend(probs)

            # Collect misclassified instances
            for idx in range(len(labels_np)):
                if labels_np[idx] != preds[idx] and len(misclassified) < 25:
                    img_np = images[idx].squeeze().cpu().numpy()
                    misclassified.append({
                        "image": img_np,
                        "true_index": int(labels_np[idx]),
                        "pred_index": int(preds[idx]),
                        "true_label": class_names[labels_np[idx]],
                        "pred_label": class_names[preds[idx]],
                        "confidence": float(probs[idx][preds[idx]])
                    })

    total_eval_time = time.perf_counter() - inference_start
    avg_latency_ms = (total_eval_time / len(y_true)) * 1000.0

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Compute comprehensive metrics
    acc = float(accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    p_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    r_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    checkpoint_size_mb = Path(checkpoint_path).stat().st_size / (1024 * 1024)

    results = {
        "model_type": "CNN",
        "model_name": metadata.get("model_architecture", "DigitSenseCNN"),
        "dataset_name": dataset_name,
        "checkpoint_path": str(checkpoint_path),
        "test_samples": int(len(y_true)),
        "num_classes": int(len(class_names)),
        "class_names": class_names,
        "accuracy": acc,
        "precision_macro": p_macro,
        "precision_weighted": p_weighted,
        "recall_macro": r_macro,
        "recall_weighted": r_weighted,
        "f1_macro": f1_m,
        "f1_weighted": f1_w,
        "avg_inference_latency_ms": avg_latency_ms,
        "total_eval_time_sec": total_eval_time,
        "model_size_mb": checkpoint_size_mb,
        "parameter_count": metadata.get("parameter_count", 0),
        "confusion_matrix": cm.tolist(),
        "per_class_report": report_dict,
        "misclassified_samples": misclassified
    }

    if verbose:
        print(f"\n==========================================")
        print(f"EVALUATION: CNN on {dataset_name.upper()} (Held-Out Test Set)")
        print(f"Accuracy: {acc * 100:.2f}% | Macro F1: {f1_m * 100:.2f}% | Latency: {avg_latency_ms:.2f} ms/sample")
        print(f"Model Size: {checkpoint_size_mb:.2f} MB | Total Test Samples: {len(y_true)}")
        print(f"==========================================\n")

    if save_results:
        results_file = RESULTS_DIR / f"metrics_cnn_{dataset_name}.json"
        save_dict = {k: v for k, v in results.items() if k != "misclassified_samples"}
        with open(results_file, "w") as f:
            json.dump(save_dict, f, indent=2)

        # Save confusion matrix plot
        cm_fig = plot_confusion_matrix(cm, class_names, title=f"CNN {dataset_name.upper()} Confusion Matrix")
        cm_fig.savefig(RESULTS_DIR / f"cm_cnn_{dataset_name}.png", dpi=200, bbox_inches="tight")

    return results


def evaluate_classical(
    checkpoint_path: Path,
    dataset_name: str = "mnist",
    max_test_samples: int = 5000,
    save_results: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate trained Classical HOG + Calibrated LinearSVC model on held-out test set.
    """
    clf = ClassicalHOGClassifier.load(checkpoint_path)
    class_names = get_class_names(dataset_name)

    if dataset_name.lower() == "mnist":
        _, _, test_ds = load_mnist_datasets()
    elif dataset_name.lower() in ["emnist", "emnist_balanced"]:
        _, _, test_ds = load_emnist_datasets(split="balanced")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    X_test, y_test = extract_numpy_dataset(test_ds, max_samples=max_test_samples)

    start_eval = time.perf_counter()
    y_probs = clf.predict_proba(X_test)
    y_pred = np.argmax(y_probs, axis=1)
    total_eval_time = time.perf_counter() - start_eval
    avg_latency_ms = (total_eval_time / len(X_test)) * 1000.0

    acc = float(accuracy_score(y_test, y_pred))
    p_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    p_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    r_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    r_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_w = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    checkpoint_size_mb = Path(checkpoint_path).stat().st_size / (1024 * 1024)

    # Collect misclassified samples
    misclassified = []
    for idx in range(len(y_test)):
        if y_test[idx] != y_pred[idx] and len(misclassified) < 25:
            misclassified.append({
                "image": X_test[idx],
                "true_index": int(y_test[idx]),
                "pred_index": int(y_pred[idx]),
                "true_label": class_names[y_test[idx]],
                "pred_label": class_names[y_pred[idx]],
                "confidence": float(y_probs[idx][y_pred[idx]])
            })

    results = {
        "model_type": "Classical (HOG + Calibrated LinearSVC)",
        "model_name": "HOG_LinearSVC",
        "dataset_name": dataset_name,
        "checkpoint_path": str(checkpoint_path),
        "test_samples": int(len(y_test)),
        "num_classes": int(len(class_names)),
        "class_names": class_names,
        "accuracy": acc,
        "precision_macro": p_macro,
        "precision_weighted": p_weighted,
        "recall_macro": r_macro,
        "recall_weighted": r_weighted,
        "f1_macro": f1_m,
        "f1_weighted": f1_w,
        "avg_inference_latency_ms": avg_latency_ms,
        "total_eval_time_sec": total_eval_time,
        "model_size_mb": checkpoint_size_mb,
        "confusion_matrix": cm.tolist(),
        "per_class_report": report_dict,
        "misclassified_samples": misclassified
    }

    if verbose:
        print(f"\n==========================================")
        print(f"EVALUATION: Classical Baseline on {dataset_name.upper()} (Held-Out Test Set)")
        print(f"Accuracy: {acc * 100:.2f}% | Macro F1: {f1_m * 100:.2f}% | Latency: {avg_latency_ms:.2f} ms/sample")
        print(f"Model Size: {checkpoint_size_mb:.2f} MB | Samples Evaluated: {len(y_test)}")
        print(f"==========================================\n")

    if save_results:
        results_file = RESULTS_DIR / f"metrics_classical_{dataset_name}.json"
        save_dict = {k: v for k, v in results.items() if k != "misclassified_samples"}
        with open(results_file, "w") as f:
            json.dump(save_dict, f, indent=2)

        cm_fig = plot_confusion_matrix(cm, class_names, title=f"Classical {dataset_name.upper()} Confusion Matrix")
        cm_fig.savefig(RESULTS_DIR / f"cm_classical_{dataset_name}.png", dpi=200, bbox_inches="tight")

    return results
