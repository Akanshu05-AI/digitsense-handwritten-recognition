"""
DigitSense: ONNX Export Script
Converts trained PyTorch DigitSenseCNN models to Open Neural Network Exchange (ONNX) format
for high-performance cross-platform hardware-accelerated inference.
"""

import sys
from pathlib import Path
import argparse
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import onnx
import onnxruntime as ort

from config import (
    CHECKPOINT_CNN_MNIST,
    CHECKPOINT_CNN_EMNIST,
    CHECKPOINTS_DIR
)
from src.cnn_model import load_cnn_checkpoint


def export_model_to_onnx(
    checkpoint_path: Path,
    output_onnx_path: Path,
    opset_version: int = 14
) -> Path:
    """
    Export a PyTorch DigitSenseCNN checkpoint to ONNX format with dynamic batch axis.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Source PyTorch checkpoint not found at: {checkpoint_path}")

    print(f"Loading PyTorch checkpoint from: {checkpoint_path}...")
    model, metadata = load_cnn_checkpoint(checkpoint_path, device=torch.device("cpu"))
    model.eval()

    dummy_input = torch.randn(1, 1, 28, 28, dtype=torch.float32)

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to ONNX: {output_onnx_path} (opset {opset_version})...")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        dynamo=False
    )

    # Validate ONNX model integrity
    onnx_model = onnx.load(str(output_onnx_path))
    onnx.checker.check_model(onnx_model)
    print("  ONNX model graph verified successfully by onnx.checker.")

    # Validate inference equivalence between PyTorch and ONNX Runtime
    session = ort.InferenceSession(str(output_onnx_path), providers=["CPUExecutionProvider"])
    test_batch = np.random.randn(4, 1, 28, 28).astype(np.float32)

    with torch.no_grad():
        torch_logits = model(torch.from_numpy(test_batch)).numpy()

    onnx_logits = session.run(["logits"], {"input": test_batch})[0]

    max_abs_diff = np.max(np.abs(torch_logits - onnx_logits))
    print(f"  PyTorch vs ONNX Max Absolute Output Difference: {max_abs_diff:.6e}")
    assert max_abs_diff < 1e-4, f"Output disparity between PyTorch and ONNX too high ({max_abs_diff})"

    print(f"Export and numerical validation complete for {output_onnx_path.name}\n")
    return output_onnx_path


def main():
    parser = argparse.ArgumentParser(description="Export DigitSense PyTorch CNN models to ONNX format.")
    parser.add_argument("--dataset", choices=["all", "mnist", "emnist"], default="all", help="Target model dataset.")
    args = parser.parse_args()

    targets = []
    if args.dataset in ["all", "mnist"]:
        targets.append((CHECKPOINT_CNN_MNIST, CHECKPOINTS_DIR / "cnn_mnist.onnx"))
    if args.dataset in ["all", "emnist"]:
        targets.append((CHECKPOINT_CNN_EMNIST, CHECKPOINTS_DIR / "cnn_emnist_balanced.onnx"))

    for pt_path, onnx_path in targets:
        export_model_to_onnx(pt_path, onnx_path)


if __name__ == "__main__":
    main()
