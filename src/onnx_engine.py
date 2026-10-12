"""
DigitSense: ONNX Runtime High-Performance Inference Engine
Provides accelerated CPU/GPU inference for DigitSense CNN models using ONNX Runtime.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from config import (
    CHECKPOINTS_DIR,
    MNIST_CLASSES,
    EMNIST_BALANCED_CLASSES,
    CHECKPOINT_CNN_MNIST,
    CHECKPOINT_CNN_EMNIST
)
from src.confidence import compute_softmax_probabilities, analyze_prediction_confidence


class ONNXDigitSenseEngine:
    """
    Hardware-accelerated inference engine powered by ONNX Runtime.
    Provides sub-millisecond character classification with reduced CPU overhead.
    """

    def __init__(self, checkpoints_dir: Optional[Path] = None):
        self.checkpoints_dir = checkpoints_dir or CHECKPOINTS_DIR
        self.sessions: Dict[str, ort.InferenceSession] = {}
        self.class_mappings = {
            "mnist": MNIST_CLASSES,
            "emnist": EMNIST_BALANCED_CLASSES,
            "emnist_balanced": EMNIST_BALANCED_CLASSES
        }

    def _get_session(self, dataset_name: str) -> ort.InferenceSession:
        ds_key = dataset_name.lower()
        if ds_key in ["emnist", "emnist_balanced"]:
            key = "emnist_balanced"
            onnx_path = self.checkpoints_dir / "cnn_emnist_balanced.onnx"
        else:
            key = "mnist"
            onnx_path = self.checkpoints_dir / "cnn_mnist.onnx"

        if key not in self.sessions:
            if not onnx_path.exists():
                raise FileNotFoundError(
                    f"ONNX model file not found at: {onnx_path}. "
                    "Run 'python scripts/export_onnx.py' to generate ONNX models."
                )

            # Session options optimized for low-latency batch-1 to batch-32 CPU inference
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 2

            # Use CPUExecutionProvider
            providers = ["CPUExecutionProvider"]
            self.sessions[key] = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)

        return self.sessions[key]

    def predict_crops(
        self,
        crops_28x28: List[np.ndarray],
        dataset_name: str = "mnist",
        top_k: int = 5,
        confidence_threshold: float = 0.60
    ) -> List[Dict[str, Any]]:
        """
        Execute ONNX accelerated inference over a list of 28x28 character crops.
        """
        if not crops_28x28:
            return []

        session = self._get_session(dataset_name)
        class_names = self.class_mappings.get(dataset_name.lower(), MNIST_CLASSES)

        # Build batched float32 array: (N, 1, 28, 28) normalized to [0.0, 1.0]
        batch = np.stack([
            c.astype(np.float32) / 255.0 if c.max() > 1.0 else c.astype(np.float32)
            for c in crops_28x28
        ], axis=0)
        batch = np.expand_dims(batch, axis=1)  # (N, 1, 28, 28)

        # Run ONNX inference
        onnx_outputs = session.run(["logits"], {"input": batch})
        logits = onnx_outputs[0]  # (N, num_classes)

        # Softmax probabilities using NumPy
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        all_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        results = []
        for i in range(len(crops_28x28)):
            probs = all_probs[i]
            analysis = analyze_prediction_confidence(
                probs,
                class_names=class_names,
                top_k=top_k,
                confidence_threshold=confidence_threshold,
                model_type="ONNX Runtime"
            )
            results.append(analysis)

        return results


def benchmark_pytorch_vs_onnx(
    dataset_name: str = "mnist",
    num_samples: int = 100,
    batch_size: int = 1
) -> Dict[str, Any]:
    """
    Empirical benchmark comparing inference latency between PyTorch and ONNX Runtime.
    """
    from src.cnn_model import load_cnn_checkpoint

    pt_ckpt = CHECKPOINT_CNN_MNIST if dataset_name == "mnist" else CHECKPOINT_CNN_EMNIST
    onnx_engine = ONNXDigitSenseEngine()

    pt_model, _ = load_cnn_checkpoint(pt_ckpt, device=torch.device("cpu"))
    pt_model.eval()

    # Synthetic test input
    synthetic_batch_np = np.random.rand(batch_size, 1, 28, 28).astype(np.float32)
    synthetic_batch_torch = torch.from_numpy(synthetic_batch_np)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = pt_model(synthetic_batch_torch)
        _ = onnx_engine._get_session(dataset_name).run(["logits"], {"input": synthetic_batch_np})

    # Benchmark PyTorch
    pt_times = []
    with torch.no_grad():
        for _ in range(num_samples):
            t0 = time.perf_counter()
            _ = pt_model(synthetic_batch_torch)
            pt_times.append((time.perf_counter() - t0) * 1000.0)

    # Benchmark ONNX Runtime
    onnx_session = onnx_engine._get_session(dataset_name)
    onnx_times = []
    for _ in range(num_samples):
        t0 = time.perf_counter()
        _ = onnx_session.run(["logits"], {"input": synthetic_batch_np})
        onnx_times.append((time.perf_counter() - t0) * 1000.0)

    avg_pt = float(np.mean(pt_times))
    avg_onnx = float(np.mean(onnx_times))
    speedup = avg_pt / max(1e-6, avg_onnx)

    return {
        "dataset": dataset_name,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "pytorch_latency_ms": round(avg_pt, 3),
        "onnx_latency_ms": round(avg_onnx, 3),
        "speedup_factor": round(speedup, 2),
        "reduction_percentage": round((1.0 - avg_onnx / avg_pt) * 100.0, 1) if avg_pt > avg_onnx else 0.0
    }
