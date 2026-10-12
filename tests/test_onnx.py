"""
Unit tests for ONNX Runtime inference engine.
"""

import numpy as np
import pytest

from src.onnx_engine import ONNXDigitSenseEngine, benchmark_pytorch_vs_onnx


def test_onnx_inference_mnist():
    engine = ONNXDigitSenseEngine()
    dummy_crop = np.zeros((28, 28), dtype=np.uint8)
    # Draw simple vertical stroke (like digit 1)
    dummy_crop[5:23, 13:15] = 255

    preds = engine.predict_crops([dummy_crop], dataset_name="mnist")
    assert len(preds) == 1
    assert "predicted_char" in preds[0]
    assert "confidence" in preds[0]
    assert 0.0 <= preds[0]["confidence"] <= 1.0


def test_onnx_inference_emnist():
    engine = ONNXDigitSenseEngine()
    dummy_crop = np.zeros((28, 28), dtype=np.uint8)
    dummy_crop[10:18, 10:18] = 255

    preds = engine.predict_crops([dummy_crop], dataset_name="emnist_balanced")
    assert len(preds) == 1
    assert "predicted_char" in preds[0]
    assert "top_k" in preds[0]
    assert len(preds[0]["top_k"]) == 5


def test_onnx_benchmark():
    benchmark_res = benchmark_pytorch_vs_onnx(dataset_name="mnist", num_samples=10, batch_size=1)
    assert "pytorch_latency_ms" in benchmark_res
    assert "onnx_latency_ms" in benchmark_res
    assert "speedup_factor" in benchmark_res
    assert benchmark_res["onnx_latency_ms"] > 0
