"""
Unit Tests for Inference and Confidence Module
Covers CNN model forward pass, Softmax calculation, Top-K ranking,
normalized entropy, and missing model exception handling.
"""

import pytest
import numpy as np
import torch

from src.cnn_model import DigitSenseCNN
from src.confidence import (
    compute_softmax_probabilities,
    calculate_prediction_entropy,
    analyze_prediction_confidence
)
from src.inference import DigitSenseInferenceEngine


def test_cnn_architecture_shapes():
    # 10-class MNIST model
    model_mnist = DigitSenseCNN(num_classes=10)
    dummy_input = torch.randn(4, 1, 28, 28)
    logits = model_mnist(dummy_input)
    assert logits.shape == (4, 10)

    # 47-class EMNIST Balanced model
    model_emnist = DigitSenseCNN(num_classes=47)
    logits_emnist = model_emnist(dummy_input)
    assert logits_emnist.shape == (4, 47)


def test_softmax_probabilities():
    logits = torch.tensor([[2.0, 1.0, 0.1, -1.5]])
    probs = compute_softmax_probabilities(logits)
    assert probs.shape == (1, 4)
    # Sum of probabilities across classes must be 1.0
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[0, 0] > probs[0, 1] > probs[0, 2] > probs[0, 3]


def test_prediction_entropy():
    # Highly confident (one class has probability 1.0) -> entropy should be 0.0
    certain_probs = np.array([1.0, 0.0, 0.0, 0.0])
    assert calculate_prediction_entropy(certain_probs) == 0.0

    # Completely uniform distribution -> entropy should be 1.0
    uniform_probs = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.isclose(calculate_prediction_entropy(uniform_probs), 1.0)


def test_analyze_prediction_confidence():
    probs = np.array([0.85, 0.10, 0.03, 0.02])
    classes = ["0", "1", "2", "3"]
    res = analyze_prediction_confidence(probs, class_names=classes, top_k=3, confidence_threshold=0.60)

    assert res["predicted_char"] == "0"
    assert res["predicted_index"] == 0
    assert np.isclose(res["confidence"], 0.85)
    assert res["is_low_confidence"] is False
    assert len(res["top_k"]) == 3
    assert res["top_k"][0]["character"] == "0"
    assert res["top_k"][1]["character"] == "1"


def test_low_confidence_warning():
    probs = np.array([0.35, 0.33, 0.20, 0.12])
    classes = ["A", "B", "C", "D"]
    res = analyze_prediction_confidence(probs, class_names=classes, top_k=2, confidence_threshold=0.60)

    assert res["is_low_confidence"] is True
    assert len(res["warnings"]) > 0


def test_inference_engine_missing_checkpoint():
    engine = DigitSenseInferenceEngine()
    with pytest.raises(FileNotFoundError):
        # Requesting a non-existent checkpoint path
        engine.get_model("cnn", dataset_name="non_existent_dataset_abc")
