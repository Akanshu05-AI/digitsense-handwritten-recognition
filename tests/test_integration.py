"""
Integration Tests for DigitSense
Tests end-to-end execution of the real inference pipeline using trained models.
"""

from pathlib import Path
import pytest
import numpy as np

from config import CHECKPOINT_CNN_MNIST, CHECKPOINT_CLASSICAL_MNIST
from src.inference import DigitSenseInferenceEngine


@pytest.mark.skipif(not CHECKPOINT_CNN_MNIST.exists(), reason="CNN MNIST model must be trained first.")
def test_end_to_end_single_digit_cnn():
    engine = DigitSenseInferenceEngine()
    sample_path = Path("data/samples/digit_7.png")
    assert sample_path.exists(), "Sample test image must exist."

    result = engine.recognize(
        image_input=str(sample_path),
        mode="single",
        model_type="cnn",
        dataset_name="mnist"
    )

    assert result.predicted_text == "7", f"Expected '7', got '{result.predicted_text}'"
    assert len(result.character_predictions) == 1
    assert result.character_predictions[0].confidence > 0.80
    assert len(result.character_predictions[0].top_k) == 5
    assert result.latency_ms > 0.0


@pytest.mark.skipif(not CHECKPOINT_CLASSICAL_MNIST.exists(), reason="Classical MNIST model must be trained first.")
def test_end_to_end_single_digit_classical():
    engine = DigitSenseInferenceEngine()
    sample_path = Path("data/samples/digit_7.png")
    assert sample_path.exists()

    result = engine.recognize(
        image_input=str(sample_path),
        mode="single",
        model_type="classical",
        dataset_name="mnist"
    )

    assert result.predicted_text == "7"
    assert len(result.character_predictions) == 1
    assert result.character_predictions[0].confidence > 0.50


@pytest.mark.skipif(not CHECKPOINT_CNN_MNIST.exists(), reason="CNN MNIST model must be trained first.")
def test_end_to_end_multi_digit_segmentation():
    engine = DigitSenseInferenceEngine()
    sample_path = Path("data/samples/multi_42.png")
    assert sample_path.exists()

    result = engine.recognize(
        image_input=str(sample_path),
        mode="auto",
        model_type="cnn",
        dataset_name="mnist"
    )

    assert len(result.character_predictions) == 2, f"Expected 2 characters segmented, got {len(result.character_predictions)}"
    assert result.predicted_text == "42", f"Expected '42', got '{result.predicted_text}'"
    assert result.is_multi_character is True


@pytest.mark.skipif(not CHECKPOINT_CNN_MNIST.exists(), reason="CNN EMNIST model test")
def test_end_to_end_letter_a_recognition():
    from config import CHECKPOINT_CNN_EMNIST
    if not CHECKPOINT_CNN_EMNIST.exists():
        pytest.skip("CNN EMNIST checkpoint not yet saved.")

    engine = DigitSenseInferenceEngine()
    sample_path = Path("data/samples/letter_A.png")
    assert sample_path.exists()

    result = engine.recognize(
        image_input=str(sample_path),
        mode="single",
        model_type="cnn",
        dataset_name="emnist_balanced"
    )

    assert result.predicted_text == "A", f"Expected 'A', got '{result.predicted_text}'"
    assert len(result.character_predictions) == 1
    assert result.character_predictions[0].confidence > 0.50


def test_end_to_end_onnx_recognition():
    engine = DigitSenseInferenceEngine()
    sample_path = Path("data/samples/digit_7.png")
    assert sample_path.exists()

    result = engine.recognize(
        image_input=str(sample_path),
        mode="single",
        model_type="onnx",
        dataset_name="mnist"
    )

    assert result.predicted_text == "7"
    assert result.model_type == "ONNX"
    assert len(result.character_predictions) == 1
    assert result.character_predictions[0].confidence > 0.80


def test_end_to_end_gradcam_generation():
    engine = DigitSenseInferenceEngine()
    sample_path = Path("data/samples/digit_7.png")
    assert sample_path.exists()

    result = engine.recognize(
        image_input=str(sample_path),
        mode="single",
        model_type="cnn",
        dataset_name="mnist",
        generate_cam=True
    )

    assert len(result.character_predictions) == 1
    char_pred = result.character_predictions[0]
    assert char_pred.gradcam_heatmap is not None
    assert char_pred.gradcam_heatmap.shape == (28, 28)
    assert char_pred.gradcam_overlay is not None
    assert char_pred.gradcam_overlay.shape == (28, 28, 3)
    assert char_pred.gradcam_analysis is not None


def test_end_to_end_sauvola_recognition():
    engine = DigitSenseInferenceEngine()
    sample_path = Path("data/samples/digit_7.png")
    assert sample_path.exists()

    result = engine.recognize(
        image_input=str(sample_path),
        mode="single",
        model_type="cnn",
        dataset_name="mnist",
        threshold_method="sauvola",
        sauvola_window=25,
        sauvola_k=0.2
    )

    assert result.predicted_text == "7"
    assert len(result.character_predictions) == 1

