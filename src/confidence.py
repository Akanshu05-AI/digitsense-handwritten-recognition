"""
DigitSense: Confidence and Uncertainty Module
Calculates class probabilities, top-k rankings, prediction margins,
normalized Shannon entropy, and low-confidence threshold warnings.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from config import LOW_CONFIDENCE_THRESHOLD, TOP_K_DEFAULT


def compute_softmax_probabilities(logits: torch.Tensor) -> np.ndarray:
    """
    Transform raw CNN logits to calibrated categorical probabilities using Softmax.
    Args:
        logits: PyTorch tensor of shape (batch_size, num_classes) or (num_classes,)
    Returns:
        np.ndarray of shape (batch_size, num_classes) or (num_classes,) with values in [0.0, 1.0].
    """
    if logits.ndim == 1:
        probs = F.softmax(logits, dim=0).detach().cpu().numpy()
    else:
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    return probs.astype(np.float64)


def calculate_prediction_entropy(probs: np.ndarray) -> float:
    """
    Compute normalized Shannon entropy:
        H(p) = - sum(p_i * log2(p_i)) / log2(K)
    Normalized into range [0.0, 1.0].
    A score of 0.0 indicates peak certainty (one class dominates).
    A score near 1.0 indicates maximum uncertainty (uniform distribution).
    """
    num_classes = len(probs)
    if num_classes <= 1:
        return 0.0

    # Only include non-zero probabilities (lim p->0 p*log(p) = 0)
    valid_probs = probs[probs > 1e-12]
    if len(valid_probs) <= 1:
        return 0.0

    entropy = -np.sum(valid_probs * np.log2(valid_probs))
    max_entropy = np.log2(num_classes)
    norm_entropy = float(entropy / max_entropy)
    return float(np.clip(norm_entropy, 0.0, 1.0))


def analyze_prediction_confidence(
    probabilities: np.ndarray,
    class_names: List[str],
    top_k: int = TOP_K_DEFAULT,
    confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    model_type: str = "CNN"
) -> Dict[str, Any]:
    """
    Analyze prediction confidence, top-k alternatives, margin, entropy, and warnings.
    Args:
        probabilities: 1D array of shape (num_classes,) summing to 1.0.
        class_names: List of class strings.
        top_k: Number of highest scoring candidates to return.
        confidence_threshold: Threshold below which a low-confidence warning is generated.
        model_type: 'CNN' (Softmax probabilities) or 'Classical' (Calibrated probabilities).
    """
    if len(probabilities) != len(class_names):
        raise ValueError(
            f"Probabilities count ({len(probabilities)}) does not match class count ({len(class_names)})."
        )

    # Sort descending
    sorted_indices = np.argsort(probabilities)[::-1]
    top_idx = int(sorted_indices[0])
    top_char = class_names[top_idx]
    top_prob = float(probabilities[top_idx])

    # Top-K candidate breakdown
    k = min(top_k, len(class_names))
    top_k_candidates: List[Dict[str, Any]] = []
    for rank in range(k):
        idx = int(sorted_indices[rank])
        prob = float(probabilities[idx])
        top_k_candidates.append({
            "rank": rank + 1,
            "class_index": idx,
            "character": class_names[idx],
            "class_name": class_names[idx],
            "probability": prob,
            "percentage": f"{prob * 100:.2f}%"
        })

    # Confidence margin (difference between #1 and #2)
    margin = (top_prob - float(probabilities[sorted_indices[1]])) if len(sorted_indices) > 1 else 1.0

    # Uncertainty entropy
    entropy = calculate_prediction_entropy(probabilities)

    # Low-confidence flag
    is_low_confidence = top_prob < confidence_threshold
    warnings = []
    if is_low_confidence:
        warnings.append(
            f"Low confidence ({top_prob * 100:.1f}% < threshold {confidence_threshold * 100:.0f}%). "
            f"Character might be ambiguous or noisy."
        )

    if margin < 0.15 and len(sorted_indices) > 1:
        alt_char = class_names[sorted_indices[1]]
        warnings.append(
            f"High ambiguity: Runner-up '{alt_char}' has close probability "
            f"({probabilities[sorted_indices[1]] * 100:.1f}% vs {top_prob * 100:.1f}%)."
        )

    confidence_type = (
        "Softmax Model Probability" if model_type.upper() == "CNN" else "Platt-Calibrated Probability"
    )

    return {
        "predicted_index": top_idx,
        "predicted_char": top_char,
        "confidence": top_prob,
        "confidence_percentage": f"{top_prob * 100:.2f}%",
        "confidence_type": confidence_type,
        "margin": float(margin),
        "normalized_entropy": float(entropy),
        "is_low_confidence": is_low_confidence,
        "warnings": warnings,
        "top_k": top_k_candidates,
        "full_probabilities": probabilities
    }
