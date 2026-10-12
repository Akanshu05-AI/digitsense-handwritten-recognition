"""
DigitSense: Classical Machine Learning Baseline Module
Implements HOG (Histogram of Oriented Gradients) feature extraction
coupled with a Scikit-Learn Calibrated Linear Classifier.
"""

from typing import Tuple, List, Optional, Dict, Any, Union
import time
from pathlib import Path
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import joblib

from config import (
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HOG_CELLS_PER_BLOCK,
    RANDOM_SEED
)


def extract_hog_feature(
    image_28x28: np.ndarray,
    orientations: int = HOG_ORIENTATIONS,
    pixels_per_cell: Tuple[int, int] = HOG_PIXELS_PER_CELL,
    cells_per_block: Tuple[int, int] = HOG_CELLS_PER_BLOCK
) -> np.ndarray:
    """
    Extract HOG descriptor vector from a single 28x28 grayscale image.
    Image should be float32 in range [0.0, 1.0].
    """
    if image_28x28.shape != (28, 28):
        raise ValueError(f"Expected 28x28 image, got {image_28x28.shape}")

    feat = hog(
        image_28x28,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        visualize=False
    )
    return feat.astype(np.float32)


def extract_batch_hog(
    images: np.ndarray,
    orientations: int = HOG_ORIENTATIONS,
    pixels_per_cell: Tuple[int, int] = HOG_PIXELS_PER_CELL,
    cells_per_block: Tuple[int, int] = HOG_CELLS_PER_BLOCK
) -> np.ndarray:
    """
    Extract HOG features for an array of images (N, 28, 28).
    """
    features = [
        extract_hog_feature(img, orientations, pixels_per_cell, cells_per_block)
        for img in images
    ]
    return np.array(features, dtype=np.float32)


class ClassicalHOGClassifier:
    """
    Classical Machine Learning Baseline combining HOG feature extraction
    with Calibrated Linear Support Vector Classification.
    """

    def __init__(
        self,
        orientations: int = HOG_ORIENTATIONS,
        pixels_per_cell: Tuple[int, int] = HOG_PIXELS_PER_CELL,
        cells_per_block: Tuple[int, int] = HOG_CELLS_PER_BLOCK,
        random_state: int = RANDOM_SEED
    ):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.random_state = random_state

        # LinearSVC provides fast, high-margin linear boundaries for high-dimensional HOG features.
        # CalibratedClassifierCV applies Platt scaling (sigmoid calibration)
        # to produce well-calibrated posterior probabilities.
        base_svc = LinearSVC(
            C=1.0,
            dual="auto",
            random_state=random_state,
            max_iter=2500
        )
        self.model = CalibratedClassifierCV(estimator=base_svc, cv=3)
        self.classes_: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        self.metadata: Dict[str, Any] = {}

    def fit(self, X_images: np.ndarray, y_labels: np.ndarray, dataset_name: str = "mnist") -> "ClassicalHOGClassifier":
        """
        Extract HOG features and train the calibrated classifier.
        Args:
            X_images: np.ndarray of shape (N, 28, 28)
            y_labels: np.ndarray of shape (N,)
        """
        start_time = time.perf_counter()

        feat_start = time.perf_counter()
        X_feats = extract_batch_hog(
            X_images,
            self.orientations,
            self.pixels_per_cell,
            self.cells_per_block
        )
        feature_extraction_time = time.perf_counter() - feat_start

        train_start = time.perf_counter()
        self.model.fit(X_feats, y_labels)
        training_time = time.perf_counter() - train_start

        total_time = time.perf_counter() - start_time
        self.classes_ = self.model.classes_
        self.is_fitted = True

        self.metadata = {
            "model_type": "HOG + Calibrated LinearSVC",
            "dataset_name": dataset_name,
            "num_samples": len(X_images),
            "num_classes": len(self.classes_),
            "feature_dim": X_feats.shape[1],
            "feature_extraction_time_sec": feature_extraction_time,
            "training_time_sec": training_time,
            "total_time_sec": total_time,
            "hog_config": {
                "orientations": self.orientations,
                "pixels_per_cell": self.pixels_per_cell,
                "cells_per_block": self.cells_per_block
            }
        }
        return self

    def predict(self, X_images: np.ndarray) -> np.ndarray:
        """
        Predict class labels for given 28x28 images.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Train or load a checkpoint first.")

        if X_images.ndim == 2:
            X_images = np.expand_dims(X_images, axis=0)

        X_feats = extract_batch_hog(
            X_images,
            self.orientations,
            self.pixels_per_cell,
            self.cells_per_block
        )
        return self.model.predict(X_feats)

    def predict_proba(self, X_images: np.ndarray) -> np.ndarray:
        """
        Return calibrated class probability estimates for given images.
        Shape: (N, num_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Train or load a checkpoint first.")

        if X_images.ndim == 2:
            X_images = np.expand_dims(X_images, axis=0)

        X_feats = extract_batch_hog(
            X_images,
            self.orientations,
            self.pixels_per_cell,
            self.cells_per_block
        )
        return self.model.predict_proba(X_feats)

    def save(self, file_path: Union[str, Path]) -> None:
        """
        Serialize model and metadata to disk.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": self.model,
            "classes_": self.classes_,
            "is_fitted": self.is_fitted,
            "metadata": self.metadata,
            "orientations": self.orientations,
            "pixels_per_cell": self.pixels_per_cell,
            "cells_per_block": self.cells_per_block,
            "random_state": self.random_state
        }
        joblib.dump(bundle, str(path), compress=3)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "ClassicalHOGClassifier":
        """
        Load trained model bundle from disk.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Classical model checkpoint not found: {path}")

        bundle = joblib.load(str(path))
        instance = cls(
            orientations=bundle.get("orientations", HOG_ORIENTATIONS),
            pixels_per_cell=bundle.get("pixels_per_cell", HOG_PIXELS_PER_CELL),
            cells_per_block=bundle.get("cells_per_block", HOG_CELLS_PER_BLOCK),
            random_state=bundle.get("random_state", RANDOM_SEED)
        )
        instance.model = bundle["model"]
        instance.classes_ = bundle["classes_"]
        instance.is_fitted = bundle["is_fitted"]
        instance.metadata = bundle.get("metadata", {})
        return instance
