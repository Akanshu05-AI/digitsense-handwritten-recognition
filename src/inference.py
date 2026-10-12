"""
DigitSense: Inference Engine Module
Coordinates image validation, preprocessing, multi-character segmentation,
model prediction (CNN or Classical), and confidence estimation into a unified pipeline.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time
import numpy as np
import torch
import cv2

from config import (
    DEVICE,
    LOW_CONFIDENCE_THRESHOLD,
    TOP_K_DEFAULT,
    CHECKPOINT_CNN_MNIST,
    CHECKPOINT_CNN_EMNIST,
    CHECKPOINT_CLASSICAL_MNIST,
    CHECKPOINT_CLASSICAL_EMNIST
)
from src.preprocessing import (
    load_image,
    preprocess_image,
    to_model_tensor,
    to_classical_features
)
from src.segmentation import segment_characters
from src.cnn_model import DigitSenseCNN, load_cnn_checkpoint
from src.classical_model import ClassicalHOGClassifier
from src.confidence import (
    compute_softmax_probabilities,
    analyze_prediction_confidence
)
from src.postprocessing import HomoglyphDisambiguator
from src.onnx_engine import ONNXDigitSenseEngine
from src.explainability import explain_character_prediction


@dataclass
class CharacterPrediction:
    """Detailed prediction result for an individual segmented character."""
    character: str
    confidence: float
    confidence_percentage: str
    class_index: int
    bbox: Optional[Tuple[int, int, int, int]]
    normalized_crop: np.ndarray
    top_k: List[Dict[str, Any]]
    normalized_entropy: float
    is_low_confidence: bool
    warnings: List[str] = field(default_factory=list)
    gradcam_heatmap: Optional[np.ndarray] = None
    gradcam_overlay: Optional[np.ndarray] = None
    gradcam_analysis: Optional[Dict[str, Any]] = None


@dataclass
class InferenceResult:
    """Unified result bundle for end-to-end recognition."""
    predicted_text: str
    character_predictions: List[CharacterPrediction]
    annotated_image: np.ndarray
    is_multi_character: bool
    latency_ms: float
    model_type: str
    dataset_name: str
    raw_unadjusted_text: Optional[str] = None
    postprocessing_actions: List[Dict[str, Any]] = field(default_factory=list)
    pipeline_preview: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)


class DigitSenseInferenceEngine:
    """
    Unified Inference Engine for DigitSense.
    Caches loaded models and executes single/multi-character recognition pipelines.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or DEVICE
        self.cnn_models: Dict[str, Tuple[DigitSenseCNN, Dict[str, Any]]] = {}
        self.classical_models: Dict[str, ClassicalHOGClassifier] = {}
        self.onnx_engine = ONNXDigitSenseEngine()
        self.disambiguator = HomoglyphDisambiguator()

    def get_model(self, model_type: str = "cnn", dataset_name: str = "mnist"):
        """
        Load or retrieve cached model instance.
        """
        model_type = model_type.lower()
        dataset_name = dataset_name.lower()

        if model_type == "cnn":
            key = f"cnn_{dataset_name}"
            if key not in self.cnn_models:
                if dataset_name == "mnist":
                    ckpt_path = CHECKPOINT_CNN_MNIST
                elif dataset_name in ["emnist", "emnist_balanced"]:
                    ckpt_path = CHECKPOINT_CNN_EMNIST
                else:
                    raise FileNotFoundError(f"CNN checkpoint for unknown dataset '{dataset_name}' not found.")

                if not ckpt_path.exists():
                    raise FileNotFoundError(
                        f"CNN checkpoint for '{dataset_name}' not found at {ckpt_path}. "
                        f"Please train the model first using: python scripts/train_models.py --model cnn --dataset {dataset_name}"
                    )
                model, metadata = load_cnn_checkpoint(ckpt_path, device=self.device)
                self.cnn_models[key] = (model, metadata)
            return self.cnn_models[key]

        elif model_type in ["classical", "svm", "hog"]:
            key = f"classical_{dataset_name}"
            if key not in self.classical_models:
                if dataset_name == "mnist":
                    ckpt_path = CHECKPOINT_CLASSICAL_MNIST
                elif dataset_name in ["emnist", "emnist_balanced"]:
                    ckpt_path = CHECKPOINT_CLASSICAL_EMNIST
                else:
                    raise FileNotFoundError(f"Classical checkpoint for unknown dataset '{dataset_name}' not found.")

                if not ckpt_path.exists():
                    raise FileNotFoundError(
                        f"Classical checkpoint for '{dataset_name}' not found at {ckpt_path}. "
                        f"Please train the model first using: python scripts/train_models.py --model classical --dataset {dataset_name}"
                    )
                clf = ClassicalHOGClassifier.load(ckpt_path)
                self.classical_models[key] = clf
            return self.classical_models[key]

        elif model_type in ["onnx", "onnxruntime"]:
            return self.onnx_engine

        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose 'cnn', 'classical', or 'onnx'.")

    def explain_character(
        self,
        crop_28x28: np.ndarray,
        dataset_name: str = "mnist",
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate Grad-CAM visual explanation for an individual 28x28 character crop.
        """
        cnn_model, metadata = self.get_model("cnn", dataset_name)
        class_names = metadata["class_names"]
        return explain_character_prediction(
            model=cnn_model,
            crop_28x28=crop_28x28,
            target_class=target_class,
            class_names=class_names,
            device=self.device
        )

    def predict_crops(
        self,
        crops_28x28: List[np.ndarray],
        bboxes: List[Tuple[int, int, int, int]],
        model_type: str = "cnn",
        dataset_name: str = "mnist",
        top_k: int = TOP_K_DEFAULT,
        confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD
    ) -> List[CharacterPrediction]:
        """
        Pass a batch of normalized 28x28 crops to the designated model.
        """
        if not crops_28x28:
            return []

        char_results: List[CharacterPrediction] = []
        model_type_clean = model_type.lower()

        if model_type_clean in ["onnx", "onnxruntime"]:
            onnx_analyses = self.onnx_engine.predict_crops(
                crops_28x28=crops_28x28,
                dataset_name=dataset_name,
                top_k=top_k,
                confidence_threshold=confidence_threshold
            )
            for i, analysis in enumerate(onnx_analyses):
                char_results.append(
                    CharacterPrediction(
                        character=analysis["predicted_char"],
                        confidence=analysis["confidence"],
                        confidence_percentage=analysis["confidence_percentage"],
                        class_index=analysis["predicted_index"],
                        bbox=bboxes[i] if i < len(bboxes) else None,
                        normalized_crop=crops_28x28[i],
                        top_k=analysis["top_k"],
                        normalized_entropy=analysis["normalized_entropy"],
                        is_low_confidence=analysis["is_low_confidence"],
                        warnings=analysis["warnings"]
                    )
                )

        elif model_type_clean == "cnn":
            model, metadata = self.get_model("cnn", dataset_name)
            class_names = metadata["class_names"]

            # Prepare batch tensor
            tensors = [to_model_tensor(crop, device=self.device) for crop in crops_28x28]
            batch_tensor = torch.cat(tensors, dim=0)

            model.eval()
            with torch.no_grad():
                logits = model(batch_tensor)
                all_probs = compute_softmax_probabilities(logits)

            for i in range(len(crops_28x28)):
                probs = all_probs[i]
                analysis = analyze_prediction_confidence(
                    probs,
                    class_names=class_names,
                    top_k=top_k,
                    confidence_threshold=confidence_threshold,
                    model_type="CNN"
                )
                char_results.append(
                    CharacterPrediction(
                        character=analysis["predicted_char"],
                        confidence=analysis["confidence"],
                        confidence_percentage=analysis["confidence_percentage"],
                        class_index=analysis["predicted_index"],
                        bbox=bboxes[i] if i < len(bboxes) else None,
                        normalized_crop=crops_28x28[i],
                        top_k=analysis["top_k"],
                        normalized_entropy=analysis["normalized_entropy"],
                        is_low_confidence=analysis["is_low_confidence"],
                        warnings=analysis["warnings"]
                    )
                )

        else:  # Classical
            clf = self.get_model("classical", dataset_name)
            class_names = clf.metadata.get("class_names") or (
                [str(i) for i in range(len(clf.classes_))]
            )
            if hasattr(clf.classes_, "__iter__") and isinstance(clf.classes_[0], (int, np.integer)):
                from src.datasets import get_class_names
                class_names = get_class_names(dataset_name)

            crops_float = np.array([to_classical_features(c) for c in crops_28x28], dtype=np.float32)
            all_probs = clf.predict_proba(crops_float)

            for i in range(len(crops_28x28)):
                probs = all_probs[i]
                analysis = analyze_prediction_confidence(
                    probs,
                    class_names=class_names,
                    top_k=top_k,
                    confidence_threshold=confidence_threshold,
                    model_type="Classical"
                )
                char_results.append(
                    CharacterPrediction(
                        character=analysis["predicted_char"],
                        confidence=analysis["confidence"],
                        confidence_percentage=analysis["confidence_percentage"],
                        class_index=analysis["predicted_index"],
                        bbox=bboxes[i] if i < len(bboxes) else None,
                        normalized_crop=crops_28x28[i],
                        top_k=analysis["top_k"],
                        normalized_entropy=analysis["normalized_entropy"],
                        is_low_confidence=analysis["is_low_confidence"],
                        warnings=analysis["warnings"]
                    )
                )

        return char_results

    def recognize(
        self,
        image_input: Any,
        mode: str = "auto",  # 'auto', 'single', or 'multi'
        model_type: str = "cnn",
        dataset_name: str = "mnist",
        top_k: int = TOP_K_DEFAULT,
        confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
        denoise_method: str = "gaussian",
        threshold_method: str = "otsu",
        fixed_thresh: int = 127,
        sauvola_window: int = 25,
        sauvola_k: float = 0.2,
        normalize_light: bool = False,
        split_ligatures: bool = True,
        disambiguate_homoglyphs: bool = True,
        context_mode: str = "auto",
        generate_cam: bool = False,
        morph_close: bool = True,
        morph_ksize: int = 3,
        merge_fragments: bool = True,
        adaptive_noise_filter: bool = True,
        enhance_contrast_flag: bool = False,
        **kwargs
    ) -> InferenceResult:
        """
        Execute full end-to-end recognition pipeline with advanced preprocessing,
        ligature splitting, inference, Explainable AI, and lexical post-processing.
        """
        start_time = time.perf_counter()
        warnings: List[str] = []

        # 1. Load image
        raw_img = load_image(image_input)

        pipeline_preview = None
        crops = []
        bboxes = []
        annotated_vis = None

        if mode == "single":
            norm_28, preview = preprocess_image(
                raw_img,
                denoise_method=denoise_method,
                threshold_method=threshold_method,
                fixed_thresh=fixed_thresh,
                sauvola_window=sauvola_window,
                sauvola_k=sauvola_k,
                normalize_light=normalize_light,
                enhance_contrast_flag=enhance_contrast_flag
            )
            crops = [norm_28]
            bboxes = [preview.get("bbox") or (0, 0, raw_img.shape[1], raw_img.shape[0])]
            pipeline_preview = preview

            if len(raw_img.shape) == 2:
                annotated_vis = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
            else:
                annotated_vis = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB) if raw_img.shape[2] == 3 else raw_img[:, :, :3]
            is_multi = False

        else:  # 'multi' or 'auto'
            crops, bboxes, annotated_vis, meta = segment_characters(
                raw_img,
                denoise_method=denoise_method,
                threshold_method=threshold_method,
                fixed_thresh=fixed_thresh,
                sauvola_window=sauvola_window,
                sauvola_k=sauvola_k,
                normalize_light=normalize_light,
                split_ligatures=split_ligatures,
                morph_close=morph_close,
                morph_ksize=morph_ksize,
                merge_fragments=merge_fragments,
                adaptive_noise_filter=adaptive_noise_filter,
                enhance_contrast_flag=enhance_contrast_flag
            )
            warnings.extend(meta.get("warnings", []))
            is_multi = meta.get("is_multi_character", False)

            if len(crops) == 0:
                norm_28, preview = preprocess_image(
                    raw_img,
                    denoise_method=denoise_method,
                    threshold_method=threshold_method,
                    fixed_thresh=fixed_thresh,
                    sauvola_window=sauvola_window,
                    sauvola_k=sauvola_k,
                    normalize_light=normalize_light,
                    enhance_contrast_flag=enhance_contrast_flag
                )
                crops = [norm_28]
                bboxes = [(0, 0, raw_img.shape[1], raw_img.shape[0])]
                pipeline_preview = preview
                is_multi = False
                warnings.append("No distinct characters segmented; processed entire image as single character.")

        # 2. Run model prediction
        char_predictions = self.predict_crops(
            crops_28x28=crops,
            bboxes=bboxes,
            model_type=model_type,
            dataset_name=dataset_name,
            top_k=top_k,
            confidence_threshold=confidence_threshold
        )

        raw_unadjusted_text = "".join(c.character for c in char_predictions)
        postprocessing_actions: List[Dict[str, Any]] = []

        # 3. Context-Aware Homoglyph Disambiguation
        if disambiguate_homoglyphs and char_predictions:
            raw_chars = [c.character for c in char_predictions]
            top_k_lists = [c.top_k for c in char_predictions]
            resolved_text, postprocessing_actions = self.disambiguator.disambiguate(
                predicted_chars=raw_chars,
                top_k_lists=top_k_lists,
                mode=context_mode
            )
            # Update prediction characters to match disambiguated tokens
            for i, ch in enumerate(resolved_text):
                char_predictions[i].character = ch
            predicted_text = resolved_text
        else:
            predicted_text = raw_unadjusted_text

        # 4. Optional Grad-CAM Generation
        if generate_cam and char_predictions:
            for pred in char_predictions:
                try:
                    exp = self.explain_character(
                        crop_28x28=pred.normalized_crop,
                        dataset_name=dataset_name,
                        target_class=pred.class_index
                    )
                    pred.gradcam_heatmap = exp["heatmap"]
                    pred.gradcam_overlay = exp["blended_overlay"]
                    pred.gradcam_analysis = exp["attention_analysis"]
                except Exception:
                    pass

        # 5. Annotate character predictions onto the visualization
        if annotated_vis is not None and len(bboxes) == len(char_predictions):
            for i, pred in enumerate(char_predictions):
                bx, by, bw, bh = bboxes[i]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                label_str = f"'{pred.character}' ({pred.confidence*100:.0f}%)"
                (tw, th), _ = cv2.getTextSize(label_str, font, font_scale, thickness)

                tag_y = min(annotated_vis.shape[0] - 5, by + bh + th + 6)
                cv2.rectangle(annotated_vis, (bx, tag_y - th - 4), (bx + tw + 4, tag_y + 2), (16, 185, 129), -1)
                cv2.putText(annotated_vis, label_str, (bx + 2, tag_y - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        total_latency = (time.perf_counter() - start_time) * 1000.0

        return InferenceResult(
            predicted_text=predicted_text,
            character_predictions=char_predictions,
            annotated_image=annotated_vis if annotated_vis is not None else raw_img,
            is_multi_character=is_multi,
            latency_ms=total_latency,
            model_type=model_type.upper(),
            dataset_name=dataset_name,
            raw_unadjusted_text=raw_unadjusted_text,
            postprocessing_actions=postprocessing_actions,
            pipeline_preview=pipeline_preview,
            warnings=warnings
        )
