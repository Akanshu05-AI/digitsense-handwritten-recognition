"""
DigitSense: Visualization Utilities Module
Provides Matplotlib and Seaborn plotting routines for confusion matrices,
top-k confidence distributions, preprocessing previews, and misclassified galleries.
"""

from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Apply clean scientific aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    normalize: bool = True,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Generate an annotated confusion matrix heatmap.
    """
    if normalize:
        with np.errstate(all='ignore'):
            row_sums = cm.sum(axis=1)[:, np.newaxis]
            cm_display = np.where(row_sums > 0, cm.astype('float') / row_sums, 0.0)
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_display = cm
        fmt = "d"
        vmax = None

    num_classes = len(class_names)
    if figsize is None:
        if num_classes <= 10:
            figsize = (7, 6)
        else:
            figsize = (14, 12)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = sns.color_palette("Blues", as_cmap=True)

    show_annot = num_classes <= 15  # Do not crowd heatmap text if 47 classes

    sns.heatmap(
        cm_display,
        annot=show_annot,
        fmt=fmt,
        cmap=cmap,
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        vmax=vmax,
        vmin=0.0 if normalize else None,
        ax=ax,
        linewidths=0.5 if num_classes <= 15 else 0.0
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45 if num_classes > 10 else 0, ha="right" if num_classes > 10 else "center", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    return fig


def plot_top_k_confidence(
    top_k_data: List[Dict[str, Any]],
    title: str = "Top Predictions & Confidence Distribution"
) -> plt.Figure:
    """
    Generate horizontal bar chart of top-k predicted candidates.
    """
    chars = [f"'{item['character']}'" for item in reversed(top_k_data)]
    probs = [item["probability"] * 100.0 for item in reversed(top_k_data)]

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#3b82f6" if i < len(probs) - 1 else "#10b981" for i in range(len(probs))]

    bars = ax.barh(chars, probs, color=colors, height=0.55, edgecolor="none")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    # Add numeric labels to bars
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color="#1e293b"
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_preprocessing_stages(preview_dict: Dict[str, Any]) -> plt.Figure:
    """
    Create a multi-panel visual comparative breakdown of the preprocessing pipeline stages.
    """
    stages = [
        ("Original", preview_dict.get("original"), "gray" if preview_dict.get("original", np.array([])).ndim == 2 else None),
        ("Grayscale", preview_dict.get("grayscale"), "gray"),
        ("Polarity Fixed", preview_dict.get("polarity_corrected"), "gray"),
        ("Denoised", preview_dict.get("denoised"), "gray"),
        ("Binary", preview_dict.get("binary"), "gray"),
        ("Normalized 28x28", preview_dict.get("normalized_28x28"), "gray")
    ]

    fig, axes = plt.subplots(1, len(stages), figsize=(14, 2.8))
    for ax, (name, img, cmap) in zip(axes, stages):
        if img is not None and img.size > 0:
            if img.ndim == 3 and img.shape[2] == 3 and cmap is None:
                # If RGB
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap or "gray")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.suptitle("DigitSense Preprocessing Lab: Step-by-Step Transformation", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_misclassified_gallery(
    misclassified_items: List[Dict[str, Any]],
    max_items: int = 10
) -> plt.Figure:
    """
    Display a row of misclassified examples showing true label and predicted label.
    """
    items_to_show = misclassified_items[:max_items]
    num_items = len(items_to_show)

    if num_items == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No misclassifications found (100% accuracy on sample)!", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(1, num_items, figsize=(min(num_items * 2, 16), 2.5))
    if num_items == 1:
        axes = [axes]

    for ax, item in zip(axes, items_to_show):
        img = item["image"]
        true_lbl = item["true_label"]
        pred_lbl = item["pred_label"]
        conf = item.get("confidence", 0.0)

        ax.imshow(img, cmap="gray")
        ax.set_title(f"True: '{true_lbl}'\nPred: '{pred_lbl}'\n({conf * 100:.1f}%)", fontsize=8, color="#dc2626")
        ax.axis("off")

    plt.suptitle("Analysis of Misclassified Samples", fontsize=12, fontweight="bold", y=1.05)
    plt.tight_layout()
    return fig


def plot_gradcam_explanation(
    crop_28x28: np.ndarray,
    heatmap_28x28: np.ndarray,
    overlay_rgb: np.ndarray,
    predicted_label: str = "",
    confidence_str: str = ""
) -> plt.Figure:
    """
    Generate a 3-panel side-by-side visualization of:
    1. Grayscale character crop
    2. Grad-CAM salience heatmap
    3. Heatmap overlaid on handwriting strokes
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))

    # Panel 1: Grayscale Input
    axes[0].imshow(crop_28x28, cmap="gray")
    axes[0].set_title("Input Stroke", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Normalized Activation Heatmap
    im2 = axes[1].imshow(heatmap_28x28, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("Grad-CAM Salience", fontsize=9, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Blended Overlay
    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Salience Overlay", fontsize=9, fontweight="bold")
    axes[2].axis("off")

    title_text = f"Explainable AI: Model Attention for '{predicted_label}'" if predicted_label else "Grad-CAM Visual Attention"
    if confidence_str:
        title_text += f" ({confidence_str})"
    fig.suptitle(title_text, fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig

