# DigitSense: Handwritten Digit and Character Recognition System

> **A Complete AI/ML, Computer Vision & Deep Learning Capstone Project**  
> **Department of Computer Science and Engineering (Artificial Intelligence & Machine Learning)**  
> **IILM University, Greater Noida**

---

## 👥 Academic Credentials & Team

- **Faculty Project Guide:** Mr. Vikas Kumar Tiwari  
- **Institution:** IILM University, Greater Noida  
- **Project Team Members:**
  - **Yash Mahindroo**
  - **Isha Shekhar**
  - **Akanshu Goel**
  - **Shivam Maddheshiya**

---

## 📄 Abstract

Handwritten text recognition remains one of the core challenges in pattern recognition and computer vision due to significant interpersonal variance in stroke width, letterforms, slant, cursive connectivity, and background noise. **DigitSense** is a modular, high-performance, and locally runnable handwritten character recognition system designed for real-world numeric and alphanumeric inputs. The system integrates classical computer vision preprocessing (polarity detection, Otsu thresholding, LeCun-style aspect-ratio preserving centering), connected-component segmentation for multi-character horizontal handwriting, and dual inference paradigms: a **Classical Machine Learning baseline** (HOG feature extraction + Calibrated Linear Support Vector Classifier) and a **Deep Learning Convolutional Neural Network (PyTorch)**. The application provides uncertainty quantification via normalized Shannon entropy and class-calibrated probabilities, wrapped inside an interactive **Streamlit** user interface.

---

## 🎯 Problem Statement & Objectives

### Problem Statement
In domains such as banking check processing, historical document archiving, medical prescription digitization, and academic grading, manual data transcription is error-prone, labor-intensive, and slow. Existing commercial OCR solutions often perform poorly on unconstrained handwriting or require cloud dependencies. Moreover, standard benchmark models trained on isolated $28 \times 28$ crops fail when presented with full handwritten words or multi-character sequences containing uneven backgrounds or inverted polarities.

### Core Objectives
1. **End-to-End Recognition:** Accept raw uploaded handwriting or digital sketches and recognize single digits, alphabetic characters, and multi-character sequences.
2. **Robust Computer Vision Preprocessing:** Automatically detect background polarity (dark-on-light vs. light-on-dark), apply noise reduction, and center characters according to the MNIST center-of-mass standard.
3. **Multi-Character Segmentation:** Implement external contour analysis to isolate candidate character regions, filter noise specks, and sequence bounding boxes in horizontal reading order.
4. **Empirical Model Comparison:** Train and evaluate both a Classical Baseline (HOG + LinearSVC) and a Deep Learning CNN (PyTorch) on identical held-out test sets.
5. **Calibrated Confidence & Uncertainty:** Quantify prediction reliability through Softmax/Platt-calibrated probabilities, top-k candidate rankings, and normalized Shannon entropy.
6. **Usable Local UI:** Deliver a responsive, professional Streamlit web application with drawing canvas, live preprocessing inspection, model comparison, and dataset exploration.

---

## 🔬 Research Gap & Proposed Contribution

| Aspect | Traditional Student Projects | DigitSense Implementation |
| :--- | :--- | :--- |
| **Input Format** | Expects pre-cropped $28 \times 28$ images only | Accepts arbitrary dimensions, sketches, and multi-character strings |
| **Polarity** | Hardcoded to white strokes on black backgrounds | Dynamic polarity detection; inverts dark ink on paper seamlessly |
| **Segmentation** | Absent (single character only) | Contour-based multi-character segmentation with reading-order sorting |
| **Model Scope** | Single demo model with unverified metrics | Dual-paradigm benchmark (Classical HOG-SVM vs. Deep CNN) |
| **Confidence** | Naive max-probability display | Platt-calibrated probabilities, normalized Shannon entropy, low-confidence warning |
| **Datasets** | MNIST only (10 digits) | MNIST (10 digits) + EMNIST Balanced (47 alphanumeric classes) with orientation correction |

---

## 🏗️ System Architecture

The DigitSense pipeline is organized into modular decoupled stages:

```
[Input: Interactive Canvas / File Upload / Webcam Snapshot / Benchmark Gallery]
                                    │
                                    ▼
                      [1. Image Ingestion & Validation]
                                    │
                                    ▼
                      [2. Computer Vision Preprocessing]
       ├── Grayscale Conversion & Alpha Blending
       ├── Border Polarity Detection (Dark vs Light BG)
       ├── Polarity Normalization (White Stroke on Dark BG)
       ├── Flat-Field Illumination Normalization (Shadow Removal)
       ├── Noise Reduction (Gaussian / Median / Bilateral)
       └── Binarization (Otsu Adaptive / Sauvola Local Adaptive / Fixed)
                                    │
                                    ▼
                      [3. Character Segmentation Module]
       ├── External Contour Extraction (cv2.RETR_EXTERNAL)
       ├── Noise Dot Filtering (Area < Min Threshold)
       ├── Touching Ligature Splitting (Vertical Projection Profile Stroke Valley Bisection)
       ├── Line Grouping & Left-to-Right Horizontal Reading Order Sorting
       ├── Aspect-Ratio Preserving Resize (Fit inside 20x20 box)
       └── Center-of-Mass Translation onto 28x28 Canvas
                                    │
                                    ▼
                      [4. Multi-Paradigm Classification Engine]
       ├── Path A: Classical HOG Feature Extraction -> Calibrated LinearSVC
       ├── Path B: 3-Stage Convolutional Neural Network (PyTorch CUDA/CPU)
       └── Path C: ONNX Runtime SIMD Vectorized Graph Accelerated Engine
                                    │
                                    ▼
                      [5. Explainable AI & Salience Mapping]
       ├── PyTorch Grad-CAM Hook into Final Convolutional Layer (conv5)
       ├── Channel-Wise Gradient Pooling & Activation Weighted Sum
       ├── Bilinear Heatmap Upsampling to 28x28 & Jet Colormap Overlay
       └── Anatomical Attention Attribution Breakdown (Top/Middle/Bottom zones)
                                    │
                                    ▼
                      [6. Lexical Post-Processing & Disambiguation]
       ├── Context-Aware Homoglyph Disambiguation ('0' vs 'O', '1' vs 'I', '5' vs 'S', '8' vs 'B')
       ├── Token N-gram Constraints (Numeric Context vs Alphabetic Context vs Mixed)
       └── Top-K Alternative Candidate Re-Ranking
                                    │
                                    ▼
                      [7. Confidence & Interactive UI Presentation]
       ├── Softmax Posterior / Calibrated Probabilities
       ├── Top-5 Alternative Candidates & Normalized Shannon Entropy
       ├── Annotated Reading-Order Bounding Box Visualizer
       └── Streamlit Studio, Live Webcam Capture & Preprocessing Lab
```

### Mathematical Formulations
- **Softmax Posterior Probability:**
  $$P(y = c \mid \mathbf{x}) = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}}$$
- **Normalized Shannon Prediction Entropy:**
  $$\mathcal{H}(\mathbf{p}) = -\frac{1}{\log_2 C} \sum_{i=1}^{C} p_i \log_2(p_i) \quad \in [0.0, 1.0]$$
- **Grad-CAM Salience Activation:**
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right), \quad \alpha_k^c = \frac{1}{Z} \sum_{i=1}^{H} \sum_{j=1}^{W} \frac{\partial y^c}{\partial A_{i,j}^k}$$
- **Sauvola Local Adaptive Thresholding:**
  $$T(x,y) = m(x,y) \cdot \left[1 + k \cdot \left(\frac{s(x,y)}{R} - 1\right)\right]$$
- **Vertical Projection Profile for Ligature Splitting:**
  $$V(x) = \sum_{y=0}^{H-1} \mathbb{I}(B(y, x) > 0), \quad x_{\text{split}} = \arg\min_{x \in [0.25W, 0.75W]} V(x)$$

---

## 📊 Datasets & Class Mappings

DigitSense supports two established open-source benchmarks:

### 1. MNIST (Modified National Institute of Standards and Technology)
- **Classes:** 10 digits (`0` to `9`).
- **Samples:** 60,000 training images, 10,000 held-out test images.
- **Resolution:** $28 \times 28$ grayscale pixels.

### 2. EMNIST Balanced (Extended MNIST)
- **Split Used:** `balanced` (derived from NIST Special Database 19).
- **Classes:** 47 balanced classes (0–9 digits, A–Z uppercase letters, plus 11 lowercase letters with distinct handwritten shapes: `a, b, d, e, f, g, h, n, q, r, t`).
- **Merged Classes:** Lowercase letters identical in geometry to their uppercase forms (e.g., C, I, J, K, L, M, O, P, S, U, V, W, X, Y, Z) are merged into their respective uppercase class to avoid arbitrary label confusion.
- **Orientation Correction:** EMNIST raw binary files in the original NIST format are stored transposed. DigitSense automatically applies an orientation transform (`Image.TRANSPOSE`) during ingestion.

---

## 💻 Technology Stack

- **Language:** Python 3.12+
- **Deep Learning Framework:** PyTorch 2.5+ (CUDA GPU & CPU)
- **Inference Acceleration:** ONNX, ONNX Runtime (`onnxruntime`), `onnxscript`
- **Explainable AI (XAI):** Gradient-weighted Class Activation Mapping (Grad-CAM)
- **Computer Vision & Image Processing:** OpenCV (`opencv-python`), Pillow, scikit-image
- **Classical Machine Learning:** Scikit-Learn (LinearSVC, CalibratedClassifierCV)
- **Numerical Computing:** NumPy, SciPy
- **Visualization:** Matplotlib, Seaborn
- **User Interface:** Streamlit, Streamlit-Drawable-Canvas, Live Camera Capture (`st.camera_input`)
- **Testing & Quality Assurance:** pytest (46 unit & integration tests)
- **Model Serialization:** Joblib, PyTorch Checkpoints (`.pt`), ONNX Models (`.onnx`)

---

## 📁 Project Directory Structure

```
DigitSense/
├── app.py                      # Main interactive Streamlit application
├── requirements.txt            # Python dependencies specification
├── README.md                   # Comprehensive capstone project documentation
├── .gitignore                  # Git tracking exclusion rules
├── config.py                   # Global constants, paths, and hyperparameters
├── data/
│   ├── README.md               # Dataset documentation and specs
│   ├── raw/                    # Downloaded benchmark datasets (MNIST, EMNIST)
│   ├── processed/              # Processed arrays and cache
│   └── samples/                # Demo handwriting samples (single, multi-digit, letters)
├── models/
│   ├── README.md               # Model checkpoints documentation
│   └── checkpoints/            # Trained weights (.pt, .joblib, and .onnx)
│       ├── cnn_mnist.pt
│       ├── cnn_mnist.onnx
│       ├── cnn_emnist_balanced.pt
│       ├── cnn_emnist_balanced.onnx
│       ├── classical_mnist.joblib
│       └── classical_emnist.joblib
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Grayscale, polarity, Sauvola thresholding, shadow removal, LeCun centering
│   ├── segmentation.py         # Contour detection, VPP ligature splitting, reading-order sorting
│   ├── explainability.py       # Grad-CAM attention heatmap generation and overlay analysis
│   ├── postprocessing.py       # Context-aware homoglyph disambiguation & top-k re-ranking
│   ├── onnx_engine.py          # High-performance ONNX Runtime inference & latency benchmarking
│   ├── datasets.py             # MNIST/EMNIST ingestion and orientation correction
│   ├── classical_model.py      # HOG feature extractor + Calibrated LinearSVC
│   ├── cnn_model.py            # PyTorch 3-stage Conv-BN-Dropout CNN architecture
│   ├── train.py                # Reproducible training pipelines with early stopping
│   ├── evaluate.py             # Evaluation on held-out test data, confusion matrices, latency
│   ├── inference.py            # Unified end-to-end recognition engine
│   ├── confidence.py           # Softmax, Platt calibration, top-k ranking, Shannon entropy
│   └── visualization.py        # Grad-CAM maps, confusion matrices, bar charts, pipeline grids
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py   # Unit tests for image preprocessing & Sauvola
│   ├── test_segmentation.py    # Unit tests for character segmentation & sorting
│   ├── test_ligature_splitting.py # Unit tests for vertical projection profile splitting
│   ├── test_explainability.py  # Unit tests for Grad-CAM forward/backward hooks & heatmaps
│   ├── test_postprocessing.py  # Unit tests for homoglyph lexical disambiguation
│   ├── test_onnx.py            # Unit tests for ONNX Runtime engine & benchmark
│   ├── test_inference.py       # Unit tests for CNN, confidence, and engine
│   └── test_integration.py     # End-to-end integration tests (CNN, Classical, ONNX, Sauvola, CAM)
├── scripts/
│   ├── download_data.py        # CLI script to download and verify datasets
│   ├── train_models.py         # CLI script to train Classical and CNN models
│   ├── export_onnx.py          # CLI script to export PyTorch models to ONNX
│   ├── evaluate_models.py      # CLI script to evaluate models and output summary tables
│   └── create_samples.py       # Helper script to generate demo test images
└── reports/
    └── results/                # Evaluated metrics JSON and confusion matrix PNGs
```

---

## 🚀 Windows Setup Instructions (Step-by-Step)

The project is designed to run locally on Windows using PowerShell or Command Prompt.

### Step 1: Open Terminal in Project Root
```powershell
cd "c:\Desktop\Projects\digi detection"
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Upgrade pip and Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📥 Dataset Download Commands

To download and extract the official MNIST and EMNIST benchmarks:

```powershell
# Download all datasets (MNIST and EMNIST Balanced)
python scripts/download_data.py --dataset all

# Or download individual datasets:
python scripts/download_data.py --dataset mnist
python scripts/download_data.py --dataset emnist
```
*Note: Downloading requires an internet connection once; subsequent runs execute entirely offline.*

---

## 🏋️ Model Training Commands

Train the Classical ML baseline and Deep Learning CNN models:

```powershell
# Train both models on both datasets (MNIST + EMNIST):
python scripts/train_models.py --model all --dataset all

# Train CNN on MNIST (10 epochs, with early stopping):
python scripts/train_models.py --model cnn --dataset mnist --epochs 10

# Train Classical HOG-SVM baseline on MNIST (sub-minute on CPU):
python scripts/train_models.py --model classical --dataset mnist

# Train CNN on EMNIST Balanced (47 alphanumeric classes):
python scripts/train_models.py --model cnn --dataset emnist --epochs 8
```
*Trained checkpoints are automatically saved to `models/checkpoints/`.*

---

## 📊 Model Evaluation Commands

Evaluate trained models on held-out test sets to generate real quantitative performance metrics:

```powershell
# Evaluate all trained models on held-out test sets:
python scripts/evaluate_models.py --dataset all

# Evaluate MNIST models only:
python scripts/evaluate_models.py --dataset mnist
```
*Outputs JSON metrics and confusion matrix heatmaps to `reports/results/`.*

---

## 🖥️ Launching the Web Application

Launch the Streamlit graphical interface:

```powershell
streamlit run app.py
```
The application will start locally at `http://localhost:8501`.

---

## 🧪 Automated Testing

Execute the automated pytest suite:

```powershell
pytest -v tests
```
*Executes 53 automated unit and integration tests covering image loading, polarity normalization, CLAHE contrast enhancement, Sauvola binarization, bounding-box extraction, stroke consolidation, scale-adaptive noise suppression, vertical ligature splitting, reading-order sorting, CNN forward passes, ONNX runtime parity, Grad-CAM explainability, context-aware homoglyph disambiguation, and Shannon entropy.*

---

## 🚀 Advanced Capabilities Implemented

1. **⚡ ONNX Runtime Vectorized Acceleration:** Exported graph-optimized ONNX models (`cnn_mnist.onnx`, `cnn_emnist_balanced.onnx`) providing sub-10ms CPU inference latency.
2. **🔍 PyTorch Grad-CAM Explainable AI:** Real-time salience feature attribution overlays hooking directly into the final convolutional layer with anatomical attention zone breakdown.
3. **🧩 Robust Multi-Stroke Consolidation:** Intelligent bounding-box merging fuses broken strokes, stencils, and fragmented letters without over-merging adjacent distinct characters.
4. **🛡️ Scale-Adaptive Noise Suppression:** Dynamically filters out background cloth weave, sensor noise, and lint based on median character dimensions.
5. **🪓 Vertical Ligature Splitting:** Bisection of touching handwriting characters using Vertical Projection Profile (VPP) valley analysis.
6. **🔡 Context-Aware Homoglyph Disambiguation:** Resolves ambiguous characters (`0` vs `O`, `1` vs `I`, `5` vs `S`, `8` vs `B`) using sequence token modality and top-$k$ alternate re-ranking.
7. **📷 Real-Time Webcam Snapshot Capture:** Direct photo capture with illumination normalization and CLAHE contrast enhancement.

---

## 🔮 Future Roadmap

- Integrate an unconstrained CRNN + CTC (Connectionist Temporal Classification) sequence recognition model for connected cursive text.
- Add multi-language script expansion (e.g., Devanagari script numeral recognition).
- Package as an installable desktop application via Electron / PyInstaller.
