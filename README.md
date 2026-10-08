# 🔴 Martian Eye — Autonomous Dust Devil Intelligence & Detection Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-FFD43B?style=for-the-badge&logo=python&logoColor=blue)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.0-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> **Martian Eye** is an autonomous, end-to-end deep learning platform engineered for continuous atmospheric monitoring and dust devil vortex plume detection across Martian surface rover imagery (*Navcam*, *Pancam*) and satellite orbital imagery (*HiRISE*).

---

## 📋 Table of Contents
- [🌟 Key Innovations & Features](#-key-innovations--features)
- [🔬 Core Model Architecture](#-core-model-architecture)
- [🧮 Computer Vision & Plume Detection Algorithm](#-computer-vision--plume-detection-algorithm)
- [🖥️ Web Dashboard & UI Capabilities](#️-web-dashboard--ui-capabilities)
- [📁 Project Folder Structure](#-project-folder-structure)
- [🚀 Quick Start & Installation](#-quick-start--installation)
- [📡 API Documentation](#-api-documentation)
- [📊 Evaluation Benchmarks](#-evaluation-benchmarks)

---

## 🌟 Key Innovations & Features

### 1. Faint & Small Plume Detection
- **Multi-Scale Feature Saliency**: Detects thin, short, tall, or diffuse dust columns that are visually faint against Martian sky or surface background.
- **Camera Frame Border Masking**: Automatically detects and erases slanted black camera padding borders and camera frame artifacts to prevent border false positives.
- **Localized Surround Delta Check ($\Delta_{\text{surround}}$)**: Evaluates local contrast against adjacent horizontal background terrain to distinguish localized vertical dust plumes ($\Delta \ge 2.0$) from flat horizon lines ($\Delta = 0.0$).

### 2. Stage-2 False-Positive Verification
- **Hard Negative Screening**: Uses a MobileNet-V3 verifier to classify ROI patches into 6 distinct categories (*True Dust Devil*, *Crater Rim*, *Ridge Shadow*, *Dune Ripple*, *Camera Artifact*, *Atmospheric Haze*).

### 3. Explainable AI (XAI) & Epistemic Uncertainty
- **Grad-CAM Attention Heatmaps**: Real-time visualization of model spatial attention weights.
- **Explanation Quality Metrics**: Calculates spatial overlap alignment (*Strong*, *Review Required*, *Clean Frame*).
- **Epistemic Uncertainty Estimation**: Dynamic uncertainty risk bands (*Low* vs. *Review*).

### 4. Interactive Web Control & Persistent Storage
- **Mission Control Web App**: Built with FastAPI and TailwindCSS, featuring a 3D animated solar system canvas background.
- **Persistent Observations Database**: Save detection records and telemetry to `data/storage/saved_observations.json`.
- **Review Queue Workflow**: Queue low-confidence or ambiguous detections to `data/storage/review_queue.json` for human review.
- **Instant JSON Telemetry Export**: Download detailed observation telemetry in JSON format with a single click.

---

## 🔬 Core Model Architecture

```
                    ┌─────────────────────────────────────────┐
                    │    Input Mars Image (256x256 / RGB)     │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │   ResNet-34 Feature Pyramid Backbone    │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │  4-Head Multiscale Cross-Attention      │
                    │         Transformer Module              │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
       ┌─────────────────────────────────┴─────────────────────────────────┐
       │                                                                   │
       ▼                                                                   ▼
┌───────────────┐                                           ┌─────────────────────────────┐
│ Objectness &  │                                           │  Candidate ROI Crop (128x128)│
│ Box Prediction│                                           └──────────────┬──────────────┘
└───────┬───────┘                                                          │
        │                                                                  ▼
        │                                                   ┌─────────────────────────────┐
        │                                                   │   Stage-2 MobileNet-V3      │
        │                                                   │    False-Positive Verifier  │
        │                                                   └──────────────┬──────────────┘
        │                                                                  │
        └────────────────────────────────┬─────────────────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │  Unified Plume Score & XAI Output       │
                    └─────────────────────────────────────────┘
```

---

## 🧮 Computer Vision & Plume Detection Algorithm

The candidate box extractor computes a **Fused Saliency Map**:

$$\text{Saliency} = 0.30 \cdot \text{Attention} + 0.40 \cdot \text{MultiVert} + 0.15 \cdot \text{DoG} + 0.15 \cdot \text{SobelVert}$$

Where:
- **Attention**: Spatial cross-attention weights from the Multiscale Transformer.
- **MultiVert**: Multi-kernel vertical top-hat and black-hat morphological responses ($k_h \in \{7, 13, 21, 33, 45\}$).
- **DoG**: Difference-of-Gaussians ($\sigma_1=3, \sigma_2=17$) capturing faint contrast boundaries.
- **SobelVert**: Sobel vertical gradient magnitude.

Candidates are thresholded across multi-level percentiles ($p \in \{55, 68, 78, 86\}$) and filtered using Non-Maximum Suppression ($IoU > 0.25$).

---

## 🖥️ Web Dashboard & UI Capabilities

The dashboard provides **10 dedicated work centers**:

1. 🎯 **Detect**: Upload custom imagery or load test samples, adjust threshold sliders ($0.1 - 0.9$), and switch view modes (*Original*, *Detection*, *Attention*, *Contrast*, *Crop*).
2. 🔍 **Explain**: Grad-CAM attention maps, focus quality indicators, and prediction evidence summaries.
3. 📐 **Analyze**: Morphology properties (pixel width, height, aspect ratio, orientation angle).
4. 🛡️ **Verification**: Stage-2 False-Positive classifier status and hard negative screening breakdown.
5. 🌐 **Domain Lab**: Cross-mission domain adaptation benchmark visualizer.
6. 🧪 **SSL Lab**: Self-supervised learning benchmarking under label scarcity ($25\%$, $50\%$, $100\%$).
7. 📊 **Experiments**: Interactive ablation study comparison table ($E_0 \rightarrow E_5$).
8. 🛰️ **Tracking**: Temporal sequence tracking interface.
9. ℹ️ **About**: Architectural diagram and system documentation.
10. 🌌 **Dashboard**: Hero landing page with direct launch control.

---

## 📁 Project Folder Structure

```
Martian Dust Devil Detection/
├── backend/
│   ├── app.py                    # FastAPI server & storage API endpoints
│   ├── pipeline.py               # Multiscale inference & plume detection engine
│   └── cache/                    # Generated visualization images & heatmaps
├── checkpoints/
│   ├── multiscale_detector.pt    # Multiscale Transformer PyTorch weights
│   ├── fp_verifier.pt            # Stage-2 False-Positive Verifier weights
│   └── metrics_summary.json      # Quantitative benchmark statistics
├── data/
│   ├── train/                    # Training COCO dataset & images
│   ├── valid/                    # Validation COCO dataset & images
│   ├── test/                     # Test COCO dataset & images
│   └── storage/                  # Persistent saved observations & review queue JSON
├── frontend/
│   └── index.html                # Single-page Mission Control dashboard
├── src/
│   ├── models/
│   │   ├── multiscale_transformer.py  # ResNet-34 + 4-Head Attention Transformer
│   │   └── fp_classifier.py           # MobileNet-V3 Stage-2 Candidate Verifier
│   ├── explainability/
│   │   └── gradcam.py                 # Grad-CAM heatmap generator
│   ├── morphology/
│   │   └── analyzer.py                # Bounding box & shape morphology analyzer
│   ├── uncertainty/
│   │   └── calibration.py             # Uncertainty estimator
│   ├── data/
│   │   └── dataset.py                 # PyTorch COCO dataset loader
│   ├── domain_adaptation/
│   │   └── adapter.py                 # Cross-mission domain generalizer
│   ├── evaluation/
│   │   └── ablation.py                # Ablation matrix evaluator
│   └── self_supervised/
│       └── ssl_runner.py              # SSL benchmark runner
├── README.md                     # Project documentation
└── requirements.txt              # Project dependencies
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Martian-Dust-Devil-Detection.git
cd Martian-Dust-Devil-Detection
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Using standard venv
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Launch Mission Control Web App
```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
Open **`http://localhost:8000`** in your browser to access the web application.

---

## 📡 API Documentation

### `POST /api/detect`
Performs detection on an uploaded image.
- **Query Params**: `threshold` (float, default: `0.5`)
- **Body**: `multipart/form-data` (file)
- **Response**: Detection results, bounding boxes, morphology, XAI evidence, and image URLs.

### `POST /api/observation/save`
Saves current detection telemetry into persistent storage (`data/storage/saved_observations.json`).

### `POST /api/queue/add`
Adds low-confidence detection candidate to human review queue (`data/storage/review_queue.json`).

### `GET /api/observations/list` & `GET /api/queue/list`
Lists saved observation records and review queue candidates.

---

## 📊 Evaluation Benchmarks

| Experiment Stage | Description | Precision | Recall | F1 Score | IoU | mAP |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$E_0$ Baseline** | ResNet-34 Standard Classifier | 0.812 | 0.735 | 0.771 | 0.652 | 0.784 |
| **$E_1$ FPN** | ResNet-34 + Feature Pyramid | 0.884 | 0.792 | 0.835 | 0.721 | 0.849 |
| **$E_2$ Transformer** | Multiscale Spatial Cross-Attention | 0.921 | 0.825 | 0.870 | 0.768 | 0.892 |
| **$E_3$ Verifier** | Multiscale Transformer + Stage-2 Verifier | **0.948** | **0.842** | **0.892** | **0.794** | **0.912** |

---

<div align="center">
  <sub>Martian Eye • Autonomous Dust Devil Intelligence & Planetary Vision Systems</sub>
</div>
