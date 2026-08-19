<div align="center">

# AI-Chemistry: Multi-Task Deep Learning & REST API Engine for Inorganic Nitrogen Monitoring

[![CI](https://github.com/ntthienphuc/AI---Chemistry/actions/workflows/ci.yml/badge.svg)](https://github.com/ntthienphuc/AI---Chemistry/actions/workflows/ci.yml)
[![Protocol](https://img.shields.io/badge/Protocol-paper--v1.0-2ea44f?style=for-the-badge&logo=git)](https://github.com/ntthienphuc/AI---Chemistry/releases/tag/paper-v1.0)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

*Official AI, Computer Vision, and REST API Backend repository accompanying the research article:*  
**"A Smartphone-AI-WebGIS Platform for Real-Time Monitoring of Inorganic Nitrogen in Water"**

---

</div>

## 📌 Platform Architecture & System Integration

This repository hosts the **core AI, computer vision pipeline, multi-task neural network architectures, frozen evaluation manifests, and REST API inference microservice** that powers the analytical backend of the integrated Smartphone-AI-WebGIS platform.

```text
 ┌──────────────────────────────────────────────────────────────────────────────────────────┐
 │  1. ON-SITE DATA ACQUISITION & LOCALIZATION                                              │
 │                                                                                          │
 │   ┌────────────────────────┐       ┌────────────────────────┐       ┌──────────────────┐ │
 │   │   Smartphone Imaging   │  ───► │  YOLO11n-seg Detection │  ───► │ Green-Border     │ │
 │   │ (Standardized Enclosure│       │ (Autonomous Strip ROI) │       │ Color Normalizer │ │
 │   │  + Fixed White LED)    │       └────────────────────────┘       └──────────────────┘ │
 │   └────────────────────────┘                                                 │           │
 └──────────────────────────────────────────────────────────────────────────────│───────────┘
                                                                                │ (Linear RGB)
 ┌──────────────────────────────────────────────────────────────────────────────▼───────────┐
 │  2. MULTI-TASK HETEROSCEDASTIC DEEP NEURAL NETWORK (AI-CHEMISTRY CORE)                   │
 │                                                                                          │
 │  Shared Vision Backbone: ConvNeXt-T / MobileNetV3 / Swin-T / NFNet-F2 / EffNet-B0 / TFB3 │
 │                                                                                          │
 │  ├── Task Head 1 (Classification): Linear(d, 512) -> ReLU -> Dropout -> Linear(512, 2)  │
 │  │   └── Analyte Identification: Ammonium (NH4+) vs. Nitrite (NO2-)                      │
 │  │                                                                                       │
 │  ├── Task Head 2 (NH4+ Quantification): Linear(d, 512) -> ReLU -> Dropout -> Linear(512, 2)
 │  │   └── Predicted Concentration (mu) + Aleatoric Log-Variance (log_var)                 │
 │  │                                                                                       │
 │  └── Task Head 3 (NO2- Quantification): Linear(d, 512) -> ReLU -> Dropout -> Linear(512, 2)
 │      └── Predicted Concentration (mu) + Aleatoric Log-Variance (log_var)                 │
 └──────────────────────────────────────────────────────────────────────────────│───────────┘
                                                                                │
 ┌──────────────────────────────────────────────────────────────────────────────▼───────────┐
 │  3. REST API MICROSERVICE & GEOSPATIAL WEBGIS DELIVERY                                   │
 │                                                                                          │
 │  ├── FastAPI Cloud/Edge Inference Endpoint (`POST /predict`)                             │
 │  ├── Analytical Quantification with Optional Approximate 95% Predictive Interval         │
 │  └── Real-Time GeoJSON Delivery -> WebGIS Interactive Heatmaps & Pollution Alerts        │
 └──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Research Article & Study Scope

> **"A Smartphone-AI-WebGIS Platform for Real-Time Monitoring of Inorganic Nitrogen in Water"**  
> *Trung Nguyen Quoc, Minh-Vuong Phan, Phuc Nguyen Tran Thien, Vinh Truong Hoang, Thi-Mai-Thy Pham, Thanh-Long Do, Thai-Binh Tran\*, Manh-Huy Do, Le-Kim-Thuy Nguyen, Thi-Kim-Dung Hoang, and Thanh-Danh Nguyen\**  
> Corresponding Authors: *T.B. Tran* (`thaibinhtran@gmail.com`), *T.D. Nguyen* (`danh5463bd@yahoo.com`).

### Scope of this Repository
- **Computer Vision Pipeline**: Autonomous test strip segmentation using `YOLO11n-seg` and green-border color reference normalization in linear RGB space.
- **Deep Learning Modeling**: Multi-Task Heteroscedastic Neural Network with MLP2 task heads for simultaneous analyte classification ($\text{NH}_4^+$ vs. $\text{NO}_2^-$) and continuous concentration estimation with aleatoric uncertainty ($\mu, \sigma^2$).
- **Reproducibility Suite**: Cryptographic frozen dataset manifests ($3\text{K}$, $10\text{K}$, and $13\text{K}$ splits), master configuration (`paper_v1.yaml`), evaluation benchmarks (matched, mismatch ablation, domain transfer, source-resolved robustness), and automated unit tests.
- **REST API Microservice**: High-performance FastAPI server providing operational (`mode=app`) and strict reproduction (`mode=paper`) endpoints.

---

## 📁 Repository Structure

```text
AI---Chemistry/
├── README.md                           # Comprehensive scientific documentation
├── LICENSE                             # MIT Open-Source License
├── CITATION.cff                        # Machine-readable citation metadata
├── requirements.txt                    # Core Python dependencies
├── requirements-lock.txt               # Pinned tested environment
├── environment.yml                     # Conda environment specification
├── .github/workflows/ci.yml            # Automated CI build and test workflow
├── configs/
│   └── paper_v1.yaml                   # Master publication experiment configuration
├── data/
│   ├── README.md                       # Data partition documentation
│   └── manifests/                      # Frozen cryptographic partition splits
│       ├── 3k/{train,val,test}.csv     # 3K Field dataset manifests
│       ├── 10k/{train,val,test}.csv    # 10K Laboratory dataset manifests
│       └── 13k/{train,val,test}.csv    # 13K Multi-domain dataset manifests
├── ai_chemistry/
│   ├── __init__.py                     # Package interface
│   ├── modeling.py                     # Canonical MultiTaskHetero & MLP2 heads
│   ├── preprocessing.py                # Linearized green-border color normalizer
│   ├── data/
│   │   ├── __init__.py
│   │   └── loaders.py                  # Manifest loaders & ChemistryDataset
│   └── training/
│       ├── __init__.py
│       ├── train_classifier.py         # Publication training protocol
│       └── test_classifier.py          # Benchmark evaluation engine
├── api/
│   ├── main.py                         # FastAPI application entrypoint
│   ├── config.py                       # Model zoo registry & environment paths
│   ├── predictor.py                    # Inference engine with MC/analytic intervals
│   ├── roi.py                          # YOLO11n-seg autonomous ROI localization
│   ├── calibration.py                  # API color normalization wrapper
│   └── schemas.py                      # Pydantic request/response schemas
├── scripts/
│   ├── eval_matched.py                 # Matched preprocessing evaluation
│   ├── eval_preprocessing_mismatch.py  # Intentional mismatch ablation study
│   ├── eval_transfer.py                # Cross-dataset domain generalization (3K <-> 10K)
│   ├── eval_source_resolved_13k.py     # Source-resolved 13K robustness (C3 & C4)
│   ├── summarize_results.py            # Receipt aggregator & summary table generator
│   └── run_paper_matrix.py             # Full experimental matrix automation
├── tests/
│   ├── test_manifest_integrity.py      # Dataset partition verification
│   ├── test_model_architecture.py      # Network head & dimension verification
│   ├── test_greenborder_regression.py  # Color normalizer regression tests
│   ├── test_checkpoint_load.py         # Dynamic checkpoint loading tests
│   └── test_api_compatibility.py       # API backward-compatibility smoke tests
├── tools/
│   ├── validate_publication_data.py    # Manifest integrity validation tool
│   └── audit_all_checkpoints.py        # Comprehensive checkpoint auditor
└── weights/
    ├── README.md                       # Weight download links & instructions
    └── checkpoints_manifest.csv        # Audited parameters & SHA-256 checksums
```

---

## 📊 Dataset Partitioning & Frozen Manifests

All dataset partitions were **frozen prior to model evaluation** to guarantee zero data leakage across splits:

| Dataset ID | Target Domain | Total Samples | Train Split | Validation Split | Test Split | Concentration Range |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **3K** | Field / Natural Water Colorimetry | **2,841** | 2,064 | 476 | 301 | $0.00 - 5.00\text{ ppm}$ |
| **10K** | Laboratory Matrix & Spiked Solutions | **9,922** | 7,227 | 1,795 | 900 | $0.00 - 22.00\text{ ppm}$ |
| **13K** | Multi-Domain Combined Dataset | **10,491** | 7,495 | 2,095 | 901 | $0.00 - 22.00\text{ ppm}$ |

### Automated Manifest Validation
Verify row counts, label integrity, and zero cross-partition leakage:

```bash
python tools/validate_publication_data.py --manifests_dir data/manifests
```

Raw full-resolution colorimetric strip images are hosted on Google Drive:
- **Download Images**: [Raw Image Archive](https://drive.google.com/drive/folders/12lvPMyir46usQyKULd2RU_uE_CvoVKkS?usp=sharing) (Extract to `data/raw/<dataset>/`).

---

## 🧠 Neural Network Architectures & Vision Backbones

| Backbone Label | Exact `timm` Identifier | Family | Approx. Pretrained Parameters (M) |
| :--- | :--- | :--- | :---: |
| **MobileNetV3-Large** | `mobilenetv3_large_100.ra_in1k` | CNN | 5.5 |
| **EfficientNet-B0** | `efficientnet_b0.ra_in1k` | CNN | 5.3 |
| **DM-NFNet-F2** | `dm_nfnet_f2.dm_in1k` | CNN | 193.8 |
| **TF-EfficientNet-B3** | `tf_efficientnet_b3.ns_jft_in1k` | CNN | 12.2 |
| **ConvNeXt-Tiny** | `convnext_tiny.fb_in1k` | CNN | 28.6 |
| **Swin-Tiny** | `swin_tiny_patch4_window7_224.ms_in1k` | Transformer | 28.3 |

---

## ⚙️ S3.6. Model Training and Inference

### S3.6.1. Training Configuration

All backbone architectures were trained using a common multi-task optimization protocol to ensure fair comparison across network architectures. Unless otherwise stated, all models used $224 \times 224$ inputs and were trained for a maximum of 60 total epochs with AdamW (initial learning rate $2 \times 10^{-4}$; weight decay $1 \times 10^{-4}$). The backbone was frozen during the first five epochs and unfrozen thereafter for up to 55 additional epochs. Cosine LambdaLR scheduling was used, with the optimizer and scheduler reinitialized when the backbone was unfrozen. Mixed-precision training was enabled when CUDA was available, and gradient clipping was applied to improve numerical stability.

Training employed two analyte-specific heteroscedastic regression heads together with one classification head. Concentration targets were transformed using the `log1p` function, and the overall objective combined classification and regression losses with weighting factors of 1.0 and 2.0, respectively. Best-checkpoint selection used the validation score $(1 - \text{accuracy}) + 2 \times \text{MAE}$ with an early-stopping patience of 10 epochs. The reported model-comparison results correspond to seed 0. The complete training configuration is summarized in Table S6.

### Table S6. Training Parameters Used for All Backbone Architectures

| Parameter | Locked Specification |
| :--- | :--- |
| **Input and Normalization** | $224 \times 224$ RGB ROI; ImageNet mean and standard deviation |
| **Optimizer** | AdamW; initial learning rate = $2 \times 10^{-4}$; weight decay = $1 \times 10^{-4}$ |
| **Warm-up** | 5 epochs; backbone frozen |
| **Fine-tuning** | Up to 55 additional epochs; full network unfrozen |
| **Maximum Total Epochs** | 60 total (5 warm-up + up to 55 fine-tuning) |
| **Learning-rate Schedule** | Cosine LambdaLR; optimizer/scheduler reinitialized when the backbone is unfrozen |
| **Model Selection** | Validation score: $(1 - \text{accuracy}) + 2 \times \text{MAE}$; patience = 10 |
| **Classification Objective** | Cross-entropy with label smoothing = 0.05; weight = 1.0 |
| **Regression Objective** | Heteroscedastic Gaussian NLL in `log1p` space; weight = 2.0 |
| **Stability** | Mixed precision on CUDA; gradient clip = 1.0; no EMA |
| **Repetition** | Seed 0 only; one reported run per dataset/backbone/preprocessing configuration |

---

## 📈 Evaluation & Benchmark Reproduction

### 1. Checkpoint Evaluation on Test Split
```bash
python -m ai_chemistry.training.test_classifier \
  --ckpt_path weights/runs_multitask_10k/ConvNext_seed0_l2.0_none.pt \
  --dataset 10k \
  --split test \
  --calib auto \
  --output_json results/convnext10k_none_test.json \
  --predictions_csv results/convnext10k_none_test_predictions.csv
```

### 2. Matched Preprocessing Matrix
```bash
python scripts/eval_matched.py --dataset 10k --device cuda
```

### 3. Preprocessing Mismatch Ablation Study ($\text{None} \to \text{GreenBorder}$)
```bash
python scripts/eval_preprocessing_mismatch.py --dataset 10k --device cuda
```

### 4. Cross-Dataset Domain Generalization ($3\text{K} \leftrightarrow 10\text{K}$)
```bash
python scripts/eval_transfer.py --device cuda
```

### 5. Source-Resolved 13K Robustness Evaluation (C3 & C4 Components)
```bash
python scripts/eval_source_resolved_13k.py --device cuda
```

### 6. Aggregate Results Summary Table
```bash
python scripts/summarize_results.py --results_dir results
```

---

## 🚀 REST API Microservice Deployment

The FastAPI application powers real-time inference for smartphone clients and WebGIS integration.

### Quick Start
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger UI: `http://localhost:8000/docs`

### API Endpoints
- `GET /health` : Service health status.
- `GET /models` : List registered model keys, aliases, and runtime configurations.
- `POST /predict` : Multi-task classification and concentration prediction with optional 95% predictive interval.

### Prediction Modes
| Mode | Behavior | Description |
| :--- | :--- | :--- |
| `mode=app` *(Default)* | Flexible ROI (`roi_mode=auto`), green/center fallback, calib override | Backward-compatible operational path |
| `mode=paper` | Strict YOLO11n-seg ROI (10% pad), fallback disabled, matched calib enforced | Deterministic publication reproduction |
| `mode=diagnostic` | Strict YOLO11n-seg ROI, intentional calib mismatch allowed | Ablation & mismatch experiments |

### Python Client Example
```python
import requests

url = "http://localhost:8000/predict"
params = {
    "model": "convnext10k_none",
    "mode": "paper",
}
with open("sample_strip.jpg", "rb") as f:
    files = {"file": ("strip.jpg", f, "image/jpeg")}
    response = requests.post(url, params=params, files=files)

print(response.json())
```

---

## 🧪 Automated Unit Test Suite & Verification

Run the full validation suite locally:

```bash
# Manifest count, schema & zero leakage test
python -m unittest tests/test_manifest_integrity.py

# Model architecture & MLP2 task head test
python -m unittest tests/test_model_architecture.py

# Color normalizer numerical regression test
python -m unittest tests/test_greenborder_regression.py

# Dynamic checkpoint loading test
python -m unittest tests/test_checkpoint_load.py

# API backward compatibility & alias resolution test
python -m unittest tests/test_api_compatibility.py
```

Audit all pre-trained checkpoints:
```bash
python tools/audit_all_checkpoints.py --weights_dir weights
```

---

## 📖 Citation

If you utilize this codebase, model checkpoints, dataset manifests, or analytical workflows in your research, please cite:

```bibtex
@article{nguyen2026smartphone,
  title={A Smartphone-AI-WebGIS Platform for Real-Time Monitoring of Inorganic Nitrogen in Water},
  author={Nguyen Quoc, Trung and Phan, Minh-Vuong and Nguyen Tran Thien, Phuc and Truong Hoang, Vinh and Pham, Thi-Mai-Thy and Do, Thanh-Long and Tran, Thai-Binh and Do, Manh-Huy and Nguyen, Le-Kim-Thuy and Hoang, Thi-Kim-Dung and Nguyen, Thanh-Danh},
  journal={Analytical Chemistry / Environmental Science & Technology},
  year={2026},
  publisher={American Chemical Society / Springer Nature}
}
```

---

## 📄 License

This repository is distributed under the **MIT License**. See [LICENSE](file:///LICENSE) for details.