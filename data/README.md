# AI-Chemistry Dataset Manifests & Data Protocol

This directory provides the frozen, deterministic dataset manifests and partition records supporting the publication:
**"A Smartphone-AI-WebGIS Platform for Real-Time Monitoring of Inorganic Nitrogen in Water"**.

---

## 1. Directory Structure

```text
data/
├── README.md
└── manifests/
    ├── 3k/
    │   ├── train.csv   (2,064 samples)
    │   ├── val.csv     (476 samples)
    │   └── test.csv    (301 samples)
    ├── 10k/
    │   ├── train.csv   (7,227 samples)
    │   ├── val.csv     (1,795 samples)
    │   └── test.csv    (900 samples)
    └── 13k/
        ├── train.csv   (7,495 samples)
        ├── val.csv     (2,095 samples)
        └── test.csv    (901 samples)
```

---

## 2. Dataset Overview & Split Summary

| Dataset ID | Target Domain | Image Count (Total) | Train Split | Validation Split | Test Split | Primary Analyte Range |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **3K** | Field / Natural Water Smartphone Colorimetry | **2,841** | 2,064 | 476 | 301 | $0.0 - 5.0\text{ ppm}$ ($\text{NH}_4^+$, $\text{NO}_2^-$) |
| **10K** | Laboratory Smartphone Colorimetry | **9,922** | 7,227 | 1,795 | 900 | $0.0 - 5.0\text{ ppm}$ ($\text{NH}_4^+$, $\text{NO}_2^-$) |
| **13K** | Combined Multi-Domain Robustness Set | **10,491** | 7,495 | 2,095 | 901 | Multi-device, Multi-condition |

> **Note on Split Integrity**:
> - All splits were fixed prior to model training under seed 0 and archived with cryptographic SHA-256 validation.
> - **Zero Data Leakage**: There is zero sample overlap between `train.csv`, `val.csv`, and `test.csv` across all subsets.
> - Split membership in the 13K combined set strictly inherits from its constituent source partitions.

---

## 3. CSV Manifest Schema

Each manifest CSV contains the following structured fields:

| Column | Data Type | Description | Example Values |
| :--- | :--- | :--- | :--- |
| `path` | `string` | Relative path to image within the raw image repository | `raw/10k/NH4/IMG_1024.jpg` |
| `chemical` | `string` | Inorganic nitrogen analyte class (`NH4` or `NO2`) | `NH4`, `NO2` |
| `ppm` | `float` | Analytical reference concentration in $\text{mg/L}$ ($\text{ppm}$) | `0.0`, `0.25`, `1.5`, `3.0` |
| `device` | `string` | Smartphone capture hardware model (if recorded) | `iPhone_13`, `Samsung_S21` |
| `datetime` | `string` | ISO / UTC timestamp of capture (if recorded) | `2025-10-24 14:30:00` |
| `split` | `string` | Designated experimental partition | `train`, `val`, `test` |

---

## 4. Downloading Raw Image Archives

Due to GitHub file size limits, raw full-resolution colorimetric strip images are hosted on Google Drive:
- **Image Repository URL**: [Google Drive Folder](https://drive.google.com/drive/folders/12lvPMyir46usQyKULd2RU_uE_CvoVKkS?usp=sharing)

Place the extracted images under `data/raw/<dataset>/` or specify the custom path using the `--images_root` argument during evaluation and training.

---

## 5. Automated Manifest Integrity Validation

To verify the integrity and exact counts of the publication manifests, run:

```bash
python tools/validate_publication_data.py --manifests_dir data/manifests
```