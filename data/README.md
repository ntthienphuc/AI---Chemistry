# Dataset Partitions & Manifest Integrity Documentation

This directory contains the frozen, cryptographic dataset manifests used across all experiments in the study:

> **"A Smartphone-AI-WebGIS Platform for Real-Time Monitoring of Inorganic Nitrogen in Water"**

---

## 1. Dataset Partition Summary

All dataset partitions were **frozen prior to model training and evaluation** to guarantee zero data leakage across splits.

| Dataset Identifier | Target Domain | Total Samples | Train Split | Validation Split | Test Split | Concentration Span |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **3K** | Field / Natural Water Colorimetry | **2,841** | 2,064 | 476 | 301 | $0.00 - 5.00\text{ ppm}$ |
| **10K** | Laboratory Matrix & Spiked Solutions | **9,922** | 7,227 | 1,795 | 900 | $0.00 - 22.00\text{ ppm}$ |
| **13K** | Multi-Domain Combined Dataset | **10,491** | 7,495 | 2,095 | 901 | $0.00 - 22.00\text{ ppm}$ |

---

## 2. Manifest Schema

All manifest files (`train.csv`, `val.csv`, `test.csv`) follow the standardized format:

```csv
path,chemical,ppm,device,datetime,split
3k/NH4/sample_001.jpg,NH4,0.50,iPhone12,2026-03-15 10:30:00,train
...
```

### Columns:
- `path`: Relative path to raw image from dataset root.
- `chemical`: Target analyte class (`NH4` for Ammonium $\text{NH}_4^+$, `NO2` for Nitrite $\text{NO}_2^-$).
- `ppm`: Analytical ground truth concentration in $\text{mg/L}$ ($\text{ppm}$).
- `device`: Smartphone model used for acquisition.
- `datetime`: Timestamp of capture.
- `split`: Partition assignment (`train`, `val`, `test`).

---

## 3. Automated Manifest Validation

To verify the row counts, column schemas, and ensure **zero cross-split sample leakage**:

```bash
python tools/validate_publication_data.py --manifests_dir data/manifests
```