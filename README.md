## AI Chemistry Project

End-to-end toolkit for analysing NH4/NO2 test strips: splitting and cropping raw captures, training a dual-task classifier/regressor, running batched CLI inference, and deploying a FastAPI endpoint. The repository bundles every stage needed to reproduce the production pipeline that powers the reference weights inside `weights/`.

---

### Repository Layout
- `ai_chemistry/data`: dataset utilities (`split_dataset`, YOLO ROI cropping, label generation)
- `ai_chemistry/training`: training entry point for the two-head heteroscedastic backbone
- `ai_chemistry/inference`: reusable inference pipeline plus optional NO2 calibration helper
- `ai_chemistry/tools`: maintenance scripts (e.g. repacking checkpoints)
- `weights/`: YOLO detector (`best.pt`) and four classifier checkpoints with their meta JSON
- `Spike_test_AI/`: curated validation set used for quick regression tests
- `api/main.py`: FastAPI surface over the inference pipeline

---

### Methodology
- **ROI detection**: Ultralytics YOLOv11-seg checkpoint (`weights/best.pt`) localises the strip mask. The largest mask is cropped with configurable padding (default 10%); detection boxes backstop missing masks.
- **Color normalisation**: A green-border estimator rescales RGB intensities to reduce device lighting variability. The outer ring is converted to HSV, green pixels are filtered, and mean colour is used for per-channel normalisation.
- **Feature extractor**: A timm backbone (EfficientNet, NFNet, ConvNeXt, TF-EfficientNet-B3) is fine-tuned without classifier head (`num_classes=0`) to produce dense features.
- **Multi-task heads**:  
  - `head_cls`: 2-way softmax for NH4 vs NO2.  
  - `head_reg_NH4`, `head_reg_NO2`: each outputs `(mu, log_var)` so the model learns both the ppm value and its aleatoric uncertainty (heteroscedastic regression). During inference the head matching the predicted class is used.
- **Optimisation**: Cross-entropy (optionally focal) for the classifier, Gaussian NLL for regression, Exponential Moving Average (EMA) on weights, Cosine LR with warmup, gradient clipping, and early stopping on a smoothed validation score.
- **Test-time augmentation (TTA)**: Optional horizontal/vertical flips per image with median aggregation for ppm, weighted vote for class.

---

### Pipeline Flow
1. **Image ingestion**: raw capture is read from disk or HTTP upload.
2. **YOLO detection**: ROI crop generated from segmentation mask (fallback to largest detection box).
3. **Pre-process ROI**: green-border normalisation + resizing + ImageNet standardisation.
4. **Model forward pass**: classifier predicts chemical, matching regression head returns ppm (inverse transform of log1p/minmax scale).
5. **Post-process**: combine class, ppm, confidence, and estimated sigma; optionally apply NO2 linear calibration.

The same flow powers both the CLI (`python -m ai_chemistry.inference.pipeline`) and the FastAPI service (`uvicorn api.main:app --reload`).

---

### Available Weights
| File | Backbone (timm) | Params (approx) | Notes |
| --- | --- | --- | --- |
| `best.pt` | YOLOv11-seg | 6M | Detect strip ROI |
| `twoheads_hetero_efficientnet_b0.ra_in1k.pt` | EfficientNet-B0 | 25M | Lightweight baseline |
| `twoheads_hetero_nfnet_f0.dm_in1k.pt` | NFNet-F0 | 292M | Highest accuracy on Spike set |
| `twoheads_hetero_convnext_base.fb_in1k.pt` | ConvNeXt-Base | 357M | Strong balance between size and robustness |
| `twoheads_hetero_tf_efficientnet_b3.ns_jft_in1k.pt` | TF-EfficientNet-B3 | 53M | Wider receptive field |

Each `.pt` is paired with a `.meta.json` storing class order, ppm scaling, and image size, enabling the inference module to rebuild the architecture automatically.

---

### Environment Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # use `source .venv/bin/activate` on Linux/macOS
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Dataset Preparation Workflow
1. **Split raw captures**
   ```bash
   python -m ai_chemistry.data.split_dataset \
     --source-root "path/to/NH4_NO2_sorted round-2" \
     --output-root data/raw_split \
     --seed 150
   ```
2. **Crop YOLO ROI**
   ```bash
   python -m ai_chemistry.data.crop_roi \
     --yolo-weights weights/best.pt \
     --input-root data/raw_split \
     --output-root data/images
   ```
3. **Generate labels**
   ```bash
   python -m ai_chemistry.data.generate_labels \
     --images-root data/images \
     --output-csv data/labels.csv
   ```

---

### Training
```bash
python -m ai_chemistry.training.train_classifier \
  --root_dir data/images \
  --labels_csv data/labels.csv \
  --timm_name efficientnetv2_s \
  --epochs 60 \
  --save_path weights/custom_twoheads.pt
```
Use `--help` for options covering learning rate, warmup epochs, EMA decay, gradient clipping, loss settings (focal, label smoothing), ppm scaling, and random seed.

---

### Inference Options
- **CLI batch mode**
  ```bash
  python -m ai_chemistry.inference.pipeline \
    --input-path Spike_test_AI \
    --yolo-weights weights/best.pt \
    --classifier-weights weights/twoheads_hetero_nfnet_f0.dm_in1k.pt \
    --output-csv outputs/predictions.csv
  ```
- **NO2 calibration (optional)**
  ```bash
  python -m ai_chemistry.inference.calibrate_no2 \
    --csv-path outputs/predictions_with_gt.csv \
    --output-csv outputs/predictions_calibrated.csv \
    --save-model
  ```
- **FastAPI service**
  ```bash
  uvicorn api.main:app --reload --port 8000
  ```
  - `GET /` -> health + available models  
  - `GET /models` -> enumerate `.pt` checkpoints discovered in `weights/`  
  - `POST /predict` -> multipart form (`file`, optional `model_name`)

---

### Spike_test_AI Evaluation
Metrics were captured by running the legacy `5_2_4.py` script with consistent settings (`--tta 2 --pad 0.10 --ring_frac 0.10 --min_green_pixels 500 --no-apply_calibration --clip_zero`). The same backbones can be loaded through the new pipeline.

| Classifier | Overall Acc | F1 | MAE (ppm) | MAPE | NH4 Acc | NO2 Acc | Drinking Water Acc | RO Water Acc | Tap Water Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientNet-B0 (RA) | 0.7789 | 0.7589 | 0.2424 | 0.4361 | 1.0000 | 0.4878 | 0.8333 | 0.7143 | 0.8000 |
| NFNet-F0 (DM) | **0.8842** | **0.8806** | 0.2363 | 0.4315 | 1.0000 | 0.7317 | **0.9333** | **0.8571** | **0.8667** |
| ConvNeXt-Base (FB) | 0.8842 | 0.8806 | **0.2333** | **0.4280** | 1.0000 | 0.7317 | 0.9000 | 0.8857 | 0.8667 |
| TF-EfficientNet-B3 (NS) | 0.8211 | 0.8098 | 0.2442 | 0.4571 | 1.0000 | 0.5854 | 0.8333 | 0.7714 | 0.8667 |

Key observations:
- All backbones achieve perfect classification on NH4 samples; NO2 dominates residual errors.
- ConvNeXt-Base edges NFNet-F0 on regression accuracy (MAE/MAPE), while NFNet-F0 ties on classification and is steadier across water types.
- The largest ppm misses arise from NO2 samples in low-light or reflective conditions (e.g. Samsung S20 drinking water capture predicted 1.99-2.67 ppm vs 1 ppm GT).

Representative hard cases (predicted ppm vs ground-truth):
- `Spike_test_AI/NO2/Drinking water/ozI0H_..._1ppm.jpg` -> predicted 1.98-2.67 ppm (strong glare).
- `Spike_test_AI/NO2/RO water/FPAOH_..._1ppm.jpg` -> EfficientNet-B0 and TF-EfficientNet-B3 misclassified as NH4 at ~0.04 ppm.
- `Spike_test_AI/NO2/RO water/3RTuX_..._1ppm.jpg` -> low ppm estimates across models (0.41-0.44 ppm) except NFNet-F0 due to darker ROI.

---

### Quick Self-Test
```bash
python -m ai_chemistry.inference.pipeline \
  --input-path Spike_test_AI/NH4/Drinking water/2K5qd_..._0.15ppm.jpg \
  --yolo-weights weights/best.pt \
  --classifier-weights weights/twoheads_hetero_efficientnet_b0.ra_in1k.pt
```
Expected output resembles:
```
{'chemical': 'NH4', 'ppm': 0.0432, 'confidence': 0.9893, 'ppm_scaled': 0.0423, 'sigma': 0.0498}
```

Use `--classifier-weights` to swap to any of the four checkpoints above or to a newly trained model.
