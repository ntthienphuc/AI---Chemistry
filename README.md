# AI Chemistry Project Guide

This document is a complete start-to-finish guide for setting up the project, running the API, consuming the API, and running the training workflow.

## 1) Environment Setup (From Zero)

### 1.1 Prerequisites
- Python 3.10+ (recommended: 3.10 or 3.11)
- `pip`
- Windows PowerShell (examples below use PowerShell syntax)

### 1.2 Create and activate a virtual environment
From the project root (`D:\Project\AI - Chemistry`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run once:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 1.3 Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 1.4 Download model weights (Google Drive)
Download all files/folders from this link:

- https://drive.google.com/drive/folders/1EH5YPtrfB2QARHwZvKr2Q4DKsUS1PmQ3?usp=sharing

Then place `weights/` folder in the main root next to `ai_chemistry` and `api` folder.

Important:
- Keep filenames unchanged.
- Keep the subfolder structure unchanged.
- `best.pt` must be directly inside `weights/`.

Expected structure:

```text
weights/
  best.pt
  paper_lab10k/
    ConvNext_seed0_l2.0.pt
    ConvNext_seed0_l2.0.meta.json
    EffB0_seed0_l2.0.pt
    EffB0_seed0_l2.0.meta.json
    NFNet_seed0_l2.0.pt
    NFNet_seed0_l2.0.meta.json
    TFB3_seed0_l2.0.pt
    TFB3_seed0_l2.0.meta.json
  paper_field3k/
    ConvNext_field3k_seed0_l2.0_bs24.pt
    ConvNext_field3k_seed0_l2.0_bs24.meta.json
    EffB0_field3k_seed0_l2.0_bs24.pt
    EffB0_field3k_seed0_l2.0_bs24.meta.json
    DMNFNet_field3k_seed0_l2.0_bs24.pt
    DMNFNet_field3k_seed0_l2.0_bs24.meta.json
    TFB3_field3k_seed0_l2.0_bs24.pt
    TFB3_field3k_seed0_l2.0_bs24.meta.json
```

Quick check:

```powershell
Get-ChildItem -Recurse .\weights
```

## 2) Run the API

### 2.1 Start API server
From project root with venv activated:

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2.2 Available endpoints
- `GET /health`
- `GET /models`
- `POST /predict`

### 2.3 Optional runtime environment variables
You can override config values in `api/config.py`:

```powershell
$env:AI_CHEM_ROOT = "D:\Project\AI - Chemistry"
$env:AI_CHEM_YOLO = "D:\Project\AI - Chemistry\weights\best.pt"
$env:AI_CHEM_DEVICE = "cuda"      # or "cpu"
$env:AI_CHEM_CALIB = "greenborder" # or "none"
```

After setting env vars, restart Uvicorn.

## 3) Use the API (Multiple Methods + Input/Output Format)

### 3.1 Endpoint contract

#### `GET /health`
Purpose: health check.

Example response:

```json
{
  "status": "ok"
}
```

#### `GET /models`
Purpose: list valid model keys for prediction.

Example response:

```json
{
  "available_models": [
    "convnext10k",
    "convnext3k",
    "effb010k",
    "effb03k",
    "nfnet10k",
    "nfnet3k",
    "tfb310k",
    "tfb33k"
  ],
  "yolo_weights": "D:\\Project\\AI - Chemistry\\weights\\best.pt",
  "device": "cuda",
  "calib_mode": "greenborder"
}
```

#### `POST /predict`
Purpose: predict chemical class and concentration from one uploaded image.

Request format:
- Query params:
  - `model` (required): one value from `/models`
  - `roi_mode` (optional, default `auto`): `auto | yolo | green | center`
  - `debug` (optional, default `false`): return raw internals when `true`
- Multipart form:
  - `file` (required): image file (`.jpg`, `.jpeg`, `.png`)

Response format (`200 OK`):

```json
{
  "model": "convnext10k",
  "predicted_chemical": "NO2",
  "chemical_confidence": 0.9734,
  "concentration": {
    "ppm": 1.24,
    "ppm_ci95": [1.01, 1.48],
    "ppm_sigma": 0.12,
    "method": "heteroscedastic_gaussian"
  },
  "calib_mode": "greenborder",
  "roi": {
    "source": "mask",
    "bbox_xyxy": [120, 85, 650, 920],
    "padding": 0.1,
    "imgsz": 640
  },
  "raw": null
}
```

Field meaning:
- `predicted_chemical`: predicted analyte label (`NH4` or `NO2`).
- `concentration.ppm`: predicted concentration value in ppm.
- `chemical_confidence`: model confidence for the predicted class.
- `concentration.ppm_ci95`: optional 95% confidence interval of ppm.
- `concentration.ppm_sigma`: optional uncertainty spread (standard deviation) in ppm.
- `roi`: ROI extraction metadata (source and bounding box used before classification/regression).
- `raw`: only populated when `debug=true`, includes internal values for troubleshooting.

If you only need the final result (predicted chemical name + ppm), focus on:
- `predicted_chemical`
- `concentration.ppm`

Error behavior:
- `400`: invalid `model` or invalid image decode
- `422`: ROI extraction failure
- `500`: model inference failure

### 3.2 Method A: Swagger UI (browser)
1. Open `http://127.0.0.1:8000/docs`
2. Try `GET /models` first.
3. Open `POST /predict`.
4. Fill query `model` with a valid key from `/models`.
5. Upload `file`.
6. Execute and inspect JSON response.

### 3.3 Method B: cURL (terminal)

```bash
curl -X POST "http://127.0.0.1:8000/predict?model=convnext10k&roi_mode=auto&debug=false" \
  -F "file=@Spike_test_AI/NH4/Drinking water/sample.jpg"
```

### 3.4 Method C: PowerShell (`Invoke-RestMethod`)

```powershell
$url = "http://127.0.0.1:8000/predict?model=convnext10k&roi_mode=auto&debug=true"
$form = @{ file = Get-Item "D:\Project\AI - Chemistry\Spike_test_AI\NH4\Drinking water\sample.jpg" }
Invoke-RestMethod -Uri $url -Method Post -Form $form
```

### 3.5 Method D: Python (`requests`)

```python
import requests

url = "http://127.0.0.1:8000/predict"
params = {
    "model": "convnext10k",
    "roi_mode": "auto",
    "debug": False,
}

with open(r"Spike_test_AI/NH4/Drinking water/sample.jpg", "rb") as f:
    resp = requests.post(url, params=params, files={"file": f}, timeout=60)

resp.raise_for_status()
print(resp.json())
```

### 3.6 Method E: Postman
- Method: `POST`
- URL: `http://127.0.0.1:8000/predict?model=convnext10k&roi_mode=auto&debug=false`
- Body -> `form-data`
  - key `file`, type `File`, value = your image
- Send and inspect JSON response

## 4) General Training Workflow

This section is intentionally general and practical. The exact experiment settings can vary by dataset size and device.

### 4.1 Prepare raw data
Recommended raw structure:

```text
<raw_root>/
  NH4/
    <device>/
      <round_or_session>/
        *.jpg|*.jpeg|*.png
  NO2/
    <device>/
      <round_or_session>/
        *.jpg|*.jpeg|*.png
```

### 4.2 Split dataset into train/val/test
Preferred (group-aware, avoids leakage across near-duplicate captures):

```powershell
python ai_chemistry/data/split_dataset_grouped.py `
  --source-root "D:\path\to\raw_data" `
  --output-root "D:\path\to\data_split" `
  --train 0.7 --val 0.15 --test 0.15 `
  --seed 150
```

Optional: add `--manifest-json` for reproducible splits, or `--holdout-devices` for cross-device evaluation.

### 4.3 Crop ROI using YOLO

```powershell
python ai_chemistry/data/crop_roi.py `
  --yolo-weights "weights\best.pt" `
  --input-root "D:\path\to\data_split" `
  --output-root "D:\path\to\data_roi" `
  --padding 0.10 --conf 0.25 --imgsz 640
```

### 4.4 Generate labels CSV

```powershell
python ai_chemistry/data/generate_labels.py `
  --images-root "D:\path\to\data_roi" `
  --output-csv "D:\path\to\data_clsreg\labels.csv"
```

Expected columns in CSV:
- `path`
- `chemical` (`NH4` or `NO2`)
- `ppm`
- `device`
- `datetime`
- `split`

Optional cleaning step (drop unreadable files):

```powershell
python ai_chemistry/data/clean_labels.py `
  --root-dir "D:\path\to\data_clsreg" `
  --labels-csv "D:\path\to\data_clsreg\labels.csv" `
  --output-csv "D:\path\to\data_clsreg\labels.clean.csv"
```

### 4.5 Train a model

```powershell
python ai_chemistry/training/train_classifier.py `
  --root_dir "D:\path\to\data_clsreg" `
  --labels_csv "D:\path\to\data_clsreg\labels.csv" `
  --timm_name "convnext_base.fb_in1k" `
  --image_size 224 `
  --batch_size 24 `
  --epochs 60 `
  --warmup_epochs 5 `
  --save_path "weights\custom\convnext_custom.pt" `
  --device cuda `
  --calib greenborder
```

Training output:
- model checkpoint: `*.pt`
- metadata file: matching `*.meta.json`

### 4.6 Evaluate a trained checkpoint

```powershell
python ai_chemistry/training/test_classifier.py `
  --root_dir "D:\path\to\data_clsreg" `
  --labels_csv "D:\path\to\data_clsreg\labels.csv" `
  --ckpt_path "weights\custom\convnext_custom.pt" `
  --split test `
  --device cuda
```

Optional: append metrics to CSV with `--csv_path`.

### 4.7 Practical training notes
- Start with one backbone (`convnext_base.fb_in1k` or `efficientnetv2_s`) before large sweeps.
- Keep calibration mode consistent between training and serving (`greenborder` vs `none`).
- Always validate on `val` before comparing `test` performance.
- Save split manifests for reproducibility when comparing experiments.
- Keep model and meta file together (`.pt` + `.meta.json`) for deployment.

---

If you need, this guide can be extended with a reproducible experiment template (folder conventions, naming, and run logs).
