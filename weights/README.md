# Pre-Trained Model Weights & Checkpoints Manifest

This directory documents the pre-trained model checkpoints for all six vision backbones across the 3K, 10K, and 13K datasets under both uncalibrated (`none`) and green-border calibrated (`greenborder` / `green`) conditions.

---

## 1. Checkpoint Archive & Download Link

Due to Git LFS and repository size limits, binary weights are hosted on Google Drive:
- **Google Drive Weights Folder**: [Pre-trained Checkpoints](https://drive.google.com/drive/folders/12lvPMyir46usQyKULd2RU_uE_CvoVKkS?usp=sharing)

After downloading, place the files following this exact hierarchy:

```text
weights/
├── best.pt                          (YOLOv8 nano strip ROI detector)
├── checkpoints_manifest.csv         (Cryptographic SHA-256 manifest)
├── runs_multitask_3k/
│   ├── <Backbone>_seed0_l2.0_<calib>.pt
│   └── <Backbone>_seed0_l2.0_<calib>.meta.json
├── runs_multitask_10k/
│   ├── <Backbone>_seed0_l2.0_<calib>.pt
│   └── <Backbone>_seed0_l2.0_<calib>.meta.json
└── runs_multitask_13k/
    ├── <Backbone>_seed0_l2.0_<calib>.pt
    └── <Backbone>_seed0_l2.0_<calib>.meta.json
```

Where `<Backbone>` $\in$ `{ConvNext, EffB0, MNV3, NFNet, SwinT, TFB3}` and `<calib>` $\in$ `{none, green}`.

---

## 2. Checkpoint SHA-256 Integrity Verification

All 36 published checkpoints and the YOLO detector are registered in `checkpoints_manifest.csv`:

| Checkpoint Identifier | Backbone Architecture | Dataset | Training Calib | Loss Weight $\lambda_{\text{reg}}$ | SHA-256 Checksum (Prefix) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `best.pt` | YOLOv8n-Seg (ROI Detector) | - | - | - | `656601dfcc10...` |
| `ConvNext_seed0_l2.0_none.pt` | `convnext_tiny.fb_in1k` | 10K | None | 2.0 | `7e21310f241b...` |
| `ConvNext_seed0_l2.0_green.pt` | `convnext_tiny.fb_in1k` | 10K | GreenBorder | 2.0 | `e8b5c8684f6a...` |
| `MNV3_seed0_l2.0_none.pt` | `mobilenetv3_large_100.ra_in1k` | 13K | None | 2.0 | `b32a9c5b9f69...` |
| `MNV3_seed0_l2.0_green.pt` | `mobilenetv3_large_100.ra_in1k` | 13K | GreenBorder | 2.0 | `834dc3d59b70...` |
| `EffB0_seed0_l2.0_none.pt` | `efficientnet_b0.ra_in1k` | 10K | None | 2.0 | `6c0cb3d24466...` |
| `NFNet_seed0_l2.0_none.pt` | `dm_nfnet_f2.dm_in1k` | 10K | None | 2.0 | `5ebaba0df5c8...` |
| `SwinT_seed0_l2.0_none.pt` | `swin_tiny_patch4_window7_224.ms_in1k` | 10K | None | 2.0 | `95e71735146a...` |
| `TFB3_seed0_l2.0_none.pt` | `tf_efficientnet_b3.ns_jft_in1k` | 10K | None | 2.0 | `ea3559c9baed...` |

*(See `checkpoints_manifest.csv` for the complete 37-file cryptographic list).*

---

## 3. Strict Checkpoint Loading

All released checkpoints strictly load into the canonical `MultiTaskHetero` architecture:

```python
import torch
from ai_chemistry.modeling import MultiTaskHetero

model = MultiTaskHetero(
    timm_name="mobilenetv3_large_100.ra_in1k",
    num_classes=2,
    pretrained=False,
)
ckpt = torch.load("weights/runs_multitask_13k/MNV3_seed0_l2.0_none.pt", map_location="cpu")
model.load_state_dict(ckpt["state_dict"], strict=True)
print("Loaded successfully with strict=True!")
```