# Pre-Trained Model Weights & Checkpoints Manifest

This directory documents the pre-trained model checkpoints for all vision backbones across the 3K, 10K, and 13K datasets under both uncalibrated (`none`) and green-border calibrated (`greenborder` / `green`) conditions.

---

## 1. Checkpoint Archive & Download Link

Due to Git repository size limits, binary weights are hosted on Google Drive:
- **Google Drive Weights Folder**: [Pre-trained Checkpoints](https://drive.google.com/drive/folders/12lvPMyir46usQyKULd2RU_uE_CvoVKkS?usp=sharing)

After downloading, place the files following this exact hierarchy:

```text
weights/
├── best.pt                          (YOLO11n-seg autonomous strip ROI detector)
├── checkpoints_manifest.csv         (Cryptographic SHA-256 manifest & architecture audit)
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

---

## 2. Checkpoint Architecture & Checksum Audit

All 36 published multi-task checkpoints and the `YOLO11n-seg` detector have been independently audited with `tools/audit_all_checkpoints.py`:

| Checkpoint File | Backbone Identifier (`timm_name`) | Head Variant | Head Input Dim ($d$) | Total Parameters | SHA-256 (Prefix) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `best.pt` | `yolo11n-seg` | Seg Head | N/A | 2.60 M | `656601dfcc10...` |
| `runs_multitask_13k/MNV3_seed0_l2.0_none.pt` | `mobilenetv3_large_100.ra_in1k` | `mlp2` | 1280 | 6.20 M | `b32a9c5b9f69...` |
| `runs_multitask_13k/MNV3_seed0_l2.0_green.pt` | `mobilenetv3_large_100.ra_in1k` | `mlp2` | 1280 | 6.20 M | `834dc3d59b70...` |
| `runs_multitask_13k/EffB0_seed0_l2.0_none.pt` | `efficientnet_b0.ra_in1k` | `mlp2` | 1280 | 6.02 M | `e3d83d93c45d...` |
| `runs_multitask_13k/EffB0_seed0_l2.0_green.pt` | `efficientnet_b0.ra_in1k` | `mlp2` | 1280 | 6.02 M | `35a39a95ff2c...` |
| `runs_multitask_13k/SwinT_seed0_l2.0_none.pt` | `swin_tiny_patch4_window7_224.ms_in1k` | `mlp2` | 768 | 28.70 M | `8c196f45e730...` |
| `runs_multitask_13k/SwinT_seed0_l2.0_green.pt` | `swin_tiny_patch4_window7_224.ms_in1k` | `mlp2` | 768 | 28.70 M | `9573f1448bec...` |
| `runs_multitask_13k/TFB3_seed0_l2.0_none.pt` | `tf_efficientnet_b3.ns_jft_in1k` | `mlp2` | 1536 | 13.15 M | `b86561b24ea3...` |
| `runs_multitask_13k/TFB3_seed0_l2.0_green.pt` | `tf_efficientnet_b3.ns_jft_in1k` | `mlp2` | 1536 | 13.15 M | `06b35c16c464...` |
| `runs_multitask_13k/ConvNext_seed0_l2.0_none.pt` | `convnext_tiny.fb_in1k` | `mlp2` | 768 | 29.00 M | `9032c3fa2f87...` |
| `runs_multitask_13k/ConvNext_seed0_l2.0_green.pt` | `convnext_tiny.fb_in1k` | `mlp2` | 768 | 29.00 M | `b3c529ba5920...` |
| `runs_multitask_13k/NFNet_seed0_l2.0_none.pt` | `dm_nfnet_f2.dm_in1k` | `mlp2` | 3072 | 195.43 M | `6417b98255c0...` |
| `runs_multitask_13k/NFNet_seed0_l2.0_green.pt` | `dm_nfnet_f2.dm_in1k` | `mlp2` | 3072 | 195.43 M | `e79438909d62...` |

*(See `checkpoints_manifest.csv` for the full audited list of all 37 checkpoint files).*

---

## 3. Automated Checkpoint Audit Tool

To audit local checkpoints and re-verify strict loadability:

```bash
python tools/audit_all_checkpoints.py --weights_dir weights --output_csv weights/checkpoints_manifest.csv
```