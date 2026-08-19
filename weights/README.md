# Pre-Trained Model Weights & Directory Structure

This directory documents the pre-trained model checkpoints for all vision backbones across the 3K, 10K, and 13K datasets under both uncalibrated (`none`) and green-border calibrated (`greenborder` / `green`) conditions.

---

## 1. Checkpoint Archive & Download Link

Due to Git repository size limits, binary weights are hosted on Google Drive:
- **Google Drive Weights Folder**: [Pre-trained Checkpoints](https://drive.google.com/drive/folders/12lvPMyir46usQyKULd2RU_uE_CvoVKkS?usp=sharing)

After downloading, place the files following this exact hierarchy:

```text
weights/
├── best.pt                          (YOLO11n-seg autonomous strip ROI detector)
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

## 2. Checkpoint Loading Example

All published multi-task checkpoints load directly into `MultiTaskHeteroFlexible`:

```python
import torch
from ai_chemistry.modeling import MultiTaskHeteroFlexible, build_meta_from_ckpt, strip_state_dict_prefix, infer_head_variant, infer_reg_out_dim, infer_head_in_features

ckpt_path = "weights/runs_multitask_13k/MNV3_seed0_l2.0_none.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state = strip_state_dict_prefix(ckpt.get("state_dict", ckpt))
meta = build_meta_from_ckpt(ckpt, ckpt_path=ckpt_path)

model = MultiTaskHeteroFlexible(
    timm_name=meta.timm_name,
    num_classes=meta.num_classes,
    pretrained=False,
    drop=meta.drop,
    drop_path=meta.drop_path,
    image_size=meta.image_size,
    head_variant=infer_head_variant(state),
    reg_out_dim=infer_reg_out_dim(state),
    expected_feat_dim=infer_head_in_features(state),
)
model.load_state_dict(state, strict=True)
model.eval()
print(f"Loaded {meta.timm_name} successfully with strict=True!")
```