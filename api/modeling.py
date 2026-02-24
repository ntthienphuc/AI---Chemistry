from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import timm


@dataclass
class Meta:
    timm_name: str
    num_classes: int
    image_size: int
    ppm_scale: str = "log1p"
    ppm_min: Optional[float] = None
    ppm_max: Optional[float] = None
    classes: Tuple[str, ...] = ("NH4", "NO2")
    drop: float = 0.2
    drop_path: float = 0.1


def strip_state_dict_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # handle "module." or "model." etc.
    out = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith("model."):
            k = k[len("model.") :]
        out[k] = v
    return out


def infer_head_variant(state: Dict[str, torch.Tensor]) -> str:
    # checkpoint style:
    # - "mlp2": head_cls.0.weight and head_cls.3.weight exist
    # - "linear": head_cls.1.weight exists (Dropout at 0, Linear at 1)
    keys = state.keys()
    if any(k.startswith("head_cls.3.") for k in keys) or any(k.startswith("head_reg_NH4.3.") for k in keys):
        return "mlp2"
    if any(k.startswith("head_cls.1.") for k in keys):
        return "linear"
    # default to mlp2 (safer for your newest weights)
    return "mlp2"


def infer_reg_out_dim(state: Dict[str, torch.Tensor]) -> int:
    # look at last linear weight in reg head
    for cand in ("head_reg_NH4.3.weight", "head_reg_NH4.1.weight", "head_reg_NH4.weight"):
        if cand in state:
            return int(state[cand].shape[0])
    return 2  # fallback


class MultiTaskHeteroFlexible(nn.Module):
    def __init__(
        self,
        timm_name: str,
        num_classes: int = 2,
        pretrained: bool = True,
        drop: float = 0.2,
        drop_path: float = 0.1,
        head_variant: str = "mlp2",
        reg_out_dim: int = 2,
    ):
        super().__init__()
        self.timm_name = timm_name
        self.num_classes = int(num_classes)
        self.head_variant = head_variant
        self.reg_out_dim = int(reg_out_dim)

        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop,
            drop_path_rate=drop_path,
        )
        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None and hasattr(self.backbone, "feature_info"):
            feat_dim = self.backbone.feature_info[-1]["num_chs"]
        if feat_dim is None:
            raise RuntimeError(f"Không tìm được feat_dim cho timm model: {timm_name}")

        if head_variant == "linear":
            self.head_cls = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, self.num_classes))
            self.head_reg_NH4 = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, self.reg_out_dim))
            self.head_reg_NO2 = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, self.reg_out_dim))
        else:
            # mlp2: Linear -> ReLU -> Dropout -> Linear
            self.head_cls = nn.Sequential(
                nn.Linear(feat_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, self.num_classes),
            )
            self.head_reg_NH4 = nn.Sequential(
                nn.Linear(feat_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, self.reg_out_dim),
            )
            self.head_reg_NO2 = nn.Sequential(
                nn.Linear(feat_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, self.reg_out_dim),
            )

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        cls_out = self.head_cls(feats)
        reg_NH4 = self.head_reg_NH4(feats)
        reg_NO2 = self.head_reg_NO2(feats)
        return cls_out, reg_NH4, reg_NO2, feats


def build_meta_from_ckpt(ckpt: Dict) -> Meta:
    # ckpt may store these keys directly (as in your saved .pt)
    timm_name = ckpt.get("timm_name", "convnext_base.fb_in1k")
    num_classes = int(ckpt.get("num_classes", 2))
    image_size = int(ckpt.get("image_size", 224))
    drop = float(ckpt.get("drop", 0.2))
    drop_path = float(ckpt.get("drop_path", 0.1))
    ppm_scale = str(ckpt.get("ppm_scale", "log1p"))
    ppm_min = ckpt.get("ppm_min", None)
    ppm_max = ckpt.get("ppm_max", None)
    classes = tuple(ckpt.get("classes", ["NH4", "NO2"]))
    return Meta(
        timm_name=timm_name,
        num_classes=num_classes,
        image_size=image_size,
        ppm_scale=ppm_scale,
        ppm_min=ppm_min,
        ppm_max=ppm_max,
        classes=classes,
        drop=drop,
        drop_path=drop_path,
    )
