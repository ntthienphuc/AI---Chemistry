# -*- coding: utf-8 -*-
"""
Multi-Task Heteroscedastic Neural Network Architecture for Inorganic Nitrogen Monitoring.

Defines:
- Canonical MultiTaskHetero: Shared deep backbone with 3 MLP2 task heads
  (Classification: NH4 vs. NO2, Regression NH4: mu + log_var, Regression NO2: mu + log_var).
- Robust feature dimension inference for timm backbones.
- Flexible model loader for legacy checkpoints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)

PAPER_BACKBONES: Dict[str, str] = {
    "mnv3": "mobilenetv3_large_100.ra_in1k",
    "effb0": "efficientnet_b0.ra_in1k",
    "nfnet": "dm_nfnet_f2.dm_in1k",
    "tfb3": "tf_efficientnet_b3.ns_jft_in1k",
    "convnext": "convnext_tiny.fb_in1k",
    "swint": "swin_tiny_patch4_window7_224.ms_in1k",
}


@dataclass
class ModelMeta:
    timm_name: str
    num_classes: int = 2
    image_size: int = 224
    ppm_scale: str = "log1p"
    ppm_min: Optional[float] = None
    ppm_max: Optional[float] = None
    classes: Tuple[str, ...] = ("NH4", "NO2")
    drop: float = 0.2
    drop_path: float = 0.1
    head_variant: str = "mlp2"
    calib_mode_train: str = "none"
    loss_weight_cls: float = 1.0
    loss_weight_reg: float = 2.0
    seed: int = 0


def strip_state_dict_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip common distributed wrapper prefixes ('module.', 'model.') from keys."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith("model."):
            k = k[len("model.") :]
        out[k] = v
    return out


def infer_head_variant(state: Dict[str, torch.Tensor]) -> str:
    """
    Detect head architecture from state dict:
    - 'mlp2': Linear -> ReLU -> Dropout -> Linear (keys like head_cls.0.weight and head_cls.3.weight)
    - 'linear': Dropout -> Linear (keys like head_cls.1.weight or head_cls.weight)
    """
    keys = list(state.keys())
    if any(k.startswith("head_cls.3.") for k in keys) or any(k.startswith("head_reg_NH4.3.") for k in keys):
        return "mlp2"
    if any(k.startswith("head_cls.1.") for k in keys):
        return "linear"
    return "mlp2"


def infer_head_in_features(state: Dict[str, torch.Tensor]) -> Optional[int]:
    """Infer feature dimension expected by the classification or regression head weights."""
    for k in ["head_cls.0.weight", "head_cls.1.weight", "head_cls.weight", "head_reg_NH4.0.weight", "head_reg_NH4.1.weight"]:
        if k in state and hasattr(state[k], "shape") and len(state[k].shape) >= 2:
            return int(state[k].shape[1])
    return None


def infer_reg_out_dim(state: Dict[str, torch.Tensor]) -> int:
    """Infer regression output dimension: 2 (heteroscedastic: mu, logvar) or 1 (homoscedastic: mu)."""
    for k in ["head_reg_NH4.3.weight", "head_reg_NH4.1.weight", "head_reg_NH4.weight"]:
        if k in state and hasattr(state[k], "shape") and len(state[k].shape) >= 1:
            return int(state[k].shape[0])
    return 2


def build_meta_from_ckpt(ckpt: Dict[str, Any], ckpt_path: Optional[Union[str, Path]] = None) -> ModelMeta:
    """Extract metadata dictionary or assign defaults from a loaded checkpoint with provenance fallbacks."""
    timm_name = ckpt.get("timm_name", "convnext_tiny.fb_in1k")
    num_classes = int(ckpt.get("num_classes", 2))
    image_size = int(ckpt.get("image_size", 224))
    ppm_scale = str(ckpt.get("ppm_scale", "log1p"))
    ppm_min = ckpt.get("ppm_min", None)
    ppm_max = ckpt.get("ppm_max", None)
    classes = tuple(ckpt.get("classes", ["NH4", "NO2"]))
    drop = float(ckpt.get("drop", 0.2))
    drop_path = float(ckpt.get("drop_path", 0.1))

    # Robust calibration provenance extraction
    raw_calib = (
        ckpt.get("calib_mode_train")
        or ckpt.get("calib_mode")
        or ckpt.get("calib")
        or ckpt.get("calibration")
        or ckpt.get("train_calib")
    )
    if raw_calib is None and ckpt_path is not None:
        p_str = str(ckpt_path).lower()
        if "_green" in p_str:
            raw_calib = "greenborder"
        elif "_none" in p_str:
            raw_calib = "none"

    raw_calib_str = str(raw_calib or "none").lower().strip()
    if raw_calib_str in {"green", "greenborder", "green_border"}:
        calib_mode_train = "greenborder"
    else:
        calib_mode_train = "none"

    loss_weight_cls = float(ckpt.get("loss_weight_cls", 1.0))
    loss_weight_reg = float(ckpt.get("loss_weight_reg", 2.0))
    seed = int(ckpt.get("seed", 0))

    return ModelMeta(
        timm_name=timm_name,
        num_classes=num_classes,
        image_size=image_size,
        ppm_scale=ppm_scale,
        ppm_min=ppm_min,
        ppm_max=ppm_max,
        classes=classes,
        drop=drop,
        drop_path=drop_path,
        calib_mode_train=calib_mode_train,
        loss_weight_cls=loss_weight_cls,
        loss_weight_reg=loss_weight_reg,
        seed=seed,
    )


def _infer_feat_dim(backbone: nn.Module, image_size: int = 224) -> int:
    """Robustly infer feature dimension from a timm backbone via forward pass probing."""
    try:
        backbone.eval()
        dummy = torch.zeros(1, 3, int(image_size), int(image_size))
        with torch.no_grad():
            feat = backbone(dummy)
            if isinstance(feat, (tuple, list)):
                feat = feat[-1]
            if feat.ndim > 2:
                feat = torch.flatten(feat, 1)
            return int(feat.shape[1])
    except Exception as e:
        logger.debug(f"Forward probing error: {e}. Falling back to attribute inference.")

    if hasattr(backbone, "num_features") and isinstance(backbone.num_features, int) and backbone.num_features > 0:
        return backbone.num_features
    if hasattr(backbone, "head_hidden_size") and isinstance(backbone.head_hidden_size, int) and backbone.head_hidden_size > 0:
        return backbone.head_hidden_size

    return 768


infer_feat_dim = _infer_feat_dim


def build_mlp2_head(in_features: int, out_features: int, drop: float = 0.3) -> nn.Sequential:
    """
    Canonical MLP2 Head: Linear(in, 512) -> ReLU -> Dropout(drop) -> Linear(512, out)
    """
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=drop),
        nn.Linear(512, out_features),
    )


def build_linear_head(in_features: int, out_features: int, drop: float = 0.2) -> nn.Sequential:
    """
    Legacy Linear Head: Dropout(drop) -> Linear(in, out)
    """
    return nn.Sequential(
        nn.Dropout(p=drop),
        nn.Linear(in_features, out_features),
    )


class MultiTaskHetero(nn.Module):
    """
    Canonical Multi-Task Heteroscedastic Neural Network.
    Shared backbone + 3 canonical MLP2 task heads.
    """

    def __init__(
        self,
        timm_name: str = "convnext_tiny.fb_in1k",
        num_classes: int = 2,
        pretrained: bool = False,
        drop: float = 0.2,
        drop_path: float = 0.1,
        image_size: int = 224,
    ):
        super().__init__()
        self.timm_name = timm_name
        self.num_classes = num_classes
        self.image_size = image_size

        kwargs: Dict[str, Any] = {"pretrained": pretrained, "num_classes": 0}
        if drop > 0:
            kwargs["drop_rate"] = drop
        if drop_path > 0:
            kwargs["drop_path_rate"] = drop_path

        self.backbone = timm.create_model(timm_name, **kwargs)
        self.feat_dim = _infer_feat_dim(self.backbone, image_size=image_size)

        # Canonical MLP2 heads
        self.head_cls = build_mlp2_head(self.feat_dim, num_classes, drop=0.3)
        self.head_reg_NH4 = build_mlp2_head(self.feat_dim, 2, drop=0.3)
        self.head_reg_NO2 = build_mlp2_head(self.feat_dim, 2, drop=0.3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[-1]
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)

        logits = self.head_cls(feat)
        r_nh4 = self.head_reg_NH4(feat)
        r_no2 = self.head_reg_NO2(feat)
        return logits, r_nh4, r_no2


class MultiTaskHeteroFlexible(nn.Module):
    """
    Adaptive loader supporting both canonical MLP2 heads and legacy Linear heads.
    """

    def __init__(
        self,
        timm_name: str,
        num_classes: int = 2,
        pretrained: bool = False,
        drop: float = 0.2,
        drop_path: float = 0.1,
        image_size: int = 224,
        head_variant: str = "mlp2",
        reg_out_dim: int = 2,
        expected_feat_dim: Optional[int] = None,
    ):
        super().__init__()
        self.timm_name = timm_name
        self.num_classes = num_classes
        self.image_size = image_size
        self.head_variant = head_variant
        self.reg_out_dim = reg_out_dim

        kwargs: Dict[str, Any] = {"pretrained": pretrained, "num_classes": 0}
        if drop > 0:
            kwargs["drop_rate"] = drop
        if drop_path > 0:
            kwargs["drop_path_rate"] = drop_path

        self.backbone = timm.create_model(timm_name, **kwargs)
        in_dim = expected_feat_dim if expected_feat_dim else _infer_feat_dim(self.backbone, image_size=image_size)
        self.feat_dim = in_dim

        if head_variant == "mlp2":
            self.head_cls = build_mlp2_head(in_dim, num_classes, drop=0.3)
            self.head_reg_NH4 = build_mlp2_head(in_dim, reg_out_dim, drop=0.3)
            self.head_reg_NO2 = build_mlp2_head(in_dim, reg_out_dim, drop=0.3)
        else:
            self.head_cls = build_linear_head(in_dim, num_classes, drop=drop)
            self.head_reg_NH4 = build_linear_head(in_dim, reg_out_dim, drop=drop)
            self.head_reg_NO2 = build_linear_head(in_dim, reg_out_dim, drop=drop)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[-1]
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)

        logits = self.head_cls(feat)
        r_nh4 = self.head_reg_NH4(feat)
        r_no2 = self.head_reg_NO2(feat)
        return logits, r_nh4, r_no2