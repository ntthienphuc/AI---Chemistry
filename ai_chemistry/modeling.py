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


def infer_reg_out_dim(state: Dict[str, torch.Tensor]) -> int:
    """Infer regression output dimension (2 for heteroscedastic mu+logvar, 1 for homoscedastic mu)."""
    for cand in ("head_reg_NH4.3.weight", "head_reg_NH4.1.weight", "head_reg_NH4.weight"):
        if cand in state:
            return int(state[cand].shape[0])
    return 2


def infer_head_in_features(state: Dict[str, torch.Tensor]) -> Optional[int]:
    """Infer input feature dimension expected by the first linear layer in the saved head."""
    for cand in ("head_cls.0.weight", "head_cls.1.weight", "head_cls.weight"):
        if cand in state:
            return int(state[cand].shape[1])
    return None


def unwrap_output(y: Any) -> torch.Tensor:
    """Unwrap backbone outputs that return tuple, list, or dict."""
    if isinstance(y, (tuple, list)):
        y = y[0]
    if isinstance(y, dict):
        y = y.get("x", next(iter(y.values())))
    return y


def infer_feat_dim(backbone: nn.Module, image_size: int = 224) -> int:
    """
    Robustly determine pre-logits feature dimension using a mock zero tensor forward pass.
    Restores original train/eval state after probing.
    """
    was_training = backbone.training
    backbone.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, image_size, image_size)
        y = unwrap_output(backbone(x))
        if y.ndim == 4:
            y = y.mean(dim=(2, 3))
        elif y.ndim != 2:
            y = y.view(y.size(0), -1)
        feat_dim = int(y.shape[1])
    backbone.train(was_training)
    return feat_dim


class MultiTaskHetero(nn.Module):
    """
    Canonical Multi-Task Heteroscedastic Deep Neural Network for Smartphone Colorimetry.

    Architecture:
    - Backbone: Pretrained timm vision backbone (e.g. ConvNeXt, MobileNetV3, Swin, NFNet, EfficientNet)
    - Head 1 (Classification): Linear(d, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 2)
    - Head 2 (NH4+ Regression): Linear(d, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 2) [mu, log_var]
    - Head 3 (NO2- Regression): Linear(d, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 2) [mu, log_var]
    """

    def __init__(
        self,
        timm_name: str,
        num_classes: int = 2,
        pretrained: bool = True,
        drop: float = 0.2,
        drop_path: float = 0.1,
        image_size: int = 224,
        reg_out_dim: int = 2,
        hidden_dim: int = 512,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.timm_name = timm_name
        self.num_classes = int(num_classes)
        self.reg_out_dim = int(reg_out_dim)
        self.image_size = int(image_size)

        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop,
            drop_path_rate=drop_path,
        )

        self.feat_dim = infer_feat_dim(self.backbone, image_size=self.image_size)

        # Canonical MLP2 Task Heads
        self.head_cls = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )

        self.head_reg_NH4 = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, self.reg_out_dim),
        )

        self.head_reg_NO2 = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, self.reg_out_dim),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = unwrap_output(self.backbone(x))
        if feats.ndim == 4:
            feats = feats.mean(dim=(2, 3))
        elif feats.ndim != 2:
            feats = feats.view(feats.size(0), -1)
        return feats

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.extract_features(x)
        logits = self.head_cls(feats)
        reg_nh4 = self.head_reg_NH4(feats)
        reg_no2 = self.head_reg_NO2(feats)
        return logits, reg_nh4, reg_no2


class MultiTaskHeteroFlexible(nn.Module):
    """
    Flexible multi-task model capable of loading both canonical MLP2 heads
    and legacy single-layer linear heads for complete backward compatibility.
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
        hidden_dim: int = 512,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.timm_name = timm_name
        self.num_classes = int(num_classes)
        self.head_variant = head_variant
        self.reg_out_dim = int(reg_out_dim)
        self.image_size = int(image_size)
        self.use_forward_head_features = False

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
            feat_dim = infer_feat_dim(self.backbone, image_size=self.image_size)

        if expected_feat_dim is not None and int(expected_feat_dim) != int(feat_dim):
            if hasattr(self.backbone, "forward_features") and hasattr(self.backbone, "forward_head"):
                feat_dim = int(expected_feat_dim)
                self.use_forward_head_features = True

        self.feat_dim = int(feat_dim)

        if self.head_variant == "linear":
            self.head_cls = nn.Sequential(
                nn.Dropout(p=head_dropout),
                nn.Linear(self.feat_dim, self.num_classes),
            )
            self.head_reg_NH4 = nn.Sequential(
                nn.Dropout(p=head_dropout),
                nn.Linear(self.feat_dim, self.reg_out_dim),
            )
            self.head_reg_NO2 = nn.Sequential(
                nn.Dropout(p=head_dropout),
                nn.Linear(self.feat_dim, self.reg_out_dim),
            )
        else:
            # Canonical MLP2
            self.head_cls = nn.Sequential(
                nn.Linear(self.feat_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(head_dropout),
                nn.Linear(hidden_dim, self.num_classes),
            )
            self.head_reg_NH4 = nn.Sequential(
                nn.Linear(self.feat_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(head_dropout),
                nn.Linear(hidden_dim, self.reg_out_dim),
            )
            self.head_reg_NO2 = nn.Sequential(
                nn.Linear(self.feat_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(head_dropout),
                nn.Linear(hidden_dim, self.reg_out_dim),
            )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_forward_head_features:
            feats = self.backbone.forward_features(x)
            try:
                return self.backbone.forward_head(feats, pre_logits=True)
            except TypeError:
                return self.backbone.forward_head(feats)

        feats = unwrap_output(self.backbone(x))
        if feats.ndim == 4:
            feats = feats.mean(dim=(2, 3))
        elif feats.ndim != 2:
            feats = feats.view(feats.size(0), -1)
        return feats

    def forward(self, x: torch.Tensor):
        feats = self.extract_features(x)
        logits = self.head_cls(feats)
        reg_nh4 = self.head_reg_NH4(feats)
        reg_no2 = self.head_reg_NO2(feats)
        return logits, reg_nh4, reg_no2, feats


def build_meta_from_ckpt(ckpt: Dict[str, Any]) -> ModelMeta:
    """Extract metadata dictionary or assign defaults from a loaded checkpoint."""
    timm_name = ckpt.get("timm_name", "convnext_tiny.fb_in1k")
    num_classes = int(ckpt.get("num_classes", 2))
    image_size = int(ckpt.get("image_size", 224))
    ppm_scale = str(ckpt.get("ppm_scale", "log1p"))
    ppm_min = ckpt.get("ppm_min", None)
    ppm_max = ckpt.get("ppm_max", None)
    classes = tuple(ckpt.get("classes", ["NH4", "NO2"]))
    drop = float(ckpt.get("drop", 0.2))
    drop_path = float(ckpt.get("drop_path", 0.1))
    calib_mode_train = str(ckpt.get("calib_mode_train", "none"))
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