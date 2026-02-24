from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations import Compose, Normalize, Resize, ToTensorV2

from .calibration import build_normalizer
from .modeling import (
    Meta,
    MultiTaskHeteroFlexible,
    build_meta_from_ckpt,
    infer_head_variant,
    infer_reg_out_dim,
    strip_state_dict_prefix,
)


IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


def inverse_scale(ppm_scaled: float, ppm_scale: str, ppm_min: Optional[float] = None, ppm_max: Optional[float] = None) -> float:
    if ppm_scale == "log1p":
        return float(math.expm1(ppm_scaled))
    if ppm_scale == "minmax" and ppm_min is not None and ppm_max is not None:
        return float(ppm_scaled * (ppm_max - ppm_min) + ppm_min)
    return float(ppm_scaled)


def make_eval_tf(image_size: int) -> Compose:
    return Compose(
        [
            Resize(image_size, image_size, interpolation=cv2.INTER_AREA),
            Normalize(mean=IMNET_MEAN, std=IMNET_STD, max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )


@dataclass
class Prediction:
    chemical: str
    chemical_conf: float
    ppm: float
    ppm_ci95: Optional[Tuple[float, float]]
    ppm_sigma: Optional[float]
    raw: Dict[str, Any]


class LoadedPredictor:
    def __init__(self, ckpt_path: Path, meta_path: Optional[Path], device: str, calib_mode: str):
        self.ckpt_path = Path(ckpt_path)
        self.meta_path = Path(meta_path) if meta_path else None
        self.device = self._resolve_device(device)
        self.calib_mode = calib_mode

        ckpt = torch.load(str(self.ckpt_path), map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        if not isinstance(state, dict):
            raise RuntimeError(f"Checkpoint không hợp lệ: {self.ckpt_path}")

        state = strip_state_dict_prefix(state)

        # meta: prefer JSON if exists, else from ckpt
        if self.meta_path and self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            self.meta = Meta(
                timm_name=m.get("timm_name", ckpt.get("timm_name", "convnext_base.fb_in1k")),
                num_classes=int(m.get("num_classes", ckpt.get("num_classes", 2))),
                image_size=int(m.get("image_size", ckpt.get("image_size", 224))),
                ppm_scale=str(m.get("ppm_scale", ckpt.get("ppm_scale", "log1p"))),
                ppm_min=m.get("ppm_min", ckpt.get("ppm_min", None)),
                ppm_max=m.get("ppm_max", ckpt.get("ppm_max", None)),
                classes=tuple(m.get("classes", ckpt.get("classes", ["NH4", "NO2"]))),
                drop=float(m.get("drop", ckpt.get("drop", 0.2))),
                drop_path=float(m.get("drop_path", ckpt.get("drop_path", 0.1))),
            )
        else:
            self.meta = build_meta_from_ckpt(ckpt)

        # infer which head style this ckpt expects
        head_variant = infer_head_variant(state)
        reg_out_dim = infer_reg_out_dim(state)

        # build model
        self.model = MultiTaskHeteroFlexible(
            timm_name=self.meta.timm_name,
            num_classes=self.meta.num_classes,
            pretrained=False,
            drop=self.meta.drop,
            drop_path=self.meta.drop_path,
            head_variant=head_variant,
            reg_out_dim=reg_out_dim,
        )

        # load weights (try strict first; fallback if needed)
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError:
            # try the other head variant once
            alt = "linear" if head_variant == "mlp2" else "mlp2"
            self.model = MultiTaskHeteroFlexible(
                timm_name=self.meta.timm_name,
                num_classes=self.meta.num_classes,
                pretrained=False,
                drop=self.meta.drop,
                drop_path=self.meta.drop_path,
                head_variant=alt,
                reg_out_dim=reg_out_dim,
            )
            self.model.load_state_dict(state, strict=True)

        self.model.eval().to(self.device)

        self.tf = make_eval_tf(self.meta.image_size)
        self.normalizer = build_normalizer(self.calib_mode)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        device = (device or "cuda").lower().strip()
        if device.startswith("cuda") and torch.cuda.is_available():
            return torch.device("cuda")
        if device.startswith("mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @torch.inference_mode()
    def predict(self, roi_bgr: np.ndarray, mc_samples: int = 200) -> Prediction:
        # 1) calibration -> RGB in [0,1]
        rgb01 = self.normalizer(roi_bgr)  # shape HxWx3 float32

        # 2) transforms -> tensor
        x = self.tf(image=rgb01)["image"].unsqueeze(0).to(self.device)

        # 3) forward
        out = self.model(x)
        logits, reg_nh4, reg_no2 = out[0], out[1], out[2]  # support models that return (cls, nh4, no2, feats)

        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        cls_idx = int(probs.argmax())
        chemical = str(self.meta.classes[cls_idx]) if cls_idx < len(self.meta.classes) else ("NH4" if cls_idx == 0 else "NO2")
        chemical_conf = float(probs[cls_idx])

        reg = reg_nh4 if chemical.upper().startswith("NH4") else reg_no2
        reg = reg.detach().cpu().numpy()[0]

        # reg could be [mu] or [mu, logvar]
        mu_s = float(reg[0])
        logvar_s = float(reg[1]) if reg.shape[0] >= 2 else None

        ppm_mean = inverse_scale(mu_s, self.meta.ppm_scale, self.meta.ppm_min, self.meta.ppm_max)
        ppm_mean = max(0.0, float(ppm_mean))

        ppm_ci = None
        ppm_sigma = None

        if logvar_s is not None:
            sigma_s = float(np.exp(0.5 * logvar_s))
            # MC sample in scaled space -> convert to ppm
            z = np.random.randn(int(mc_samples)).astype(np.float32)
            samples_s = mu_s + sigma_s * z
            samples_ppm = np.array([inverse_scale(float(v), self.meta.ppm_scale, self.meta.ppm_min, self.meta.ppm_max) for v in samples_s], dtype=np.float32)
            samples_ppm = np.clip(samples_ppm, 0.0, np.inf)

            lo = float(np.quantile(samples_ppm, 0.025))
            hi = float(np.quantile(samples_ppm, 0.975))
            ppm_ci = (lo, hi)
            ppm_sigma = float(samples_ppm.std())

        raw = {
            "probs": probs.tolist(),
            "mu_scaled": mu_s,
            "logvar_scaled": logvar_s,
            "ppm_scale": self.meta.ppm_scale,
            "timm_name": self.meta.timm_name,
            "image_size": self.meta.image_size,
        }

        return Prediction(
            chemical=chemical,
            chemical_conf=chemical_conf,
            ppm=ppm_mean,
            ppm_ci95=ppm_ci,
            ppm_sigma=ppm_sigma,
            raw=raw,
        )
