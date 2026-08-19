# -*- coding: utf-8 -*-
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

from ai_chemistry.data.loaders import inverse_scale_ppm
from ai_chemistry.modeling import (
    ModelMeta,
    MultiTaskHeteroFlexible,
    build_meta_from_ckpt,
    infer_head_in_features,
    infer_head_variant,
    infer_reg_out_dim,
    strip_state_dict_prefix,
)
from ai_chemistry.preprocessing import get_normalizer, make_eval_transform


@dataclass
class Prediction:
    chemical: str
    chemical_conf: float
    ppm: float
    ppm_ci95: Optional[Tuple[float, float]]
    ppm_sigma: Optional[float]
    raw: Dict[str, Any]


class LoadedPredictor:
    """
    Inference wrapper for multi-task heteroscedastic strip colorimetry models.
    Provides predicted analyte class, expected concentration (ppm), and
    an optional approximate 95% predictive interval derived from the heteroscedastic log-variance.
    """

    def __init__(self, ckpt_path: Path, meta_path: Optional[Path], device: str, calib_mode: str):
        self.ckpt_path = Path(ckpt_path).resolve()
        self.meta_path = Path(meta_path).resolve() if meta_path else None
        self.device = self._resolve_device(device)
        self.calib_mode = calib_mode

        ckpt = torch.load(str(self.ckpt_path), map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        if not isinstance(state, dict):
            raise RuntimeError(f"Invalid state dict in checkpoint: {self.ckpt_path}")

        state = strip_state_dict_prefix(state)

        if self.meta_path and self.meta_path.is_file():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            self.meta = ModelMeta(
                timm_name=m.get("timm_name", ckpt.get("timm_name", "convnext_tiny.fb_in1k")),
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

        head_variant = infer_head_variant(state)
        reg_out_dim = infer_reg_out_dim(state)
        expected_feat_dim = infer_head_in_features(state)

        self.model = MultiTaskHeteroFlexible(
            timm_name=self.meta.timm_name,
            num_classes=self.meta.num_classes,
            pretrained=False,
            drop=self.meta.drop,
            drop_path=self.meta.drop_path,
            image_size=self.meta.image_size,
            head_variant=head_variant,
            reg_out_dim=reg_out_dim,
            expected_feat_dim=expected_feat_dim,
        )

        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError:
            alt_variant = "linear" if head_variant == "mlp2" else "mlp2"
            self.model = MultiTaskHeteroFlexible(
                timm_name=self.meta.timm_name,
                num_classes=self.meta.num_classes,
                pretrained=False,
                drop=self.meta.drop,
                drop_path=self.meta.drop_path,
                image_size=self.meta.image_size,
                head_variant=alt_variant,
                reg_out_dim=reg_out_dim,
                expected_feat_dim=expected_feat_dim,
            )
            self.model.load_state_dict(state, strict=True)

        self.model.eval().to(self.device)
        self.tf = make_eval_transform(self.meta.image_size)
        self.normalizer = get_normalizer(self.calib_mode)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        dev = (device or "cuda").lower().strip()
        if dev.startswith("cuda") and torch.cuda.is_available():
            return torch.device("cuda")
        if dev.startswith("mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @torch.inference_mode()
    def predict(self, roi_bgr: np.ndarray, mc_samples: int = 200, deterministic: bool = True) -> Prediction:
        # 1. Color Calibration
        rgb01 = self.normalizer(roi_bgr)

        # 2. Preprocessing Transform
        x = self.tf(image=rgb01)["image"].unsqueeze(0).to(self.device)

        # 3. Model Forward Pass
        out = self.model(x)
        logits, reg_nh4, reg_no2 = out[0], out[1], out[2]

        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        cls_idx = int(probs.argmax())
        chemical = (
            str(self.meta.classes[cls_idx])
            if cls_idx < len(self.meta.classes)
            else ("NH4" if cls_idx == 0 else "NO2")
        )
        chemical_conf = float(probs[cls_idx])

        reg = reg_nh4 if chemical.upper().startswith("NH4") else reg_no2
        reg = reg.detach().cpu().numpy()[0]

        mu_s = float(reg[0])
        logvar_s = float(reg[1]) if reg.shape[0] >= 2 else None

        ppm_mean = inverse_scale_ppm(mu_s, self.meta.ppm_scale, self.meta.ppm_min, self.meta.ppm_max)
        ppm_mean = max(0.0, float(ppm_mean))

        ppm_ci = None
        ppm_sigma = None

        if logvar_s is not None:
            sigma_s = float(np.exp(0.5 * logvar_s))
            if deterministic:
                # Exact analytical 95% interval in scaled Gaussian space (z = 1.95996)
                z95 = 1.959963984540054
                lo_s = mu_s - z95 * sigma_s
                hi_s = mu_s + z95 * sigma_s
                lo_ppm = max(0.0, float(inverse_scale_ppm(lo_s, self.meta.ppm_scale, self.meta.ppm_min, self.meta.ppm_max)))
                hi_ppm = max(0.0, float(inverse_scale_ppm(hi_s, self.meta.ppm_scale, self.meta.ppm_min, self.meta.ppm_max)))
                ppm_ci = (min(lo_ppm, hi_ppm), max(lo_ppm, hi_ppm))
                # Approximate delta-method standard deviation in original ppm space
                if self.meta.ppm_scale == "log1p":
                    ppm_sigma = float(sigma_s * math.exp(mu_s))
                else:
                    ppm_sigma = float(sigma_s)
            else:
                rng = np.random.RandomState(0)
                z = rng.randn(int(mc_samples)).astype(np.float32)
                samples_s = mu_s + sigma_s * z
                samples_ppm = np.array(
                    [
                        inverse_scale_ppm(float(v), self.meta.ppm_scale, self.meta.ppm_min, self.meta.ppm_max)
                        for v in samples_s
                    ],
                    dtype=np.float32,
                )
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