# ai_chemistry/training/gb_utils.py
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np
import torch
import torch.nn as nn

from albumentations import (
    Compose, Resize, Normalize, ToTensorV2,
    Rotate, Affine, RandomBrightnessContrast, GaussianBlur, HueSaturationValue
)
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD  = (0.229, 0.224, 0.225)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_mape(y_true, y_pred, eps=1e-6) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * (x ** (1 / 2.4)) - a)


# ================= Green Border Normalizer (same logic as your train_classifier.py) =================
class GreenBorderNormalizer:
    def __init__(
        self,
        hsv_lower=(35, 40, 40),
        hsv_upper=(95, 255, 255),
        ring_frac=0.08,
        inner_margin=2,
        min_green_pixels=300,
        epsilon=1e-6,
        gamma=1.0,
    ):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.ring_frac = float(ring_frac)
        self.inner_margin = int(inner_margin)
        self.min_green_pixels = int(min_green_pixels)
        self.eps = float(epsilon)
        self.gamma = float(gamma)

    def _to_rgb01(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr.dtype not in (np.float32, np.float64):
            img_bgr = img_bgr.astype(np.float32) / 255.0
        return np.clip(img_bgr[..., ::-1], 0.0, 1.0)

    def _ring_mask(self, h: int, w: int, ring_px: int) -> np.ndarray:
        m = np.zeros((h, w), dtype=np.uint8)
        m[:ring_px, :] = 255
        m[-ring_px:, :] = 255
        m[:, :ring_px] = 255
        m[:, -ring_px:] = 255
        im = self.inner_margin
        if 2 * im < h and 2 * im < w:
            # keep only ring; clear inside margin
            m[im:-im, im:-im] = np.where(m[im:-im, im:-im] > 0, 0, m[im:-im, im:-im])
        return m

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = self._to_rgb01(image_bgr)
        h, w = rgb.shape[:2]
        ring_px = max(2, int(min(h, w) * self.ring_frac))
        ring = self._ring_mask(h, w, ring_px)

        img_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.bitwise_and(mask, ring)

        green_pixels = mask > 0
        if green_pixels.sum() < self.min_green_pixels:
            mask = ring
            green_pixels = mask > 0

        if green_pixels.sum() == 0:
            mean_border = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        else:
            lin = srgb_to_linear(rgb)
            mean_border = lin[green_pixels].mean(axis=0).astype(np.float32)
            mean_border = np.clip(mean_border, 0.05, 1.0)

        lin = srgb_to_linear(rgb)
        norm_lin = lin / (mean_border[None, None, :] + self.eps)
        norm_lin = np.clip(norm_lin, 0.0, 1.0)
        norm = linear_to_srgb(norm_lin)
        return np.clip(norm, 0.0, 1.0).astype(np.float32)


class IdentityNormalizer:
    """No calibration: just convert BGR uint8 -> RGB float32 [0,1]."""
    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr.dtype not in (np.float32, np.float64):
            image_bgr = image_bgr.astype(np.float32) / 255.0
        rgb = image_bgr[..., ::-1]
        return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def make_transforms(image_size: int = 224, train: bool = True):
    if train:
        return Compose([
            Resize(image_size, image_size, interpolation=cv2.INTER_AREA),
            Rotate(limit=10, p=0.5),
            Affine(scale=(0.97, 1.03), translate_percent=0.02, shear=4, p=0.5),
            HueSaturationValue(hue_shift_limit=5, sat_shift_limit=12, val_shift_limit=6, p=0.4),
            RandomBrightnessContrast(brightness_limit=0.06, contrast_limit=0.06, p=0.3),
            GaussianBlur(blur_limit=(3, 3), p=0.15),
            Normalize(mean=IMNET_MEAN, std=IMNET_STD, max_pixel_value=1.0),
            ToTensorV2(),
        ])
    return Compose([
        Resize(image_size, image_size, interpolation=cv2.INTER_AREA),
        Normalize(mean=IMNET_MEAN, std=IMNET_STD, max_pixel_value=1.0),
        ToTensorV2(),
    ])


def norm_path_str(p: str) -> str:
    return str(p).replace("\\", "/")

def extract_device_from_path(path_str: str) -> str:
    p = norm_path_str(path_str)
    parts = p.split("/")
    # cases:
    # 1) train/NH4/device/round/file
    # 2) NH4/device/round/file
    if len(parts) >= 3 and parts[0] in ("train", "val", "test"):
        return parts[2] if len(parts) > 2 else "UNKNOWN"
    if len(parts) >= 2:
        return parts[1]
    return "UNKNOWN"


# ===== scaling for regression =====
def scale_ppm(ppm: float, ppm_scale: str, ppm_min: Optional[float], ppm_max: Optional[float]) -> float:
    if ppm_scale == "log1p":
        return math.log1p(ppm)
    if ppm_scale == "minmax":
        assert ppm_min is not None and ppm_max is not None
        return (ppm - ppm_min) / (ppm_max - ppm_min + 1e-12)
    return ppm

def inverse_scale_ppm(y_scaled: np.ndarray, ppm_scale: str, ppm_min: Optional[float], ppm_max: Optional[float]) -> np.ndarray:
    y_scaled = np.asarray(y_scaled, dtype=float)
    if ppm_scale == "log1p":
        return np.expm1(y_scaled)
    if ppm_scale == "minmax":
        assert ppm_min is not None and ppm_max is not None
        return y_scaled * (ppm_max - ppm_min) + ppm_min
    return y_scaled


# ===== metrics =====
def cls_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }

def reg_metrics(y_true_ppm, y_pred_ppm) -> Dict[str, float]:
    y_true_ppm = np.asarray(y_true_ppm, dtype=float)
    y_pred_ppm = np.asarray(y_pred_ppm, dtype=float)
    mae = float(mean_absolute_error(y_true_ppm, y_pred_ppm))
    mse = float(mean_squared_error(y_true_ppm, y_pred_ppm))
    rmse = float(math.sqrt(mse))
    r2 = float(r2_score(y_true_ppm, y_pred_ppm))
    mape = safe_mape(y_true_ppm, y_pred_ppm, eps=1e-8)
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mape": mape}


# ===== checkpoint meta I/O =====
def save_meta(meta_path: Path, meta: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

def load_meta(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text(encoding="utf-8"))
