# -*- coding: utf-8 -*-
"""
Preprocessing & Colorimetry Normalization Utilities.

Implements:
- IdentityNormalizer: Standard RGB [0, 1] normalization without calibration.
- GreenBorderNormalizer: Color reference patch normalization in linearized RGB space.
- Exact sRGB <-> Linear RGB transformations.
- Data augmentation pipelines for training and deterministic evaluation transforms.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from albumentations import (
    Affine,
    Compose,
    GaussianBlur,
    HueSaturationValue,
    Normalize,
    RandomBrightnessContrast,
    Resize,
    Rotate,
)

try:
    from albumentations.pytorch import ToTensorV2
except ImportError:
    from albumentations import ToTensorV2

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Convert sRGB array in [0, 1] to linear RGB."""
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1.0 + a)) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Convert linear RGB array in [0, 1] back to sRGB."""
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x, (1.0 + a) * (np.maximum(x, 0.0) ** (1.0 / 2.4)) - a)


class IdentityNormalizer:
    """
    Standard identity preprocessor: converts BGR image to RGB float32 in [0, 1].
    Used when calib_mode='none'.
    """

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None:
            raise ValueError("IdentityNormalizer received None image.")
        if image_bgr.dtype not in (np.float32, np.float64):
            rgb = image_bgr[..., ::-1].astype(np.float32) / 255.0
        else:
            rgb = image_bgr[..., ::-1].astype(np.float32)
        return np.clip(rgb, 0.0, 1.0)


class GreenBorderNormalizer:
    """
    Colorimetry normalization using the green reference border surrounding the strip.

    Workflow:
    1. Extract peripheral ring mask with nominal ring_frac and inner_margin.
    2. Segment reference green pixels via HSV color filtering.
    3. Convert to linear RGB and compute mean border channel intensities.
    4. Perform channel-wise divisive normalization in linear RGB.
    5. Convert back to sRGB [0, 1].

    Note: Retains exact historical numerical behavior for full reproduction.
    """

    def __init__(
        self,
        hsv_lower=(35, 40, 40),
        hsv_upper=(95, 255, 255),
        ring_frac: float = 0.08,
        inner_margin: int = 2,
        min_green_pixels: int = 300,
        epsilon: float = 1e-6,
        gamma: float = 1.0,
        min_border_value: float = 0.05,
        max_border_value: float = 1.0,
    ):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.ring_frac = float(ring_frac)
        self.inner_margin = int(inner_margin)
        self.min_green_pixels = int(min_green_pixels)
        self.eps = float(epsilon)
        self.gamma = float(gamma)
        self.min_border_value = float(min_border_value)
        self.max_border_value = float(max_border_value)

    def _to_rgb01(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr.dtype not in (np.float32, np.float64):
            img_bgr = img_bgr.astype(np.float32) / 255.0
        return np.clip(img_bgr[..., ::-1], 0.0, 1.0).astype(np.float32)

    def _ring_mask(self, h: int, w: int, ring_px: int) -> np.ndarray:
        m = np.zeros((h, w), dtype=np.uint8)
        m[:ring_px, :] = 255
        m[-ring_px:, :] = 255
        m[:, :ring_px] = 255
        m[:, -ring_px:] = 255

        im = self.inner_margin
        if 2 * im < h and 2 * im < w:
            m[im:-im, im:-im] = 0
        return m

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None:
            raise ValueError("GreenBorderNormalizer received None image.")

        rgb = self._to_rgb01(image_bgr)
        h, w = rgb.shape[:2]
        ring_px = max(2, int(min(h, w) * self.ring_frac))
        ring = self._ring_mask(h, w, ring_px)

        img_u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)

        mask_green = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.bitwise_and(mask_green, ring)
        green_pixels = mask > 0

        if green_pixels.sum() < self.min_green_pixels:
            mask = ring
            green_pixels = mask > 0

        lin = srgb_to_linear(rgb)

        if green_pixels.sum() == 0:
            mean_border = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        else:
            mean_border = lin[green_pixels].mean(axis=0).astype(np.float32)
            mean_border = np.clip(
                mean_border,
                self.min_border_value,
                self.max_border_value,
            )

        lin_out = lin / (mean_border[None, None, :] + self.eps)
        lin_out = np.clip(lin_out, 0.0, 1.0)

        out = linear_to_srgb(lin_out)
        if self.gamma != 1.0:
            out = np.clip(out, 0.0, 1.0) ** (1.0 / self.gamma)

        return np.clip(out, 0.0, 1.0).astype(np.float32)


def get_normalizer(
    mode: str,
    ring_frac: float = 0.08,
    inner_margin: int = 2,
    min_green_pixels: int = 300,
) -> Union[IdentityNormalizer, GreenBorderNormalizer]:
    """Factory helper returning appropriate normalizer instance."""
    m = (mode or "none").lower().strip()
    if m in ("none", "raw", "identity", "no"):
        return IdentityNormalizer()
    if m in ("greenborder", "green", "gb", "green_border"):
        return GreenBorderNormalizer(
            ring_frac=ring_frac,
            inner_margin=inner_margin,
            min_green_pixels=min_green_pixels,
        )
    raise ValueError(f"Unknown calibration mode: {mode}. Expected 'none' or 'greenborder'.")


def make_train_transform(image_size: int = 224) -> Compose:
    """Standardized training augmentations."""
    return Compose(
        [
            Resize(image_size, image_size, interpolation=cv2.INTER_AREA),
            Rotate(limit=10, p=0.5),
            Affine(scale=(0.97, 1.03), translate_percent=0.02, shear=4, p=0.5),
            HueSaturationValue(hue_shift_limit=5, sat_shift_limit=12, val_shift_limit=6, p=0.4),
            RandomBrightnessContrast(brightness_limit=0.06, contrast_limit=0.06, p=0.3),
            GaussianBlur(blur_limit=(3, 3), p=0.15),
            Normalize(mean=IMNET_MEAN, std=IMNET_STD, max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )


def make_eval_transform(image_size: int = 224) -> Compose:
    """Deterministic evaluation transforms."""
    return Compose(
        [
            Resize(image_size, image_size, interpolation=cv2.INTER_AREA),
            Normalize(mean=IMNET_MEAN, std=IMNET_STD, max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )