from __future__ import annotations

import cv2
import numpy as np


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * (x ** (1 / 2.4)) - a)


class IdentityNormalizer:
    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        # return RGB float32 in [0,1]
        if image_bgr.dtype not in (np.float32, np.float64):
            image_bgr = image_bgr.astype(np.float32) / 255.0
        rgb = image_bgr[..., ::-1]
        return np.clip(rgb, 0.0, 1.0).astype(np.float32)


class GreenBorderNormalizer:
    """
    Same idea as your training script:
    - find green border pixels in the outer ring (HSV threshold)
    - compute per-channel mean in linear space
    - normalize whole ROI by that mean, then back to sRGB
    Returns RGB float32 in [0,1].
    """

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
        if int(green_pixels.sum()) < self.min_green_pixels:
            mask = ring
            green_pixels = mask > 0

        if int(green_pixels.sum()) == 0:
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


def build_normalizer(mode: str):
    mode = (mode or "greenborder").lower().strip()
    if mode in ("none", "no", "off", "identity"):
        return IdentityNormalizer()
    return GreenBorderNormalizer()
