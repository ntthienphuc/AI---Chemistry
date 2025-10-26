"""Inference pipeline combining YOLO ROI detection and classifier/regressor predictions."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations import Compose, Normalize, Resize, ToTensorV2
from ultralytics import YOLO

import timm

LOGGER = logging.getLogger(__name__)

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


def inverse_scale(ppm_scaled: float, ppm_scale: str, ppm_min: Optional[float], ppm_max: Optional[float]) -> float:
    if ppm_scale == "log1p":
        return float(np.expm1(ppm_scaled))
    if ppm_scale == "minmax" and ppm_min is not None and ppm_max is not None:
        return float(ppm_scaled * (ppm_max - ppm_min) + ppm_min)
    return float(ppm_scaled)


class GreenBorderNormalizer:
    """Normalize strips by the mean color of the surrounding green border."""

    def __init__(
        self,
        hsv_lower: Sequence[int] = (35, 40, 40),
        hsv_upper: Sequence[int] = (95, 255, 255),
        ring_frac: float = 0.10,
        inner_margin: int = 2,
        min_green_pixels: int = 500,
        eps: float = 1e-6,
    ) -> None:
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.ring_frac = float(ring_frac)
        self.inner_margin = int(inner_margin)
        self.min_green_pixels = int(min_green_pixels)
        self.eps = float(eps)

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = rgb.shape[:2]

        ring_px = max(2, int(min(h, w) * self.ring_frac))
        ring_mask = np.zeros((h, w), dtype=np.uint8)
        ring_mask[:ring_px, :] = 1
        ring_mask[-ring_px:, :] = 1
        ring_mask[:, :ring_px] = 1
        ring_mask[:, -ring_px:] = 1
        if self.inner_margin * 2 < h and self.inner_margin * 2 < w:
            ring_mask[self.inner_margin : h - self.inner_margin, self.inner_margin : w - self.inner_margin] = 0

        hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        green_mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper) > 0
        mask = green_mask & (ring_mask > 0)
        if mask.sum() < self.min_green_pixels:
            mask = ring_mask > 0

        if mask.sum() == 0:
            border_mean = np.ones(3, dtype=np.float32)
        else:
            border_mean = rgb[mask].mean(axis=0)
            border_mean = np.clip(border_mean, 0.05, 1.0)

        normalized = rgb / (border_mean + self.eps)
        normalized = np.clip(normalized, 0.0, 1.0)
        return (normalized * 255).astype(np.uint8)


class ChemistryModel(nn.Module):
    """Backbone + two regression heads with optional heteroscedastic outputs."""

    def __init__(
        self,
        timm_name: str,
        num_classes: int,
        two_reg_heads: bool = True,
        heteroscedastic: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.heteroscedastic = heteroscedastic
        self.two_reg_heads = two_reg_heads

        self.backbone = timm.create_model(
            timm_name,
            pretrained=False,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            feat_dim = self.backbone.feature_info[-1]["num_chs"]

        self.head_cls = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        def reg_head() -> nn.Sequential:
            out_dim = 2 if heteroscedastic else 1
            return nn.Sequential(
                nn.Linear(feat_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, out_dim),
            )

        if two_reg_heads:
            self.head_reg_NH4 = reg_head()
            self.head_reg_NO2 = reg_head()
        else:
            self.head_reg_shared = reg_head()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)
        logits = self.head_cls(feats)
        if self.two_reg_heads:
            reg_nh4 = self.head_reg_NH4(feats)
            reg_no2 = self.head_reg_NO2(feats)
            return {"logits": logits, "reg_NH4": reg_nh4, "reg_NO2": reg_no2}
        reg = self.head_reg_shared(feats)
        return {"logits": logits, "reg_shared": reg}


@dataclass
class ClassifierConfig:
    classes: List[str]
    ppm_scale: str
    ppm_min: Optional[float]
    ppm_max: Optional[float]
    image_size: int
    timm_name: str
    two_reg_heads: bool = True
    heteroscedastic: bool = True


def load_classifier(weights_path: Path, device: torch.device) -> Tuple[ChemistryModel, ClassifierConfig]:
    ckpt = torch.load(weights_path, map_location=device)
    meta_path = weights_path.with_suffix(".meta.json")
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    def choose(key: str, default=None):
        return ckpt.get(key, meta.get(key, default))

    classes = choose("classes", ["NH4", "NO2"])
    config = ClassifierConfig(
        classes=classes,
        ppm_scale=choose("ppm_scale", "log1p"),
        ppm_min=choose("ppm_min"),
        ppm_max=choose("ppm_max"),
        image_size=int(choose("image_size", 224)),
        timm_name=choose("timm_name", "efficientnet_b0"),
        two_reg_heads=bool(choose("two_reg_heads", True)),
        heteroscedastic=bool(choose("heteroscedastic", True)),
    )

    model = ChemistryModel(
        timm_name=config.timm_name,
        num_classes=len(config.classes),
        two_reg_heads=config.two_reg_heads,
        heteroscedastic=config.heteroscedastic,
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, config


def build_inference_transform(image_size: int) -> Compose:
    return Compose(
        [
            Resize(image_size, image_size, interpolation=cv2.INTER_AREA),
            Normalize(mean=IMNET_MEAN, std=IMNET_STD, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def crop_with_yolo(yolo: YOLO, image: np.ndarray, imgsz: int, conf: float, padding: float) -> Optional[np.ndarray]:
    results = yolo.predict(image, imgsz=imgsz, conf=conf, verbose=False)
    result = results[0]

    if result.masks is not None:
        masks = result.masks.data  # type: ignore[attr-defined]
        areas = masks.sum(dim=(1, 2))
        idx = int(torch.argmax(areas).item())
        mask = masks[idx].cpu().numpy()
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(mask > 0.5)
        if ys.size == 0 or xs.size == 0:
            return None
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
    elif result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(np.argmax(areas))
        x0, y0, x1, y1 = boxes[idx]
    else:
        return None

    h, w = image.shape[:2]
    pad_x = int((x1 - x0) * padding)
    pad_y = int((y1 - y0) * padding)
    x0 = max(0, int(x0) - pad_x)
    x1 = min(w - 1, int(x1) + pad_x)
    y0 = max(0, int(y0) - pad_y)
    y1 = min(h - 1, int(y1) + pad_y)
    return image[y0 : y1 + 1, x0 : x1 + 1]


class ChemistryPipeline:
    """Encapsulates the ROI detector and classifier/regressor for single-image predictions."""

    def __init__(
        self,
        yolo_weights: Path,
        classifier_weights: Path,
        device: Optional[str] = None,
        yolo_conf: float = 0.25,
        yolo_imgsz: int = 640,
        roi_padding: float = 0.10,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.yolo = YOLO(str(yolo_weights))
        self.yolo_conf = yolo_conf
        self.yolo_imgsz = yolo_imgsz
        self.roi_padding = roi_padding
        self.classifier, self.config = load_classifier(Path(classifier_weights), self.device)
        self.transform = build_inference_transform(self.config.image_size)
        self.normalizer = GreenBorderNormalizer()

    def _prepare_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        normalized = self.normalizer(image_bgr)
        tensor = self.transform(image=normalized)["image"]
        return tensor.unsqueeze(0).to(self.device)

    def _predict_tensor(self, tensor: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            outputs = self.classifier(tensor)
            logits = outputs["logits"]
            probs = F.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)
            pred_idx = int(idx.item())
            confidence = float(conf.item())
            chemical = self.config.classes[pred_idx]

            if self.config.two_reg_heads:
                if pred_idx >= len(self.config.classes):
                    raise ValueError("Predicted index outside class list.")
                reg_output = outputs["reg_NH4"] if pred_idx == 0 else outputs["reg_NO2"]
            else:
                reg_output = outputs["reg_shared"]

            if self.config.heteroscedastic:
                ppm_scaled = float(reg_output[:, 0].item())
                sigma = float(torch.exp(0.5 * reg_output[:, 1]).item())
            else:
                ppm_scaled = float(reg_output[:, 0].item())
                sigma = 0.0

            ppm = inverse_scale(ppm_scaled, self.config.ppm_scale, self.config.ppm_min, self.config.ppm_max)
            return {"chemical": chemical, "ppm": ppm, "confidence": confidence, "ppm_scaled": ppm_scaled, "sigma": sigma}

    def predict_array(self, image_bgr: np.ndarray) -> Dict[str, float]:
        roi = crop_with_yolo(self.yolo, image_bgr, self.yolo_imgsz, self.yolo_conf, self.roi_padding)
        if roi is None:
            raise ValueError("YOLO could not find a valid ROI for the provided image.")
        tensor = self._prepare_tensor(roi)
        return self._predict_tensor(tensor)

    def predict_path(self, image_path: Path) -> Dict[str, float]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        result = self.predict_array(image)
        result["path"] = str(image_path)
        return result


def iter_images(input_path: Path, extensions: Iterable[str]) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    for ext in extensions:
        yield from input_path.rglob(f"*.{ext.lstrip('.')}")


def run_cli(args: argparse.Namespace) -> None:
    pipeline = ChemistryPipeline(
        yolo_weights=args.yolo_weights,
        classifier_weights=args.classifier_weights,
        device=args.device,
        yolo_conf=args.yolo_conf,
        yolo_imgsz=args.yolo_imgsz,
        roi_padding=args.roi_padding,
    )

    paths = list(iter_images(args.input_path, ("jpg", "jpeg", "png")))
    if args.output_csv:
        import csv

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "chemical", "ppm", "confidence", "ppm_scaled", "sigma"])
            for path in paths:
                result = pipeline.predict_path(path)
                writer.writerow(
                    [
                        result.get("path", str(path)),
                        result["chemical"],
                        f"{result['ppm']:.6f}",
                        f"{result['confidence']:.4f}",
                        f"{result['ppm_scaled']:.6f}",
                        f"{result['sigma']:.6f}",
                    ]
                )
        LOGGER.info("Saved predictions to %s", args.output_csv)
    else:
        for path in paths:
            result = pipeline.predict_path(path)
            LOGGER.info("%s -> %s @ %.4f ppm (conf=%.3f)", path, result["chemical"], result["ppm"], result["confidence"])


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference on NH4/NO2 strips using YOLO ROI + classifier.")
    parser.add_argument("--input-path", type=Path, required=True, help="Image file or directory to process.")
    parser.add_argument("--yolo-weights", type=Path, required=True, help="Path to YOLO detector weights (.pt).")
    parser.add_argument("--classifier-weights", type=Path, required=True, help="Path to classifier/regressor weights (.pt).")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV path for saving batch predictions.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. 'cuda' or 'cpu'.")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--roi-padding", type=float, default=0.10, help="Padding applied around detected ROI before cropping.")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    run_cli(build_cli().parse_args())
