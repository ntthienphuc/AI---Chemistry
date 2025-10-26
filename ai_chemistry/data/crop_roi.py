"""Crop regions-of-interest from raw strips using a YOLO segmentation/detection model."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)


@dataclass
class CropConfig:
    padding: float = 0.10
    conf: float = 0.25
    imgsz: int = 640
    extensions: Sequence[str] = ("jpg", "jpeg", "png")


def crop_from_mask(image: np.ndarray, mask: np.ndarray, padding: float) -> Optional[np.ndarray]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    pad_x = int((x1 - x0) * padding)
    pad_y = int((y1 - y0) * padding)
    h, w = image.shape[:2]
    x0 = max(0, x0 - pad_x)
    x1 = min(w - 1, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(h - 1, y1 + pad_y)
    return image[y0 : y1 + 1, x0 : x1 + 1]


def crop_from_box(image: np.ndarray, box: Sequence[int], padding: float) -> np.ndarray:
    x0, y0, x1, y1 = map(int, box)
    w_box = x1 - x0
    h_box = y1 - y0
    pad_x = int(w_box * padding)
    pad_y = int(h_box * padding)
    h, w = image.shape[:2]
    x0 = max(0, x0 - pad_x)
    x1 = min(w - 1, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(h - 1, y1 + pad_y)
    return image[y0 : y1 + 1, x0 : x1 + 1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def process_image(model: YOLO, image_path: Path, save_path: Path, config: CropConfig) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        LOGGER.warning("Could not load image %s", image_path)
        return False

    predictions = model.predict(str(image_path), conf=config.conf, imgsz=config.imgsz, verbose=False)
    result = predictions[0]

    if result.masks is not None:
        masks = result.masks.data  # type: ignore[attr-defined]
        areas = masks.sum(dim=(1, 2))
        best_idx = int(areas.argmax().item())
        mask = masks[best_idx].cpu().numpy()
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        crop = crop_from_mask(image, (mask > 0.5).astype(np.uint8), config.padding)
        if crop is not None:
            ensure_dir(save_path.parent)
            cv2.imwrite(str(save_path), crop)
            return True
        LOGGER.warning("Mask empty for %s", image_path)
        return False

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best_idx = int(np.argmax(areas))
        crop = crop_from_box(image, boxes[best_idx], config.padding)
        ensure_dir(save_path.parent)
        cv2.imwrite(str(save_path), crop)
        return True

    LOGGER.warning("No masks or boxes detected for %s", image_path)
    return False


def process_dataset(model: YOLO, input_root: Path, output_root: Path, config: CropConfig | None = None) -> None:
    cfg = config or CropConfig()
    for split_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        for img_path in _iter_split_images(split_dir, cfg.extensions):
            rel_path = img_path.relative_to(input_root)
            save_path = output_root / rel_path
            processed = process_image(model, img_path, save_path, cfg)
            if not processed:
                LOGGER.warning("Failed to process %s", img_path)


def _iter_split_images(split_dir: Path, extensions: Iterable[str]) -> Iterable[Path]:
    for chem_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for device_dir in sorted(p for p in chem_dir.iterdir() if p.is_dir()):
            for round_dir in sorted(p for p in device_dir.iterdir() if p.is_dir()):
                for ext in extensions:
                    yield from round_dir.glob(f"*.{ext.lstrip('.')}")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crop ROI patches using a YOLO segmentation/detection model.")
    parser.add_argument("--yolo-weights", type=Path, required=True, help="Path to the YOLO checkpoint (.pt).")
    parser.add_argument("--input-root", type=Path, required=True, help="Root folder with split images (train/val/test).")
    parser.add_argument("--output-root", type=Path, required=True, help="Destination folder for cropped ROI images.")
    parser.add_argument("--padding", type=float, default=0.10, help="Relative padding applied around the detected region.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size for YOLO.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = build_cli()
    args = parser.parse_args()
    cfg = CropConfig(padding=args.padding, conf=args.conf, imgsz=args.imgsz)
    model = YOLO(str(args.yolo_weights))
    process_dataset(model, args.input_root, args.output_root, cfg)


if __name__ == "__main__":
    main()
