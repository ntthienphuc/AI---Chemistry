# api/roi.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class YoloRoiConfig:
    padding: float = 0.10
    conf: float = 0.25
    imgsz: int = 640


@dataclass
class GreenRoiConfig:
    """
    Green fallback (GIỮ VIỀN XANH) nhưng chỉ crop THÔ vừa đủ để YOLO refine lại.
    Tight HSV để tránh bắt nhầm nền lưới/texture xám.
    """
    ratio_center: float = 0.75

    # tăng S,V để loại nền xám; H vẫn giữ vùng xanh
    hsv_lower: Tuple[int, int, int] = (35, 80, 80)
    hsv_upper: Tuple[int, int, int] = (95, 255, 255)

    pad_ratio: float = 0.05
    min_area_ratio: float = 0.002

    morph_kernel: int = 5
    morph_open_iter: int = 1
    morph_close_iter: int = 1


def _clip_xyxy(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(1, min(w, x1))
    y1 = max(1, min(h, y1))
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return x0, y0, x1, y1


def _center_crop_rect(img: np.ndarray, ratio: float) -> Tuple[np.ndarray, int, int]:
    """
    Crop vùng giữa ảnh (không resize) để giảm nhiễu background.
    Return: sub_img, offset_x, offset_y (tọa độ sub trong ảnh gốc)
    """
    h, w = img.shape[:2]
    ratio = float(ratio)
    crop_w = int(w * ratio)
    crop_h = int(h * ratio)
    cx, cy = w // 2, h // 2
    x0 = max(0, cx - crop_w // 2)
    y0 = max(0, cy - crop_h // 2)
    x1 = min(w, cx + crop_w // 2)
    y1 = min(h, cy + crop_h // 2)
    sub = img[y0:y1, x0:x1].copy()
    return sub, x0, y0


def _crop_from_bbox(img: np.ndarray, bbox_xyxy: Tuple[int, int, int, int], pad_ratio: float) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = bbox_xyxy
    w_box = max(1, x1 - x0)
    h_box = max(1, y1 - y0)
    pad_x = int(w_box * pad_ratio)
    pad_y = int(h_box * pad_ratio)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = _clip_xyxy(x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y, w, h)
    return img[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def crop_roi_with_yolo(
    yolo: YOLO,
    image_bgr: np.ndarray,
    cfg: Optional[YoloRoiConfig] = None,
) -> Tuple[np.ndarray, str, Tuple[int, int, int, int], float, int]:
    """
    YOLO crop: mask-first, box-fallback.
    Returns: (roi_bgr, source, bbox_xyxy, padding, imgsz)
    Raises ValueError if no ROI found.
    """
    if cfg is None:
        cfg = YoloRoiConfig()

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    preds = yolo.predict(source=rgb, conf=cfg.conf, imgsz=cfg.imgsz, verbose=False)
    r = preds[0]

    # 1) Prefer masks
    if getattr(r, "masks", None) is not None and r.masks is not None:
        masks = r.masks.data  # (N,H,W)
        if masks is not None and len(masks) > 0:
            areas = masks.sum(dim=(1, 2))
            best_idx = int(areas.argmax().item())
            mask = masks[best_idx].detach().cpu().numpy()

            # resize mask về size ảnh input nếu cần
            if mask.shape[:2] != image_bgr.shape[:2]:
                mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

            ys, xs = np.where(mask > 0.5)
            if xs.size > 0 and ys.size > 0:
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                roi, bbox = _crop_from_bbox(image_bgr, (x0, y0, x1, y1), cfg.padding)
                return roi, "mask", bbox, float(cfg.padding), int(cfg.imgsz)

    # 2) Fallback to boxes
    if getattr(r, "boxes", None) is not None and r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.detach().cpu().numpy().astype(int)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best_idx = int(np.argmax(areas))
        x0, y0, x1, y1 = map(int, boxes[best_idx])
        roi, bbox = _crop_from_bbox(image_bgr, (x0, y0, x1, y1), cfg.padding)
        return roi, "box", bbox, float(cfg.padding), int(cfg.imgsz)

    raise ValueError("YOLO không phát hiện ROI (không có mask/box).")


def crop_roi_by_green_border(
    image_bgr: np.ndarray,
    cfg: Optional[GreenRoiConfig] = None,
) -> Tuple[np.ndarray, str, Tuple[int, int, int, int], float, int]:
    """
    GREEN fallback: detect component xanh (viền xanh) rồi crop bbox + padding nhỏ.
    Đây là crop THÔ (coarse) để YOLO refine lần 2.
    Returns: (roi_bgr, source="green", bbox_xyxy, pad_ratio, imgsz=0)
    """
    if cfg is None:
        cfg = GreenRoiConfig()

    # center crop giảm nhiễu nền
    sub, offx, offy = _center_crop_rect(image_bgr, cfg.ratio_center)

    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(cfg.hsv_lower, np.uint8),
        np.array(cfg.hsv_upper, np.uint8),
    )

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=cfg.morph_open_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=cfg.morph_close_iter)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        raise ValueError("Green fallback: không thấy vùng xanh (no components).")

    sub_h, sub_w = sub.shape[:2]
    cx, cy = sub_w / 2.0, sub_h / 2.0

    min_area = max(200, int(cfg.min_area_ratio * sub_w * sub_h))

    # chọn component "đúng kit": gần tâm + bbox gần vuông + area lớn
    best_i = None
    best_score = -1e18

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        area = int(area)
        if area < min_area:
            continue

        ar = w / (h + 1e-6)
        if ar < 0.6 or ar > 1.6:
            continue

        ux, uy = centroids[i]
        dist2 = (ux - cx) ** 2 + (uy - cy) ** 2
        square_penalty = abs(np.log(ar))  # 0 tốt nhất

        score = area - 0.002 * dist2 - 6000.0 * square_penalty
        if score > best_score:
            best_score = score
            best_i = i

    if best_i is None:
        raise ValueError("Green fallback: không chọn được component hợp lệ (component filter failed).")

    x, y, w, h, area = stats[best_i]

    # bbox về tọa độ ảnh gốc
    x0 = int(x + offx)
    y0 = int(y + offy)
    x1 = int(x + w + offx)
    y1 = int(y + h + offy)

    roi, bbox = _crop_from_bbox(image_bgr, (x0, y0, x1, y1), cfg.pad_ratio)
    return roi, "green", bbox, float(cfg.pad_ratio), 0


def crop_roi_center(
    image_bgr: np.ndarray,
    ratio: float = 0.75,
) -> Tuple[np.ndarray, str, Tuple[int, int, int, int], float, int]:
    """
    Fallback cuối: center crop.
    Returns: (roi, "center", bbox_xyxy, padding=0, imgsz=0)
    """
    sub, x0, y0 = _center_crop_rect(image_bgr, ratio)
    bbox = (x0, y0, x0 + sub.shape[1], y0 + sub.shape[0])
    return sub, "center", bbox, 0.0, 0


def crop_roi_auto(
    yolo: Optional[YOLO],
    image_bgr: np.ndarray,
    yolo_cfg: Optional[YoloRoiConfig] = None,
    green_cfg: Optional[GreenRoiConfig] = None,
    mode: str = "auto",
) -> Tuple[np.ndarray, str, Tuple[int, int, int, int], float, int]:
    """
    mode:
      - "yolo": chỉ YOLO
      - "green": chỉ green fallback
      - "center": chỉ center crop
      - "auto": YOLO(full) -> green(coarse) -> YOLO refine -> green -> center

    Lý do: model bạn train theo YOLO-tight ROI, nên GREEN nên dùng để "cứu YOLO", không nên feed thẳng vào model nếu crop quá rộng.
    """
    mode = (mode or "auto").lower().strip()

    if mode == "center":
        return crop_roi_center(image_bgr)

    if mode == "green":
        return crop_roi_by_green_border(image_bgr, green_cfg)

    if mode == "yolo":
        if yolo is None:
            raise ValueError("YOLO model is None (missing weights).")
        return crop_roi_with_yolo(yolo, image_bgr, yolo_cfg)

    # -------- AUTO --------
    # (1) Try YOLO on full image
    if yolo is not None:
        try:
            return crop_roi_with_yolo(yolo, image_bgr, yolo_cfg)
        except Exception:
            pass

    # (2) Green coarse crop
    try:
        roi_g, src_g, bbox_g, pad_g, _ = crop_roi_by_green_border(image_bgr, green_cfg)
    except Exception:
        # (5) last resort
        return crop_roi_center(image_bgr)

    # (3) YOLO refine inside green ROI (lower conf to catch weak detections)
    if yolo is not None:
        try:
            base_cfg = yolo_cfg or YoloRoiConfig()
            refine_cfg = YoloRoiConfig(
                padding=base_cfg.padding,
                conf=min(base_cfg.conf, 0.15),
                imgsz=base_cfg.imgsz,
            )

            roi_r, src_r, bbox_r_rel, pad_r, imgsz_r = crop_roi_with_yolo(yolo, roi_g, refine_cfg)

            # Map bbox from roi_g coords -> original coords
            xg0, yg0, _, _ = bbox_g
            bbox_final = (
                xg0 + bbox_r_rel[0],
                yg0 + bbox_r_rel[1],
                xg0 + bbox_r_rel[2],
                yg0 + bbox_r_rel[3],
            )
            return roi_r, f"{src_r}_refine", bbox_final, pad_r, imgsz_r
        except Exception:
            # refine fail -> use green coarse
            return roi_g, src_g, bbox_g, pad_g, 0

    # If no yolo, return green coarse
    return roi_g, src_g, bbox_g, pad_g, 0