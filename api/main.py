# api/main.py
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO

from .config import (
    CALIB_MODE,
    DEVICE,
    MODEL_ZOO,
    VALID_CALIB_MODES,
    VALID_DATA_TYPES,
    VALID_MODEL_TYPES,
    VALID_TRAIN_CALIBS,
    YOLO_WEIGHTS,
)
from .predictor import LoadedPredictor
from .roi import (
    YoloRoiConfig,
    GreenRoiConfig,
    crop_roi_auto,
)
from .schemas import ModelsResponse, PredictResponse, RegressionInfo, RoiInfo

app = FastAPI(title="AI-Chemistry API", version="1.1.0")


@lru_cache(maxsize=1)
def _load_yolo(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def get_yolo() -> YOLO | None:
    if not YOLO_WEIGHTS.exists():
        # Không có YOLO weights thì vẫn chạy fallback green/center
        return None
    return _load_yolo(str(YOLO_WEIGHTS))


def normalize_calib_mode(calib_mode: Optional[str]) -> str:
    mode = (calib_mode or CALIB_MODE).lower().strip()
    if mode not in VALID_CALIB_MODES:
        raise ValueError(mode)
    return mode


def infer_calib_mode_from_model(model_key: str) -> str:
    key = (model_key or "").lower().strip()
    if key.endswith("_none"):
        return "none"
    if key.endswith("_green"):
        return "greenborder"
    return CALIB_MODE


def resolve_model_key(
    model: Optional[str],
    data_type: Optional[str],
    model_type: Optional[str],
    train_calib: Optional[str],
) -> str:
    if model and model.strip():
        model_key = model.lower().strip()
        if model_key not in MODEL_ZOO:
            raise ValueError(f"Model không hợp lệ: {model}. Xem /models")
        return model_key

    missing = [
        name
        for name, value in (
            ("data_type", data_type),
            ("model_type", model_type),
            ("train_calib", train_calib),
        )
        if not value or not value.strip()
    ]
    if missing:
        raise ValueError(
            "Thiếu model hoặc bộ tham số: data_type, model_type, train_calib. "
            f"Đang thiếu: {', '.join(missing)}"
        )

    data_type = data_type.lower().strip()
    model_type = model_type.lower().strip()
    train_calib = train_calib.lower().strip()

    if data_type not in VALID_DATA_TYPES:
        raise ValueError(f"data_type không hợp lệ: {data_type}. Dùng: {'|'.join(VALID_DATA_TYPES)}")
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(f"model_type không hợp lệ: {model_type}. Dùng: {'|'.join(VALID_MODEL_TYPES)}")
    if train_calib not in VALID_TRAIN_CALIBS:
        raise ValueError(f"train_calib không hợp lệ: {train_calib}. Dùng: {'|'.join(VALID_TRAIN_CALIBS)}")

    model_key = f"{model_type}{data_type}_{train_calib}"
    if model_key not in MODEL_ZOO:
        raise ValueError(f"Không tìm thấy model key được build: {model_key}. Xem /models")
    return model_key


@lru_cache(maxsize=32)
def get_predictor(model_key: str, calib_mode: str) -> LoadedPredictor:
    if model_key not in MODEL_ZOO:
        raise KeyError(model_key)
    calib_mode = normalize_calib_mode(calib_mode)
    spec = MODEL_ZOO[model_key]
    ckpt_path = spec.ckpt_path()
    meta_path = spec.meta_path()
    if not ckpt_path.exists():
        raise RuntimeError(f"Không tìm thấy checkpoint: {ckpt_path}")
    if meta_path is not None and not meta_path.exists():
        meta_path = None
    return LoadedPredictor(
        ckpt_path=ckpt_path,
        meta_path=meta_path,
        device=DEVICE,
        calib_mode=calib_mode,
    )


@app.get("/models", response_model=ModelsResponse)
def list_models():
    return ModelsResponse(
        available_models=sorted(list(MODEL_ZOO.keys())),
        available_data_types=list(VALID_DATA_TYPES),
        available_model_types=list(VALID_MODEL_TYPES),
        available_train_calibs=list(VALID_TRAIN_CALIBS),
        available_calib_modes=list(VALID_CALIB_MODES),
        yolo_weights=str(YOLO_WEIGHTS),
        device=DEVICE,
        calib_mode=CALIB_MODE,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    model: Optional[str] = Query(None, description="Full model key, ví dụ: convnext10k_green"),
    data_type: Optional[str] = Query(None, description="3k|10k|13k, dùng khi không truyền model"),
    model_type: Optional[str] = Query(None, description="convnext|effb0|mnv3|nfnet|swint|tfb3, dùng khi không truyền model"),
    train_calib: Optional[str] = Query(None, description="green|none, dùng khi không truyền model"),
    roi_mode: str = Query("auto", description="auto|yolo|green|center"),
    calib_mode: Optional[str] = Query(None, description="greenborder|none; bỏ trống thì suy ra từ model/train_calib"),
    debug: bool = Query(False, description="Trả raw probs/mu/logvar nếu true"),
    file: UploadFile = File(...),
):
    try:
        model_key = resolve_model_key(model, data_type, model_type, train_calib)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        calib_mode = normalize_calib_mode(calib_mode or infer_calib_mode_from_model(model_key))
    except ValueError:
        valid = "|".join(VALID_CALIB_MODES)
        raise HTTPException(status_code=400, detail=f"calib_mode không hợp lệ: {calib_mode}. Dùng: {valid}")

    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Ảnh upload không đọc được (cv2.imdecode failed).")

    # ROI configs (bạn có thể tune ở đây nếu muốn)
    yolo_cfg = YoloRoiConfig(padding=0.10, conf=0.25, imgsz=640)
    green_cfg = GreenRoiConfig(
        ratio_center=0.75,
        hsv_lower=(35, 40, 40),
        hsv_upper=(95, 255, 255),
        pad_ratio=0.15,
        min_area_ratio=0.002,
    )

    # 1) ROI: YOLO -> green -> center (giữ viền xanh khi green)
    try:
        roi_bgr, source, bbox, padding, imgsz = crop_roi_auto(
            yolo=get_yolo(),
            image_bgr=img,
            yolo_cfg=yolo_cfg,
            green_cfg=green_cfg,
            mode=roi_mode,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"ROI error: {e}")

    # 2) Predict
    try:
        pred = get_predictor(model_key, calib_mode).predict(roi_bgr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    resp = PredictResponse(
        model=model_key,
        predicted_chemical=pred.chemical,
        chemical_confidence=pred.chemical_conf,
        concentration=RegressionInfo(ppm=pred.ppm, ppm_ci95=pred.ppm_ci95, ppm_sigma=pred.ppm_sigma),
        calib_mode=calib_mode,
        roi=RoiInfo(source=source, bbox_xyxy=bbox, padding=padding, imgsz=imgsz),
        raw=pred.raw if debug else None,
    )
    return JSONResponse(content=resp.model_dump())
