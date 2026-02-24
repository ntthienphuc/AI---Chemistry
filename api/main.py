# api/main.py
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO

from .config import CALIB_MODE, DEVICE, MODEL_ZOO, YOLO_WEIGHTS
from .predictor import LoadedPredictor
from .roi import (
    YoloRoiConfig,
    GreenRoiConfig,
    crop_roi_auto,
)
from .schemas import ModelsResponse, PredictResponse, RegressionInfo, RoiInfo

app = FastAPI(title="AI-Chemistry API", version="1.1.0")


@lru_cache(maxsize=1)
def get_yolo() -> YOLO | None:
    if not YOLO_WEIGHTS.exists():
        # Không có YOLO weights thì vẫn chạy fallback green/center
        return None
    return YOLO(str(YOLO_WEIGHTS))


@lru_cache(maxsize=16)
def get_predictor(model_key: str) -> LoadedPredictor:
    if model_key not in MODEL_ZOO:
        raise KeyError(model_key)
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
        calib_mode=CALIB_MODE,
    )


@app.get("/models", response_model=ModelsResponse)
def list_models():
    return ModelsResponse(
        available_models=sorted(list(MODEL_ZOO.keys())),
        yolo_weights=str(YOLO_WEIGHTS),
        device=DEVICE,
        calib_mode=CALIB_MODE,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    model: str = Query(..., description="Tên model, ví dụ: convnext10k"),
    roi_mode: str = Query("auto", description="auto|yolo|green|center"),
    debug: bool = Query(False, description="Trả raw probs/mu/logvar nếu true"),
    file: UploadFile = File(...),
):
    if model not in MODEL_ZOO:
        raise HTTPException(status_code=400, detail=f"Model không hợp lệ: {model}. Xem /models")

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
        pred = get_predictor(model).predict(roi_bgr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    resp = PredictResponse(
        model=model,
        predicted_chemical=pred.chemical,
        chemical_confidence=pred.chemical_conf,
        concentration=RegressionInfo(ppm=pred.ppm, ppm_ci95=pred.ppm_ci95, ppm_sigma=pred.ppm_sigma),
        calib_mode=CALIB_MODE,
        roi=RoiInfo(source=source, bbox_xyxy=bbox, padding=padding, imgsz=imgsz),
        raw=pred.raw if debug else None,
    )
    return JSONResponse(content=resp.model_dump())