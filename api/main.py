"""FastAPI service exposing the NH4/NO2 inference pipeline."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ai_chemistry.inference.pipeline import ChemistryPipeline

app = FastAPI(title="AI Chemistry Inference", version="1.0.0")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
YOLO_WEIGHTS = WEIGHTS_DIR / "best.pt"


def discover_classifier_weights() -> Dict[str, Path]:
    if not WEIGHTS_DIR.exists():
        raise RuntimeError(f"Weights directory not found: {WEIGHTS_DIR}")
    weights: Dict[str, Path] = {}
    for path in WEIGHTS_DIR.glob("*.pt"):
        if path.name == YOLO_WEIGHTS.name:
            continue
        weights[path.stem] = path
    if not weights:
        raise RuntimeError("No classifier weights (*.pt) found in the weights directory.")
    return weights


CLASSIFIER_WEIGHTS = discover_classifier_weights()
PIPELINE_CACHE: Dict[str, ChemistryPipeline] = {}


class ModelSummary(BaseModel):
    name: str
    path: str


class PredictionResponse(BaseModel):
    model: str
    chemical: str
    ppm: float
    confidence: float
    ppm_scaled: float
    sigma: float


def get_pipeline(model_name: str) -> ChemistryPipeline:
    if model_name not in CLASSIFIER_WEIGHTS:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
    if model_name not in PIPELINE_CACHE:
        classifier_path = CLASSIFIER_WEIGHTS[model_name]
        if not YOLO_WEIGHTS.exists():
            raise HTTPException(status_code=500, detail=f"YOLO weights not found at {YOLO_WEIGHTS}")
        PIPELINE_CACHE[model_name] = ChemistryPipeline(
            yolo_weights=YOLO_WEIGHTS,
            classifier_weights=classifier_path,
        )
    return PIPELINE_CACHE[model_name]


def decode_image(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    return image


@app.get("/models", response_model=List[ModelSummary])
def list_models() -> List[ModelSummary]:
    return [ModelSummary(name=name, path=str(path)) for name, path in CLASSIFIER_WEIGHTS.items()]


@app.post("/predict", response_model=PredictionResponse)
def predict(
    file: UploadFile = File(...),
    model_name: str = Form(list(CLASSIFIER_WEIGHTS.keys())[0]),
) -> PredictionResponse:
    pipeline = get_pipeline(model_name)
    image = decode_image(file)
    try:
        result = pipeline.predict_array(image)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    response = PredictionResponse(
        model=model_name,
        chemical=result["chemical"],
        ppm=result["ppm"],
        confidence=result["confidence"],
        ppm_scaled=result["ppm_scaled"],
        sigma=result["sigma"],
    )
    return response


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {
            "message": "AI Chemistry inference API is running.",
            "available_models": list(CLASSIFIER_WEIGHTS.keys()),
        }
    )
