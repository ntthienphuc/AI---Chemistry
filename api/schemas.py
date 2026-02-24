from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class RoiInfo(BaseModel):
    source: str = Field(..., description="mask|box|green|center")
    bbox_xyxy: Tuple[int, int, int, int] = Field(..., description="(x0,y0,x1,y1) in original image coordinates")
    padding: float = Field(..., description="relative padding applied")
    imgsz: int = Field(..., description="YOLO inference size")


class RegressionInfo(BaseModel):
    ppm: float
    ppm_ci95: Optional[Tuple[float, float]] = None
    ppm_sigma: Optional[float] = None
    method: str = "heteroscedastic_gaussian"


class PredictResponse(BaseModel):
    model: str
    predicted_chemical: str
    chemical_confidence: float
    concentration: RegressionInfo
    calib_mode: str
    roi: RoiInfo
    raw: Optional[Dict[str, Any]] = None


class ModelsResponse(BaseModel):
    available_models: List[str]
    yolo_weights: str
    device: str
    calib_mode: str
