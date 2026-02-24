from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


def _default_root() -> Path:
    # Set your project root once. Can be overridden with env AI_CHEM_ROOT.
    return Path(os.environ.get("AI_CHEM_ROOT", r"D:\Project\AI - Chemistry")).resolve()


ROOT: Path = _default_root()

# YOLO ROI detector weights (fixed, but overridable by env)
YOLO_WEIGHTS: Path = Path(os.environ.get("AI_CHEM_YOLO", str(ROOT / "weights" / "best.pt"))).resolve()

# Inference device: "cuda" if available else "cpu"
DEVICE: str = os.environ.get("AI_CHEM_DEVICE", "cuda")

# Calibration mode:
# - "greenborder" : use GreenBorderNormalizer
# - "none"        : no calibration
CALIB_MODE: str = os.environ.get("AI_CHEM_CALIB", "greenborder").lower().strip()


@dataclass(frozen=True)
class ModelSpec:
    name: str
    ckpt_rel: str
    meta_rel: Optional[str] = None

    def ckpt_path(self) -> Path:
        return (ROOT / Path(self.ckpt_rel)).resolve()

    def meta_path(self) -> Optional[Path]:
        if not self.meta_rel:
            return None
        return (ROOT / Path(self.meta_rel)).resolve()


# Mapping: query param ?model=<key> -> checkpoint/meta
MODEL_ZOO: Dict[str, ModelSpec] = {
    "convnext10k": ModelSpec(
        name="convnext10k",
        ckpt_rel=r"weights\paper_lab10k\ConvNext_seed0_l2.0.pt",
        meta_rel=r"weights\paper_lab10k\ConvNext_seed0_l2.0.meta.json",
    ),
    "effb010k": ModelSpec(
        name="effb010k",
        ckpt_rel=r"weights\paper_lab10k\EffB0_seed0_l2.0.pt",
        meta_rel=None,
    ),
    "nfnet10k": ModelSpec(
        name="nfnet10k",
        ckpt_rel=r"weights\paper_lab10k\NFNet_seed0_l2.0.pt",
        meta_rel=None,
    ),
    "tfb310k": ModelSpec(
        name="tfb310k",
        ckpt_rel=r"weights\paper_lab10k\TFB3_seed0_l2.0.pt",
        meta_rel=None,
    ),
    "convnext3k": ModelSpec(
        name="convnext3k",
        ckpt_rel=r"weights\paper_field3k\ConvNext_field3k_seed0_l2.0_bs24.pt",
        meta_rel=None,
    ),
    "effb03k": ModelSpec(
        name="effb03k",
        ckpt_rel=r"weights\paper_field3k\EffB0_field3k_seed0_l2.0_bs24.pt",
        meta_rel=None,
    ),
    "nfnet3k": ModelSpec(
        name="nfnet3k",
        ckpt_rel=r"weights\paper_field3k\DMNFNet_field3k_seed0_l2.0_bs24.pt",
        meta_rel=None,
    ),
    "tfb33k": ModelSpec(
        name="tfb33k",
        ckpt_rel=r"weights\paper_field3k\TFB3_field3k_seed0_l2.0_bs24.pt",
        meta_rel=None,
    ),
}
