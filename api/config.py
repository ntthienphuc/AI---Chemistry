from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


# =========================
# Fixed project configuration
# =========================
ROOT: Path = Path(r"D:\Project\AI - Chemistry")

# YOLO ROI detector weights
# Keep this path if your detector is still stored here.
YOLO_WEIGHTS: Path = ROOT / "weights" / "best.pt"

# Inference device
DEVICE: str = "cuda"

# Default calibration mode used by the app
# - "greenborder" : use GreenBorderNormalizer
# - "none"        : no calibration
CALIB_MODE: str = "greenborder"
VALID_CALIB_MODES = ("greenborder", "none")


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


def _spec(name: str, folder: str, stem: str, suffix: str) -> ModelSpec:
    return ModelSpec(
        name=name,
        ckpt_rel=fr"weights\{folder}\{stem}_{suffix}.pt",
        meta_rel=fr"weights\{folder}\{stem}_{suffix}.meta.json",
    )


MODEL_ZOO_EXPLICIT: Dict[str, ModelSpec] = {
    # ===== 3K =====
    "convnext3k_green": _spec("convnext3k_green", "runs_multitask_3k", "ConvNext_seed0_l2.0", "green"),
    "convnext3k_none": _spec("convnext3k_none", "runs_multitask_3k", "ConvNext_seed0_l2.0", "none"),
    "effb03k_green": _spec("effb03k_green", "runs_multitask_3k", "EffB0_seed0_l2.0", "green"),
    "effb03k_none": _spec("effb03k_none", "runs_multitask_3k", "EffB0_seed0_l2.0", "none"),
    "mnv33k_green": _spec("mnv33k_green", "runs_multitask_3k", "MNV3_seed0_l2.0", "green"),
    "mnv33k_none": _spec("mnv33k_none", "runs_multitask_3k", "MNV3_seed0_l2.0", "none"),
    "nfnet3k_green": _spec("nfnet3k_green", "runs_multitask_3k", "NFNet_seed0_l2.0", "green"),
    "nfnet3k_none": _spec("nfnet3k_none", "runs_multitask_3k", "NFNet_seed0_l2.0", "none"),
    "swint3k_green": _spec("swint3k_green", "runs_multitask_3k", "SwinT_seed0_l2.0", "green"),
    "swint3k_none": _spec("swint3k_none", "runs_multitask_3k", "SwinT_seed0_l2.0", "none"),
    "tfb33k_green": _spec("tfb33k_green", "runs_multitask_3k", "TFB3_seed0_l2.0", "green"),
    "tfb33k_none": _spec("tfb33k_none", "runs_multitask_3k", "TFB3_seed0_l2.0", "none"),

    # ===== 10K =====
    "convnext10k_green": _spec("convnext10k_green", "runs_multitask_10k", "ConvNext_seed0_l2.0", "green"),
    "convnext10k_none": _spec("convnext10k_none", "runs_multitask_10k", "ConvNext_seed0_l2.0", "none"),
    "effb010k_green": _spec("effb010k_green", "runs_multitask_10k", "EffB0_seed0_l2.0", "green"),
    "effb010k_none": _spec("effb010k_none", "runs_multitask_10k", "EffB0_seed0_l2.0", "none"),
    "mnv310k_green": _spec("mnv310k_green", "runs_multitask_10k", "MNV3_seed0_l2.0", "green"),
    "mnv310k_none": _spec("mnv310k_none", "runs_multitask_10k", "MNV3_seed0_l2.0", "none"),
    "nfnet10k_green": _spec("nfnet10k_green", "runs_multitask_10k", "NFNet_seed0_l2.0", "green"),
    "nfnet10k_none": _spec("nfnet10k_none", "runs_multitask_10k", "NFNet_seed0_l2.0", "none"),
    "swint10k_green": _spec("swint10k_green", "runs_multitask_10k", "SwinT_seed0_l2.0", "green"),
    "swint10k_none": _spec("swint10k_none", "runs_multitask_10k", "SwinT_seed0_l2.0", "none"),
    "tfb310k_green": _spec("tfb310k_green", "runs_multitask_10k", "TFB3_seed0_l2.0", "green"),
    "tfb310k_none": _spec("tfb310k_none", "runs_multitask_10k", "TFB3_seed0_l2.0", "none"),

    # ===== 13K =====
    "convnext13k_green": _spec("convnext13k_green", "runs_multitask_13k", "ConvNext_seed0_l2.0", "green"),
    "convnext13k_none": _spec("convnext13k_none", "runs_multitask_13k", "ConvNext_seed0_l2.0", "none"),
    "effb013k_green": _spec("effb013k_green", "runs_multitask_13k", "EffB0_seed0_l2.0", "green"),
    "effb013k_none": _spec("effb013k_none", "runs_multitask_13k", "EffB0_seed0_l2.0", "none"),
    "mnv313k_green": _spec("mnv313k_green", "runs_multitask_13k", "MNV3_seed0_l2.0", "green"),
    "mnv313k_none": _spec("mnv313k_none", "runs_multitask_13k", "MNV3_seed0_l2.0", "none"),
    "nfnet13k_green": _spec("nfnet13k_green", "runs_multitask_13k", "NFNet_seed0_l2.0", "green"),
    "nfnet13k_none": _spec("nfnet13k_none", "runs_multitask_13k", "NFNet_seed0_l2.0", "none"),
    "swint13k_green": _spec("swint13k_green", "runs_multitask_13k", "SwinT_seed0_l2.0", "green"),
    "swint13k_none": _spec("swint13k_none", "runs_multitask_13k", "SwinT_seed0_l2.0", "none"),
    "tfb313k_green": _spec("tfb313k_green", "runs_multitask_13k", "TFB3_seed0_l2.0", "green"),
    "tfb313k_none": _spec("tfb313k_none", "runs_multitask_13k", "TFB3_seed0_l2.0", "none"),
}


# Backward-compatible default aliases:
# Keep old query keys working and point them to GREEN versions by default.
MODEL_ZOO: Dict[str, ModelSpec] = {
    # old-style keys
    "convnext3k": MODEL_ZOO_EXPLICIT["convnext3k_green"],
    "effb03k": MODEL_ZOO_EXPLICIT["effb03k_green"],
    "mnv33k": MODEL_ZOO_EXPLICIT["mnv33k_green"],
    "nfnet3k": MODEL_ZOO_EXPLICIT["nfnet3k_green"],
    "swint3k": MODEL_ZOO_EXPLICIT["swint3k_green"],
    "tfb33k": MODEL_ZOO_EXPLICIT["tfb33k_green"],

    "convnext10k": MODEL_ZOO_EXPLICIT["convnext10k_green"],
    "effb010k": MODEL_ZOO_EXPLICIT["effb010k_green"],
    "mnv310k": MODEL_ZOO_EXPLICIT["mnv310k_green"],
    "nfnet10k": MODEL_ZOO_EXPLICIT["nfnet10k_green"],
    "swint10k": MODEL_ZOO_EXPLICIT["swint10k_green"],
    "tfb310k": MODEL_ZOO_EXPLICIT["tfb310k_green"],

    "convnext13k": MODEL_ZOO_EXPLICIT["convnext13k_green"],
    "effb013k": MODEL_ZOO_EXPLICIT["effb013k_green"],
    "mnv313k": MODEL_ZOO_EXPLICIT["mnv313k_green"],
    "nfnet13k": MODEL_ZOO_EXPLICIT["nfnet13k_green"],
    "swint13k": MODEL_ZOO_EXPLICIT["swint13k_green"],
    "tfb313k": MODEL_ZOO_EXPLICIT["tfb313k_green"],

    # explicit keys
    **MODEL_ZOO_EXPLICIT,
}
