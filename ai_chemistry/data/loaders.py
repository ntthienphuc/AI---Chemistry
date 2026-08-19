# -*- coding: utf-8 -*-
"""
PyTorch Dataset & DataLoader utilities for AI-Chemistry.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ai_chemistry.preprocessing import (
    IdentityNormalizer,
    GreenBorderNormalizer,
    get_normalizer,
    make_train_transform,
    make_eval_transform,
)

SPLIT_NAMES = ("train", "val", "test")
REQUIRED_LABEL_COLUMNS = {"path", "chemical", "ppm"}


def scale_ppm(
    ppm: float,
    ppm_scale: str = "log1p",
    ppm_min: Optional[float] = None,
    ppm_max: Optional[float] = None,
) -> float:
    """Scale concentration values to target regression space."""
    val = float(ppm)
    if ppm_scale == "log1p":
        return float(math.log1p(val))
    if ppm_scale == "minmax":
        if ppm_min is None or ppm_max is None:
            raise ValueError("minmax scaling requires ppm_min and ppm_max")
        return float((val - ppm_min) / (ppm_max - ppm_min + 1e-12))
    return val


def inverse_scale_ppm(
    y_scaled: Union[float, np.ndarray],
    ppm_scale: str = "log1p",
    ppm_min: Optional[float] = None,
    ppm_max: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """Invert scaled predictions back to physical concentration (ppm / mg/L)."""
    if isinstance(y_scaled, (np.ndarray, list)):
        arr = np.asarray(y_scaled, dtype=np.float64)
        if ppm_scale == "log1p":
            return np.expm1(arr)
        if ppm_scale == "minmax" and ppm_min is not None and ppm_max is not None:
            return arr * (ppm_max - ppm_min) + ppm_min
        return arr

    val = float(y_scaled)
    if ppm_scale == "log1p":
        return float(math.expm1(val))
    if ppm_scale == "minmax" and ppm_min is not None and ppm_max is not None:
        return float(val * (ppm_max - ppm_min) + ppm_min)
    return val


def normalise_label_path(path_value: Union[str, Path]) -> Path:
    """Return a safe platform-agnostic relative path."""
    raw = str(path_value).strip().replace("\\", "/")
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if not parts or ".." in parts or Path(raw).is_absolute():
        raise ValueError(f"Invalid relative image path in labels: {path_value!r}")
    return Path(*parts)


def prepare_split_frame(df: pd.DataFrame, split_name: str, csv_path: Path) -> pd.DataFrame:
    missing = REQUIRED_LABEL_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(sorted(missing))}")
    if df.empty:
        raise ValueError(f"{csv_path} contains zero records.")

    frame = df.copy()
    frame["chemical"] = frame["chemical"].astype(str).str.strip().str.upper()
    invalid_chems = set(frame["chemical"]) - {"NH4", "NO2"}
    if invalid_chems:
        raise ValueError(f"{csv_path} contains invalid chemical classes: {invalid_chems}")

    frame["ppm"] = pd.to_numeric(frame["ppm"], errors="coerce")
    invalid_ppm = frame["ppm"].isna() | ~np.isfinite(frame["ppm"]) | (frame["ppm"] < 0)
    if invalid_ppm.any():
        raise ValueError(f"{csv_path} contains {int(invalid_ppm.sum())} invalid/negative ppm values.")

    frame["split"] = split_name
    return frame.reset_index(drop=True)


def load_publication_splits(
    manifests_dir: Union[str, Path],
    dataset: str,
    images_root: Optional[Union[str, Path]] = None,
) -> Tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the frozen train/val/test manifests for dataset ('3k', '10k', '13k').
    """
    m_dir = Path(manifests_dir).resolve()
    ds_dir = m_dir / dataset.lower().strip()
    if not ds_dir.is_dir():
        # Fallback check if manifests_dir is data/
        if (m_dir / "manifests" / dataset).is_dir():
            ds_dir = m_dir / "manifests" / dataset

    if not ds_dir.is_dir():
        raise FileNotFoundError(f"Manifests folder not found for dataset '{dataset}' at: {ds_dir}")

    frames: Dict[str, pd.DataFrame] = {}
    for split in SPLIT_NAMES:
        p = ds_dir / f"{split}.csv"
        if not p.is_file():
            raise FileNotFoundError(f"Missing manifest file: {p}")
        frames[split] = prepare_split_frame(pd.read_csv(p), split, p)

    img_root = Path(images_root).resolve() if images_root else Path(".")
    return img_root, frames["train"], frames["val"], frames["test"]


class ChemistryDataset(Dataset):
    """
    Dataset for inorganic nitrogen strip colorimetry.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_root: Union[str, Path],
        class_to_idx: Optional[Dict[str, int]] = None,
        ppm_scale: str = "log1p",
        transform=None,
        normalizer: Optional[Union[IdentityNormalizer, GreenBorderNormalizer]] = None,
        ppm_min: Optional[float] = None,
        ppm_max: Optional[float] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.images_root = Path(images_root)
        self.class_to_idx = class_to_idx or {"NH4": 0, "NO2": 1}
        self.ppm_scale = ppm_scale
        self.transform = transform
        self.normalizer = normalizer if normalizer is not None else IdentityNormalizer()
        self.ppm_min = ppm_min
        self.ppm_max = ppm_max

        if self.ppm_scale == "minmax":
            if self.ppm_min is None or self.ppm_max is None:
                self.ppm_min = float(self.df["ppm"].min())
                self.ppm_max = float(self.df["ppm"].max())

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float, str]:
        row = self.df.iloc[idx]
        rel_path = normalise_label_path(row["path"])
        img_path = self.images_root / rel_path

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image file at: {img_path}")

        # 1. Color normalization
        rgb01 = self.normalizer(img_bgr)

        # 2. Augmentations / Tensor conversion
        if self.transform is not None:
            out = self.transform(image=rgb01)
            img_t = out["image"]
        else:
            img_t = torch.from_numpy(rgb01.transpose(2, 0, 1)).float()

        chem = str(row["chemical"]).strip().upper()
        chem_idx = int(self.class_to_idx[chem])
        ppm_scaled = scale_ppm(float(row["ppm"]), self.ppm_scale, self.ppm_min, self.ppm_max)

        return img_t, chem_idx, float(ppm_scaled), str(img_path)