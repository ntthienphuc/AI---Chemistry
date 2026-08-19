#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication Data & Manifest Integrity Validator.

Validates:
1. Expected row counts for 3K, 10K, and 13K frozen splits.
2. Required columns: ('path', 'chemical', 'ppm').
3. Chemical labels strictly within {'NH4', 'NO2'}.
4. Non-negative, finite numeric concentration values.
5. Zero leakage (no identical relative image paths across train, val, and test splits).
6. Optional physical image existence verification under specified image root directory.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


EXPECTED_COUNTS: Dict[str, Dict[str, int]] = {
    "3k": {"train": 2064, "val": 476, "test": 301},
    "10k": {"train": 7227, "val": 1795, "test": 900},
    "13k": {"train": 7495, "val": 2095, "test": 901},
}

REQUIRED_COLUMNS = {"path", "chemical", "ppm"}
VALID_CHEMICALS = {"NH4", "NO2"}


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_split_file(
    csv_path: Path,
    dataset: str,
    split: str,
    images_root: Optional[Path] = None,
) -> Tuple[bool, pd.DataFrame, List[str]]:
    errors: List[str] = []
    if not csv_path.is_file():
        return False, pd.DataFrame(), [f"File not found: {csv_path}"]

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, pd.DataFrame(), [f"Failed to read CSV {csv_path}: {e}"]

    # 1. Required columns
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns {missing_cols} in {csv_path}")

    # 2. Row count
    expected = EXPECTED_COUNTS[dataset][split]
    actual = len(df)
    if actual != expected:
        errors.append(f"Row count mismatch for {dataset}/{split}: expected {expected}, got {actual}")

    # 3. Valid chemicals
    if "chemical" in df.columns:
        chemicals = set(df["chemical"].astype(str).str.strip().str.upper())
        invalid_chems = chemicals - VALID_CHEMICALS
        if invalid_chems:
            errors.append(f"Invalid chemicals {invalid_chems} in {csv_path}")

    # 4. Valid PPM values
    if "ppm" in df.columns:
        ppm = pd.to_numeric(df["ppm"], errors="coerce")
        invalid_ppm = ppm.isna() | ~np.isfinite(ppm) | (ppm < 0)
        if invalid_ppm.any():
            errors.append(f"Found {int(invalid_ppm.sum())} invalid or negative ppm entries in {csv_path}")

    # 5. Image existence check if images_root provided
    if images_root is not None and images_root.is_dir() and "path" in df.columns:
        missing_images = []
        for rel_path in df["path"].astype(str):
            full_path = images_root / rel_path.strip().replace("\\", "/")
            if not full_path.is_file():
                missing_images.append(str(rel_path))
                if len(missing_images) >= 5:
                    break
        if missing_images:
            errors.append(
                f"Referenced images not found in {images_root}. First missing: {', '.join(missing_images)}"
            )

    return len(errors) == 0, df, errors


def validate_dataset(
    manifests_dir: Path,
    dataset: str,
    images_root: Optional[Path] = None,
) -> Tuple[bool, List[str]]:
    ds_dir = manifests_dir / dataset
    errors: List[str] = []
    frames: Dict[str, pd.DataFrame] = {}

    print(f"\n--- Validating Dataset: {dataset.upper()} ---")
    if not ds_dir.is_dir():
        return False, [f"Manifests directory not found: {ds_dir}"]

    for split in ["train", "val", "test"]:
        csv_path = ds_dir / f"{split}.csv"
        ok, df, split_errors = validate_split_file(csv_path, dataset, split, images_root)
        errors.extend(split_errors)
        if ok:
            sha = compute_sha256(csv_path)
            print(f"  [{split.upper()}] Rows: {len(df):<5} | SHA256: {sha[:12]}...{sha[-8:]} | OK")
            frames[split] = df
        else:
            print(f"  [{split.upper()}] FAILED:")
            for err in split_errors:
                print(f"    - {err}")

    # Cross-split leakage check
    if len(frames) == 3:
        all_paths: Dict[str, str] = {}
        duplicates: List[Tuple[str, str, str]] = []
        for split, df in frames.items():
            for p in df["path"].astype(str).str.strip().str.replace("\\", "/").str.lower():
                if p in all_paths:
                    duplicates.append((p, all_paths[p], split))
                else:
                    all_paths[p] = split
        if duplicates:
            msg = f"Data leakage: {len(duplicates)} duplicate paths across splits in {dataset}."
            errors.append(msg)
            print(f"  [LEAKAGE CHECK] FAILED: {msg}")
        else:
            print(f"  [LEAKAGE CHECK] PASSED: Zero overlap between train, val, and test splits.")

    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="Validate publication dataset manifests.")
    parser.add_argument(
        "--manifests_dir",
        type=str,
        default="data/manifests",
        help="Path to root manifests directory containing 3k, 10k, 13k subdirectories.",
    )
    parser.add_argument(
        "--images_root",
        type=str,
        default=None,
        help="Optional root directory of image files to verify image existence.",
    )
    args = parser.parse_args()

    manifests_dir = Path(args.manifests_dir).resolve()
    images_root = Path(args.images_root).resolve() if args.images_root else None

    print(f"Publication Manifest Validator")
    print(f"Manifests Directory: {manifests_dir}")
    if images_root:
        print(f"Images Root: {images_root}")

    all_ok = True
    for dataset in ["3k", "10k", "13k"]:
        ok, errors = validate_dataset(manifests_dir, dataset, images_root)
        if not ok:
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("ALL DATASET MANIFESTS VALIDATED SUCCESSFULLY. READY FOR PUBLICATION.")
        sys.exit(0)
    else:
        print("VALIDATION FAILED WITH ERRORS.")
        sys.exit(1)


if __name__ == "__main__":
    main()