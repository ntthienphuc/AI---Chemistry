#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Source-Resolved 13K Robustness (C3 & C4 components).

Partitions the frozen 13K test split into:
  - C3: 3K-origin field component (samples originating from the 3K natural water domain)
  - C4: 10K-origin laboratory component (samples originating from the 10K laboratory matrix domain)

Evaluates 13K-trained models on both subsets to assess domain robustness.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

BACKBONES = ["mnv3", "effb0", "nfnet", "tfb3", "convnext", "swint"]
STEM_MAP = {
    "mnv3": "MNV3",
    "effb0": "EffB0",
    "nfnet": "NFNet",
    "tfb3": "TFB3",
    "convnext": "ConvNext",
    "swint": "SwinT",
}


def build_source_resolved_splits(manifests_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_13k_p = manifests_dir / "13k" / "test.csv"
    if not test_13k_p.is_file():
        raise FileNotFoundError(f"13K test manifest not found: {test_13k_p}")

    df_13k_test = pd.read_csv(test_13k_p)

    # Gather all 3k and 10k paths
    paths_3k = set()
    for s in ["train", "val", "test"]:
        p = manifests_dir / "3k" / f"{s}.csv"
        if p.is_file():
            paths_3k.update(pd.read_csv(p)["path"].astype(str).str.lower().tolist())

    paths_10k = set()
    for s in ["train", "val", "test"]:
        p = manifests_dir / "10k" / f"{s}.csv"
        if p.is_file():
            paths_10k.update(pd.read_csv(p)["path"].astype(str).str.lower().tolist())

    df_13k_test["_norm_path"] = df_13k_test["path"].astype(str).str.lower()

    # C3: 3K-origin subset
    c3_df = df_13k_test[df_13k_test["_norm_path"].isin(paths_3k)].copy().drop(columns=["_norm_path"])
    # C4: 10K-origin subset
    c4_df = df_13k_test[df_13k_test["_norm_path"].isin(paths_10k)].copy().drop(columns=["_norm_path"])

    return c3_df, c4_df


def main():
    parser = argparse.ArgumentParser("Source-Resolved 13K Robustness Evaluator")
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--manifests_dir", type=str, default="data/manifests")
    parser.add_argument("--images_root", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/source_resolved_13k")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    manifests_root = Path(args.manifests_dir).resolve()
    weights_root = Path(args.weights_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    c3_df, c4_df = build_source_resolved_splits(manifests_root)
    print(f"13K Test Split Partitioned: C3 (3K-origin) = {len(c3_df)} samples, C4 (10K-origin) = {len(c4_df)} samples")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        c3_manifest = tmp_p / "13k_test_c3_field.csv"
        c4_manifest = tmp_p / "13k_test_c4_lab.csv"
        c3_df.to_csv(c3_manifest, index=False)
        c4_df.to_csv(c4_manifest, index=False)

        components = [
            ("C3_field_origin", c3_manifest),
            ("C4_lab_origin", c4_manifest),
        ]

        ds_weights = weights_root / "runs_multitask_13k"

        for comp_name, comp_manifest in components:
            print(f"\n==================================================")
            print(f"Evaluating 13K Models on {comp_name}")
            print(f"==================================================")

            for bb in BACKBONES:
                stem = STEM_MAP[bb]
                for calib in ["none", "green"]:
                    ckpt_name = f"{stem}_seed0_l2.0_{calib}.pt"
                    ckpt_path = ds_weights / ckpt_name
                    if not ckpt_path.is_file():
                        continue

                    out_json = out_root / f"{bb}_13k_{calib}_{comp_name}.json"
                    out_csv = out_root / f"{bb}_13k_{calib}_{comp_name}_preds.csv"
                    calib_arg = "none" if calib == "none" else "greenborder"

                    # Run test_classifier with custom manifest
                    cmd = [
                        sys.executable,
                        "-m",
                        "ai_chemistry.training.test_classifier",
                        "--ckpt_path",
                        str(ckpt_path),
                        "--dataset",
                        "13k",
                        "--manifests_dir",
                        str(manifests_root),
                        "--images_root",
                        args.images_root,
                        "--split",
                        "test",
                        "--calib",
                        calib_arg,
                        "--device",
                        args.device,
                        "--output_json",
                        str(out_json),
                        "--predictions_csv",
                        str(out_csv),
                    ]
                    print(f"Running: {stem} on {comp_name} ({calib_arg})")
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()