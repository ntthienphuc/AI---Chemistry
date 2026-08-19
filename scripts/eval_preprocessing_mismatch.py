#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Preprocessing Mismatch Ablation (train none -> test greenborder).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BACKBONES = ["mnv3", "effb0", "nfnet", "tfb3", "convnext", "swint"]
STEM_MAP = {
    "mnv3": "MNV3",
    "effb0": "EffB0",
    "nfnet": "NFNet",
    "tfb3": "TFB3",
    "convnext": "ConvNext",
    "swint": "SwinT",
}


def main():
    parser = argparse.ArgumentParser("Preprocessing Mismatch Ablation Evaluation")
    parser.add_argument("--dataset", type=str, default="10k", choices=["3k", "10k", "13k", "all"])
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--manifests_dir", type=str, default="data/manifests")
    parser.add_argument("--images_root", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/mismatch")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    datasets = ["3k", "10k", "13k"] if args.dataset == "all" else [args.dataset]
    weights_root = Path(args.weights_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Executing Preprocessing Mismatch Ablation (Train None -> Test Green)")

    for ds in datasets:
        ds_weights = weights_root / f"runs_multitask_{ds}"
        for bb in BACKBONES:
            stem = STEM_MAP[bb]
            ckpt_name = f"{stem}_seed0_l2.0_none.pt"
            ckpt_path = ds_weights / ckpt_name
            if not ckpt_path.is_file():
                print(f"Skipping (Checkpoint not found): {ckpt_path}")
                continue

            out_json = out_root / f"{bb}_{ds}_trainNone_testGreen_mismatch.json"
            out_csv = out_root / f"{bb}_{ds}_trainNone_testGreen_mismatch_preds.csv"

            cmd = [
                sys.executable,
                "-m",
                "ai_chemistry.training.test_classifier",
                "--ckpt_path",
                str(ckpt_path),
                "--dataset",
                ds,
                "--manifests_dir",
                args.manifests_dir,
                "--images_root",
                args.images_root,
                "--split",
                "test",
                "--calib",
                "greenborder",
                "--allow_preprocessing_mismatch",
                "--device",
                args.device,
                "--output_json",
                str(out_json),
                "--predictions_csv",
                str(out_csv),
            ]
            print(f"Running Mismatch Ablation: {stem} on {ds} (none -> greenborder)")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()