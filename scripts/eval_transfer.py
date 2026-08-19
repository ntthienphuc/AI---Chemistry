#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Cross-Dataset Domain Generalization (3K <-> 10K Transfer).
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
    parser = argparse.ArgumentParser("Cross-Dataset Domain Transfer Evaluation")
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--manifests_dir", type=str, default="data/manifests")
    parser.add_argument("--images_root", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/transfer")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    weights_root = Path(args.weights_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    transfer_tasks = [
        ("3k", "10k"),  # Train 3K -> Test 10K
        ("10k", "3k"),  # Train 10K -> Test 3K
    ]

    for src_ds, tgt_ds in transfer_tasks:
        print(f"\n--- Cross-Dataset Transfer: Train {src_ds.upper()} -> Test {tgt_ds.upper()} ---")
        ds_weights = weights_root / f"runs_multitask_{src_ds}"

        for bb in BACKBONES:
            stem = STEM_MAP[bb]
            for calib in ["none", "green"]:
                ckpt_name = f"{stem}_seed0_l2.0_{calib}.pt"
                ckpt_path = ds_weights / ckpt_name
                if not ckpt_path.is_file():
                    continue

                out_json = out_root / f"transfer_train_{src_ds}_test_{tgt_ds}_{bb}_{calib}.json"
                out_csv = out_root / f"transfer_train_{src_ds}_test_{tgt_ds}_{bb}_{calib}_preds.csv"
                calib_arg = "none" if calib == "none" else "greenborder"

                cmd = [
                    sys.executable,
                    "-m",
                    "ai_chemistry.training.test_classifier",
                    "--ckpt_path",
                    str(ckpt_path),
                    "--dataset",
                    tgt_ds,
                    "--manifests_dir",
                    args.manifests_dir,
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
                print(f"Running: {stem} (trained on {src_ds}, tested on {tgt_ds})")
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()