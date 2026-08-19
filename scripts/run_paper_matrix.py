#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper Matrix Automation Runner: Orchestrates full experimental training matrix.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BACKBONES = {
    "mnv3": "mobilenetv3_large_100.ra_in1k",
    "effb0": "efficientnet_b0.ra_in1k",
    "nfnet": "dm_nfnet_f2.dm_in1k",
    "tfb3": "tf_efficientnet_b3.ns_jft_in1k",
    "convnext": "convnext_tiny.fb_in1k",
    "swint": "swin_tiny_patch4_window7_224.ms_in1k",
}

STEM_MAP = {
    "mnv3": "MNV3",
    "effb0": "EffB0",
    "nfnet": "NFNet",
    "tfb3": "TFB3",
    "convnext": "ConvNext",
    "swint": "SwinT",
}


def main():
    parser = argparse.ArgumentParser("AI-Chemistry Paper Experiment Matrix Runner")
    parser.add_argument("--dataset", type=str, default="13k", choices=["3k", "10k", "13k", "all"])
    parser.add_argument("--backbone", type=str, default="all", choices=list(BACKBONES.keys()) + ["all"])
    parser.add_argument("--calib", type=str, default="all", choices=["none", "greenborder", "all"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--manifests_dir", type=str, default="data/manifests")
    parser.add_argument("--images_root", type=str, default="data")
    args = parser.parse_args()

    datasets = ["3k", "10k", "13k"] if args.dataset == "all" else [args.dataset]
    backbones = list(BACKBONES.keys()) if args.backbone == "all" else [args.backbone]
    calibs = ["none", "greenborder"] if args.calib == "all" else [args.calib]

    total_runs = len(datasets) * len(backbones) * len(calibs)
    print(f"Starting Paper Matrix Execution: {total_runs} Total Runs Planned")

    run_idx = 1
    for ds in datasets:
        for bb in backbones:
            timm_name = BACKBONES[bb]
            stem = STEM_MAP[bb]
            for calib in calibs:
                calib_suffix = "none" if calib == "none" else "green"
                save_ckpt = f"weights/runs_multitask_{ds}/{stem}_seed{args.seed}_l2.0_{calib_suffix}.pt"

                print(f"\n[{run_idx}/{total_runs}] Training {stem} ({timm_name}) on {ds.upper()} ({calib})")
                cmd = [
                    sys.executable,
                    "-m",
                    "ai_chemistry.training.train_classifier",
                    "--dataset",
                    ds,
                    "--timm_name",
                    timm_name,
                    "--manifests_dir",
                    args.manifests_dir,
                    "--images_root",
                    args.images_root,
                    "--calib_mode",
                    calib,
                    "--epochs",
                    str(args.epochs),
                    "--warmup_epochs",
                    str(args.warmup_epochs),
                    "--batch_size",
                    str(args.batch_size),
                    "--seed",
                    str(args.seed),
                    "--loss_weight_cls",
                    "1.0",
                    "--loss_weight_reg",
                    "2.0",
                    "--label_smoothing",
                    "0.05",
                    "--drop",
                    "0.2",
                    "--drop_path",
                    "0.1",
                    "--grad_clip",
                    "1.0",
                    "--patience",
                    "10",
                    "--save_ckpt",
                    save_ckpt,
                    "--device",
                    args.device,
                    "--run_test",
                ]
                subprocess.run(cmd, check=True)
                run_idx += 1


if __name__ == "__main__":
    main()