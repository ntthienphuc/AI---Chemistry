# -*- coding: utf-8 -*-
"""
Multi-Task Heteroscedastic Neural Network Training Protocol for Publication Reproduction.

Key Highlights:
- Architecture: Canonical MultiTaskHetero with MLP2 task heads (512 hidden, ReLU, Dropout 0.3).
- Training Schedule: 60 total epochs (epochs 1-5 warmup with frozen backbone).
- Loss Function: Heteroscedastic Gaussian Negative Log-Likelihood + Label-smoothed Cross-Entropy.
- Loss Balancing: lambda_cls = 1.0, lambda_reg = 2.0.
- Selection Metric: Score = (1.0 - Acc) + 2.0 * MAE (unsmoothed validation).
- Seed: 0 deterministic baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch.utils.data import DataLoader
from tqdm import tqdm

from ai_chemistry.data.loaders import (
    ChemistryDataset,
    inverse_scale_ppm,
    load_publication_splits,
    prepare_split_frame,
    scale_ppm,
)
from ai_chemistry.modeling import MultiTaskHetero, PAPER_BACKBONES
from ai_chemistry.preprocessing import (
    get_normalizer,
    make_eval_transform,
    make_train_transform,
)

torch.backends.cudnn.benchmark = True
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger("ai_chemistry.train")


def set_seed(seed: int = 0) -> None:
    """Ensure fully deterministic seed initialization across all libraries."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_mape(y_true, y_pred, eps: float = 1e-6) -> float:
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_t), eps)
    return float(np.mean(np.abs((y_t - y_p) / denom))) * 100.0


class GaussianNLLLossPerSample(nn.Module):
    """Heteroscedastic Gaussian Negative Log-Likelihood Loss."""

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        inv_var = torch.exp(-log_var).clamp(max=1e6)
        return 0.5 * (inv_var * (target - mu) ** 2 + log_var)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ppm_scale: str,
    ppm_min: Optional[float],
    ppm_max: Optional[float],
    prefix: str = "Eval",
) -> Dict[str, float]:
    model.eval()
    y_true_cls, y_pred_cls = [], []
    y_true_reg_s, y_pred_reg_s = [], []

    per_class_true = {0: [], 1: []}
    per_class_pred = {0: [], 1: []}

    for x, y_cls, y_reg, _ in tqdm(loader, desc=prefix, leave=False):
        x = x.to(device, non_blocking=True)
        y_cls = y_cls.to(device, non_blocking=True)
        y_reg = y_reg.to(device, non_blocking=True).float()

        logits, rNH4, rNO2 = model(x)
        pred_cls = logits.argmax(dim=1)

        mu_NH4 = rNH4[:, 0]
        mu_NO2 = rNO2[:, 0]
        mu_heads = torch.stack([mu_NH4, mu_NO2], dim=1)
        pred_reg_s = mu_heads.gather(1, pred_cls.view(-1, 1)).squeeze(1)

        y_true_cls.extend(y_cls.detach().cpu().numpy().tolist())
        y_pred_cls.extend(pred_cls.detach().cpu().numpy().tolist())
        y_true_reg_s.extend(y_reg.detach().cpu().numpy().tolist())
        y_pred_reg_s.extend(pred_reg_s.detach().cpu().numpy().tolist())

        y_cls_np = y_cls.detach().cpu().numpy()
        y_reg_np = y_reg.detach().cpu().numpy()
        pred_reg_np = pred_reg_s.detach().cpu().numpy()
        for i, gt in enumerate(y_cls_np):
            per_class_true[int(gt)].append(float(y_reg_np[i]))
            per_class_pred[int(gt)].append(float(pred_reg_np[i]))

    acc = float(accuracy_score(y_true_cls, y_pred_cls))
    f1_macro = float(f1_score(y_true_cls, y_pred_cls, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true_cls, y_pred_cls, average="weighted", zero_division=0))

    y_true_ppm = [inverse_scale_ppm(v, ppm_scale, ppm_min, ppm_max) for v in y_true_reg_s]
    y_pred_ppm = [inverse_scale_ppm(v, ppm_scale, ppm_min, ppm_max) for v in y_pred_reg_s]
    mae = float(mean_absolute_error(y_true_ppm, y_pred_ppm))
    mape = float(safe_mape(y_true_ppm, y_pred_ppm))

    logs = {
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "mae": mae,
        "mape": mape,
    }

    logger.info(
        f"{prefix} | Acc: {acc:.4f} | Macro-F1: {f1_macro:.4f} | MAE: {mae:.4f} ppm | MAPE: {mape:.2f}%"
    )
    return logs


def resolve_data_splits(args) -> Tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resolve data manifests from either the publication manifest directory or legacy CSV."""
    if args.dataset:
        manifests_dir = args.manifests_dir or "data/manifests"
        images_root = args.images_root or args.data_root or args.root_dir or "data"
        return load_publication_splits(manifests_dir, args.dataset, images_root)

    if args.labels_csv:
        csv_path = Path(args.labels_csv).resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f"Labels CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        if "split" not in df.columns:
            raise ValueError(f"Legacy CSV must contain a 'split' column: {csv_path}")

        images_root = Path(args.images_root or args.data_root or args.root_dir or ".").resolve()
        train_df = prepare_split_frame(df[df["split"].str.lower() == "train"], "train", csv_path)
        val_df = prepare_split_frame(df[df["split"].str.lower() == "val"], "val", csv_path)
        test_df = prepare_split_frame(df[df["split"].str.lower() == "test"], "test", csv_path)
        return images_root, train_df, val_df, test_df

    # Default fallback: 13k
    manifests_dir = args.manifests_dir or "data/manifests"
    images_root = args.images_root or args.data_root or args.root_dir or "data"
    return load_publication_splits(manifests_dir, "13k", images_root)


def train(args) -> None:
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info(f"Using device: {device}")

    images_root, train_df, val_df, test_df = resolve_data_splits(args)
    logger.info(f"Loaded dataset splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    classes = ("NH4", "NO2")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    ppm_min = ppm_max = None
    if args.ppm_scale == "minmax":
        ppm_min = float(train_df["ppm"].min())
        ppm_max = float(train_df["ppm"].max())

    normalizer = get_normalizer(
        args.calib_mode,
        ring_frac=args.ring_frac,
        inner_margin=args.inner_margin,
        min_green_pixels=args.min_green_pixels,
    )

    tf_train = make_train_transform(args.image_size)
    tf_eval = make_eval_transform(args.image_size)

    train_ds = ChemistryDataset(
        train_df, images_root, class_to_idx, args.ppm_scale, tf_train, normalizer, ppm_min, ppm_max
    )
    val_ds = ChemistryDataset(
        val_df, images_root, class_to_idx, args.ppm_scale, tf_eval, normalizer, ppm_min, ppm_max
    )
    test_ds = ChemistryDataset(
        test_df, images_root, class_to_idx, args.ppm_scale, tf_eval, normalizer, ppm_min, ppm_max
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # Initialize Canonical MultiTaskHetero
    timm_name = PAPER_BACKBONES.get(args.timm_name.lower(), args.timm_name)
    logger.info(f"Initializing MultiTaskHetero with backbone: {timm_name}")

    model = MultiTaskHetero(
        timm_name=timm_name,
        num_classes=len(classes),
        pretrained=not args.scratch,
        drop=args.drop,
        drop_path=args.drop_path,
        image_size=args.image_size,
    ).to(device)

    # Backbone warmup: freeze backbone during warmup epochs
    if args.warmup_epochs > 0:
        logger.info(f"Freezing backbone for initial {args.warmup_epochs} warmup epochs.")
        for p in model.backbone.parameters():
            p.requires_grad = False

    def make_optimizer(train_all: bool) -> optim.Optimizer:
        params = model.parameters() if train_all else filter(lambda p: p.requires_grad, model.parameters())
        return optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    optimizer = make_optimizer(train_all=False if args.warmup_epochs > 0 else True)

    total_epochs = args.epochs  # 60 total epochs (warmup included)
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def lr_lambda(epoch: int) -> float:
        t = epoch / max(1, total_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    gauss_loss = GaussianNLLLossPerSample()
    cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    save_ckpt = Path(args.save_ckpt or f"weights/runs_multitask_{args.dataset or '13k'}/{timm_name.split('.')[0]}_seed{args.seed}_l{args.loss_weight_reg}_{args.calib_mode}.pt")
    save_meta = Path(args.save_meta or save_ckpt.with_suffix(".meta.json"))
    save_ckpt.parent.mkdir(parents=True, exist_ok=True)
    save_meta.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = 0
    patience = 0

    logger.info(f"Starting training run: Total Epochs={total_epochs}, Warmup={args.warmup_epochs}, Seed={args.seed}")
    for epoch in range(total_epochs):
        model.train()

        # Unfreeze backbone after warmup
        if args.warmup_epochs > 0 and epoch == args.warmup_epochs:
            logger.info("Unfreezing backbone. Rebuilding optimizer for full network training.")
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = make_optimizer(train_all=True)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        running_cls = 0.0
        running_reg = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}", leave=False)
        for x, y_cls, y_reg, _ in pbar:
            x = x.to(device, non_blocking=True)
            y_cls = y_cls.to(device, non_blocking=True)
            y_reg = y_reg.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, rNH4, rNO2 = model(x)
                l_cls = cls_loss_fn(logits, y_cls)

                mu_NH4, lv_NH4 = rNH4[:, 0], rNH4[:, 1]
                mu_NO2, lv_NO2 = rNO2[:, 0], rNO2[:, 1]
                mu_heads = torch.stack([mu_NH4, mu_NO2], dim=1)
                lv_heads = torch.stack([lv_NH4, lv_NO2], dim=1)
                mu_true = mu_heads.gather(1, y_cls.view(-1, 1)).squeeze(1)
                lv_true = lv_heads.gather(1, y_cls.view(-1, 1)).squeeze(1)

                l_reg = gauss_loss(mu_true, lv_true, y_reg).mean()
                loss = args.loss_weight_cls * l_cls + args.loss_weight_reg * l_reg

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_cls += float(l_cls.item())
            running_reg += float(l_reg.item())
            pbar.set_postfix(
                cls=f"{running_cls / max(1, pbar.n):.4f}",
                reg=f"{running_reg / max(1, pbar.n):.4f}",
            )

        scheduler.step()

        # Validation selection
        val_logs = evaluate(model, val_loader, device, args.ppm_scale, ppm_min, ppm_max, prefix="Val")
        val_score = args.loss_weight_cls * (1.0 - val_logs["acc"]) + args.loss_weight_reg * val_logs["mae"]

        if val_score < best_val - 1e-6:
            best_val = val_score
            best_epoch = epoch + 1
            patience = 0

            # Save state dict and self-describing metadata
            meta = {
                "protocol_version": "paper_v1",
                "architecture": "multitask_heteroscedastic_mlp2",
                "head_variant": "mlp2",
                "head_hidden_dim": 512,
                "head_dropout": 0.3,
                "timm_name": timm_name,
                "image_size": args.image_size,
                "classes": list(classes),
                "ppm_scale": args.ppm_scale,
                "ppm_min": ppm_min,
                "ppm_max": ppm_max,
                "calib_mode_train": args.calib_mode,
                "drop": args.drop,
                "drop_path": args.drop_path,
                "loss_weight_cls": args.loss_weight_cls,
                "loss_weight_reg": args.loss_weight_reg,
                "label_smoothing": args.label_smoothing,
                "optimizer": "AdamW",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "epochs_total_max": total_epochs,
                "warmup_epochs": args.warmup_epochs,
                "grad_clip": args.grad_clip,
                "ema": False,
                "seed": args.seed,
                "selection_score_formula": "(1-accuracy) + 2.0*MAE",
                "best_epoch": best_epoch,
                "best_val_score": best_val,
                "dataset": args.dataset or "custom",
            }

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    **meta,
                },
                save_ckpt,
            )

            with open(save_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            logger.info(f"★ Saved Best Checkpoint (Epoch {best_epoch}, Score {best_val:.4f}): {save_ckpt}")
        else:
            patience += 1
            if patience >= args.patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1} (patience={args.patience}).")
                break

    logger.info(f"Training completed. Best model from epoch {best_epoch} with val score {best_val:.4f}.")

    if args.run_test:
        logger.info("Executing final evaluation on test split using best checkpoint...")
        ckpt = torch.load(save_ckpt, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        _ = evaluate(model, test_loader, device, args.ppm_scale, ppm_min, ppm_max, prefix="Test")


def parse_args():
    parser = argparse.ArgumentParser("AI-Chemistry Publication Model Trainer")

    # Data inputs
    parser.add_argument("--dataset", type=str, default="13k", choices=["3k", "10k", "13k"])
    parser.add_argument("--manifests_dir", type=str, default="data/manifests")
    parser.add_argument("--images_root", type=str, default="data")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--labels_csv", type=str, default=None)

    # Architecture & Backbone
    parser.add_argument(
        "--timm_name",
        type=str,
        default="mobilenetv3_large_100.ra_in1k",
        help="timm model identifier or alias (mnv3, effb0, nfnet, tfb3, convnext, swint).",
    )
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--drop", type=float, default=0.2)
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument("--scratch", action="store_true", help="Train from scratch without ImageNet weights.")

    # Preprocessing
    parser.add_argument("--calib_mode", type=str, default="none", choices=["none", "greenborder"])
    parser.add_argument("--ring_frac", type=float, default=0.08)
    parser.add_argument("--inner_margin", type=int, default=2)
    parser.add_argument("--min_green_pixels", type=int, default=300)

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=60, help="Maximum TOTAL training epochs (warmup included).")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Backbone frozen warmup epochs (included in total).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss_weight_cls", type=float, default=1.0)
    parser.add_argument("--loss_weight_reg", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--ppm_scale", type=str, default="log1p", choices=["log1p", "minmax", "none"])

    # Output & Runtime
    parser.add_argument("--save_ckpt", type=str, default=None)
    parser.add_argument("--save_meta", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--run_test", action="store_true", help="Run evaluation on test split after training.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)