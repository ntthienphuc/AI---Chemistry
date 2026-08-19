# -*- coding: utf-8 -*-
"""
Multi-Task Heteroscedastic Neural Network Evaluator for Publication Benchmarking.

Computes:
- Classification: Accuracy, Macro-F1, Weighted-F1, Per-class Precision/Recall.
- End-to-end Regression (Predicted Class Routed): MAE, RMSE, R2, MAPE, per-analyte metrics.
- Exports structured JSON receipts and per-sample predictions CSV.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ai_chemistry.data.loaders import (
    ChemistryDataset,
    inverse_scale_ppm,
    load_publication_splits,
    prepare_split_frame,
    scale_ppm,
)
from ai_chemistry.modeling import (
    MultiTaskHeteroFlexible,
    build_meta_from_ckpt,
    infer_head_in_features,
    infer_head_variant,
    infer_reg_out_dim,
    strip_state_dict_prefix,
)
from ai_chemistry.preprocessing import get_normalizer, make_eval_transform

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger("ai_chemistry.test")


def safe_mape(y_true, y_pred, eps: float = 1e-6) -> float:
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_t), eps)
    return float(np.mean(np.abs((y_t - y_p) / denom))) * 100.0


def evaluate_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    ppm_scale: str,
    ppm_min: Optional[float] = None,
    ppm_max: Optional[float] = None,
    classes: Tuple[str, ...] = ("NH4", "NO2"),
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    model.eval()

    sample_paths: List[str] = []
    y_cls_true: List[int] = []
    y_cls_pred: List[int] = []
    y_cls_probs: List[List[float]] = []

    y_reg_true_raw: List[float] = []
    y_reg_pred_raw: List[float] = []
    y_reg_logvar: List[Optional[float]] = []

    with torch.no_grad():
        for images, chem_indices, ppm_scaled, paths in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            chem_indices = chem_indices.to(device, non_blocking=True)
            ppm_scaled = ppm_scaled.to(device, non_blocking=True).float()

            logits, reg_NH4, reg_NO2, _ = model(images)

            probs = F.softmax(logits, dim=1)
            pred_cls = logits.argmax(dim=1)

            mu_NH4 = reg_NH4[:, 0]
            mu_NO2 = reg_NO2[:, 0]
            mu_heads = torch.stack([mu_NH4, mu_NO2], dim=1)
            pred_reg_scaled = mu_heads.gather(1, pred_cls.view(-1, 1)).squeeze(1)

            lv_heads = None
            if reg_NH4.shape[1] >= 2 and reg_NO2.shape[1] >= 2:
                lv_heads = torch.stack([reg_NH4[:, 1], reg_NO2[:, 1]], dim=1)
                pred_lv = lv_heads.gather(1, pred_cls.view(-1, 1)).squeeze(1)
                y_reg_logvar.extend(pred_lv.cpu().numpy().tolist())
            else:
                y_reg_logvar.extend([None] * len(images))

            sample_paths.extend(paths)
            y_cls_true.extend(chem_indices.cpu().numpy().tolist())
            y_cls_pred.extend(pred_cls.cpu().numpy().tolist())
            y_cls_probs.extend(probs.cpu().numpy().tolist())

            y_reg_true_raw.extend(ppm_scaled.cpu().numpy().tolist())
            y_reg_pred_raw.extend(pred_reg_scaled.cpu().numpy().tolist())

    y_t_cls = np.array(y_cls_true)
    y_p_cls = np.array(y_cls_pred)

    y_t_ppm = np.array([inverse_scale_ppm(v, ppm_scale, ppm_min, ppm_max) for v in y_reg_true_raw])
    y_p_ppm = np.clip(
        np.array([inverse_scale_ppm(v, ppm_scale, ppm_min, ppm_max) for v in y_reg_pred_raw]), 0.0, np.inf
    )

    # Classification Metrics
    acc = float(accuracy_score(y_t_cls, y_p_cls))
    f1_macro = float(f1_score(y_t_cls, y_p_cls, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_t_cls, y_p_cls, average="weighted", zero_division=0))
    cm = confusion_matrix(y_t_cls, y_p_cls).tolist()

    # Regression Metrics (Overall)
    mae = float(mean_absolute_error(y_t_ppm, y_p_ppm))
    rmse = float(np.sqrt(mean_squared_error(y_t_ppm, y_p_ppm)))
    r2 = float(r2_score(y_t_ppm, y_p_ppm)) if len(y_t_ppm) > 1 else 1.0
    mape = float(safe_mape(y_t_ppm, y_p_ppm))

    # Per-analyte breakdown
    per_analyte: Dict[str, Any] = {}
    for idx, cname in enumerate(classes):
        mask = y_t_cls == idx
        if np.any(mask):
            sub_t = y_t_ppm[mask]
            sub_p = y_p_ppm[mask]
            per_analyte[cname] = {
                "count": int(np.sum(mask)),
                "mae": float(mean_absolute_error(sub_t, sub_p)),
                "rmse": float(np.sqrt(mean_squared_error(sub_t, sub_p))),
                "r2": float(r2_score(sub_t, sub_p)) if len(sub_t) > 1 else 1.0,
                "mape": float(safe_mape(sub_t, sub_p)),
            }
        else:
            per_analyte[cname] = None

    metrics = {
        "classification": {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "confusion_matrix": cm,
        },
        "regression": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
            "per_analyte": per_analyte,
        },
        "sample_count": len(sample_paths),
    }

    # Build predictions dataframe
    records = []
    for i in range(len(sample_paths)):
        records.append(
            {
                "path": sample_paths[i],
                "true_chemical": classes[y_t_cls[i]] if y_t_cls[i] < len(classes) else str(y_t_cls[i]),
                "pred_chemical": classes[y_p_cls[i]] if y_p_cls[i] < len(classes) else str(y_p_cls[i]),
                "prob_NH4": float(y_cls_probs[i][0]),
                "prob_NO2": float(y_cls_probs[i][1]) if len(y_cls_probs[i]) > 1 else 0.0,
                "true_ppm": float(y_t_ppm[i]),
                "pred_ppm": float(y_p_ppm[i]),
                "logvar": float(y_reg_logvar[i]) if y_reg_logvar[i] is not None else None,
            }
        )
    df_preds = pd.DataFrame(records)
    return metrics, df_preds


def main():
    parser = argparse.ArgumentParser("AI-Chemistry Publication Test Evaluator")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--meta_path", type=str, default=None, help="Optional path to .meta.json sidecar.")
    parser.add_argument("--dataset", type=str, default="10k", choices=["3k", "10k", "13k"])
    parser.add_argument("--manifests_dir", type=str, default="data/manifests")
    parser.add_argument("--images_root", type=str, default="data")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"])
    parser.add_argument("--calib", type=str, default="auto", help="auto | none | greenborder")
    parser.add_argument("--allow_preprocessing_mismatch", action="store_true", help="Allow intentional mismatch for ablation experiments.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save evaluation summary JSON.")
    parser.add_argument("--predictions_csv", type=str, default=None, help="Path to save sample-level predictions CSV.")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt_path = Path(args.ckpt_path).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    # Load checkpoint & metadata
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    if not isinstance(state, dict):
        raise RuntimeError(f"Invalid state dict in checkpoint: {ckpt_path}")

    state = strip_state_dict_prefix(state)

    meta_path = Path(args.meta_path) if args.meta_path else ckpt_path.with_suffix(".meta.json")
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            m = json.load(f)
            meta = build_meta_from_ckpt({**ckpt, **m})
    else:
        meta = build_meta_from_ckpt(ckpt)

    # Resolve Calibration Mode
    train_calib = meta.calib_mode_train
    if args.calib == "auto":
        eval_calib = train_calib
    else:
        eval_calib = args.calib

    if eval_calib != train_calib and not args.allow_preprocessing_mismatch:
        raise ValueError(
            f"Preprocessing mismatch detected! Checkpoint trained with '{train_calib}', but evaluation requested '{eval_calib}'. "
            f"Pass --allow_preprocessing_mismatch only for intentional mismatch ablation experiments."
        )

    logger.info(f"Loaded Checkpoint: {ckpt_path.name}")
    logger.info(f"Architecture: {meta.timm_name} | Trained Calib: {train_calib} | Eval Calib: {eval_calib}")

    # Build Model
    head_variant = infer_head_variant(state)
    reg_out_dim = infer_reg_out_dim(state)
    expected_feat_dim = infer_head_in_features(state)

    model = MultiTaskHeteroFlexible(
        timm_name=meta.timm_name,
        num_classes=meta.num_classes,
        pretrained=False,
        drop=meta.drop,
        drop_path=meta.drop_path,
        image_size=meta.image_size,
        head_variant=head_variant,
        reg_out_dim=reg_out_dim,
        expected_feat_dim=expected_feat_dim,
    )

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        alt_variant = "linear" if head_variant == "mlp2" else "mlp2"
        model = MultiTaskHeteroFlexible(
            timm_name=meta.timm_name,
            num_classes=meta.num_classes,
            pretrained=False,
            drop=meta.drop,
            drop_path=meta.drop_path,
            image_size=meta.image_size,
            head_variant=alt_variant,
            reg_out_dim=reg_out_dim,
            expected_feat_dim=expected_feat_dim,
        )
        model.load_state_dict(state, strict=True)

    model.to(device).eval()

    # Load dataset
    images_root, train_df, val_df, test_df = load_publication_splits(
        args.manifests_dir, args.dataset, args.images_root
    )
    splits = {"train": train_df, "val": val_df, "test": test_df}
    target_df = splits[args.split]

    class_to_idx = {c: i for i, c in enumerate(meta.classes)}
    normalizer = get_normalizer(eval_calib)
    tf_eval = make_eval_transform(meta.image_size)

    ds = ChemistryDataset(
        target_df, images_root, class_to_idx, meta.ppm_scale, tf_eval, normalizer, meta.ppm_min, meta.ppm_max
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Evaluate
    metrics, df_preds = evaluate_dataset(
        model, loader, device, meta.ppm_scale, meta.ppm_min, meta.ppm_max, meta.classes
    )

    print("\n" + "=" * 60)
    print(f"EVALUATION REPORT: {ckpt_path.stem} on {args.dataset.upper()} [{args.split.upper()}]")
    print("=" * 60)
    cls_res = metrics["classification"]
    reg_res = metrics["regression"]
    print(f"Classification Accuracy : {cls_res['accuracy']*100:.2f}%")
    print(f"Classification Macro-F1 : {cls_res['f1_macro']:.4f}")
    print(f"Regression Overall MAE  : {reg_res['mae']:.4f} ppm (mg/L)")
    print(f"Regression Overall RMSE : {reg_res['rmse']:.4f} ppm (mg/L)")
    print(f"Regression Overall R^2  : {reg_res['r2']:.4f}")
    print(f"Regression Overall MAPE : {reg_res['mape']:.2f}%")
    print("-" * 60)
    for cname, pmetrics in reg_res["per_analyte"].items():
        if pmetrics:
            print(f"  [{cname:<3}] N={pmetrics['count']:<4} | MAE={pmetrics['mae']:.4f} | RMSE={pmetrics['rmse']:.4f} | R^2={pmetrics['r2']:.4f}")
    print("=" * 60 + "\n")

    # Export outputs
    if args.output_json:
        out_j = Path(args.output_json).resolve()
        out_j.parent.mkdir(parents=True, exist_ok=True)
        with open(out_j, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": ckpt_path.name,
                    "dataset": args.dataset,
                    "split": args.split,
                    "calib_train": train_calib,
                    "calib_eval": eval_calib,
                    "metrics": metrics,
                },
                f,
                indent=2,
            )
        logger.info(f"Saved evaluation JSON receipt: {out_j}")

    if args.predictions_csv:
        out_c = Path(args.predictions_csv).resolve()
        out_c.parent.mkdir(parents=True, exist_ok=True)
        df_preds.to_csv(out_c, index=False)
        logger.info(f"Saved predictions CSV: {out_c}")


if __name__ == "__main__":
    main()