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

    analyte_true_names: List[str] = []
    analyte_pred_names: List[str] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            images = batch["image"].to(device)
            cls_targets = batch["chemical_idx"].numpy()
            reg_targets = batch["ppm_raw"].numpy()
            paths = batch["path"]

            logits, reg_nh4, reg_no2 = model(images)

            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds_cls = probs.argmax(axis=1)

            reg_nh4_np = reg_nh4.cpu().numpy()
            reg_no2_np = reg_no2.cpu().numpy()

            for i in range(len(cls_targets)):
                c_true = int(cls_targets[i])
                c_pred = int(preds_cls[i])
                p_vec = probs[i].tolist()

                sample_paths.append(str(paths[i]))
                y_cls_true.append(c_true)
                y_cls_pred.append(c_pred)
                y_cls_probs.append(p_vec)

                true_chem = classes[c_true] if c_true < len(classes) else "Unknown"
                pred_chem = classes[c_pred] if c_pred < len(classes) else "Unknown"
                analyte_true_names.append(true_chem)
                analyte_pred_names.append(pred_chem)

                # Route regression head based on PREDICTED analyte
                if pred_chem.upper().startswith("NH4"):
                    reg_out = reg_nh4_np[i]
                else:
                    reg_out = reg_no2_np[i]

                mu_scaled = float(reg_out[0])
                lv_scaled = float(reg_out[1]) if reg_out.shape[0] >= 2 else None

                ppm_pred = inverse_scale_ppm(mu_scaled, ppm_scale, ppm_min, ppm_max)
                ppm_pred = max(0.0, float(ppm_pred))

                y_reg_true_raw.append(float(reg_targets[i]))
                y_reg_pred_raw.append(ppm_pred)
                y_reg_logvar.append(lv_scaled)

    # Compute Classification Metrics
    acc = float(accuracy_score(y_cls_true, y_cls_pred))
    macro_f1 = float(f1_score(y_cls_true, y_cls_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_cls_true, y_cls_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_cls_true, y_cls_pred).tolist()

    # Compute End-to-End Regression Metrics
    mae = float(mean_absolute_error(y_reg_true_raw, y_reg_pred_raw))
    rmse = float(math.sqrt(mean_squared_error(y_reg_true_raw, y_reg_pred_raw)))
    r2 = float(r2_score(y_reg_true_raw, y_reg_pred_raw))
    mape = safe_mape(y_reg_true_raw, y_reg_pred_raw)

    # Per-analyte breakdown
    per_analyte: Dict[str, Dict[str, float]] = {}
    for c_idx, c_name in enumerate(classes):
        mask = [t == c_idx for t in y_cls_true]
        if any(mask):
            y_t_sub = [y_reg_true_raw[j] for j, m in enumerate(mask) if m]
            y_p_sub = [y_reg_pred_raw[j] for j, m in enumerate(mask) if m]
            per_analyte[c_name] = {
                "count": int(sum(mask)),
                "mae": float(mean_absolute_error(y_t_sub, y_p_sub)),
                "rmse": float(math.sqrt(mean_squared_error(y_t_sub, y_p_sub))),
                "r2": float(r2_score(y_t_sub, y_p_sub)) if len(y_t_sub) > 1 else 0.0,
                "mape": safe_mape(y_t_sub, y_p_sub),
            }

    metrics: Dict[str, Any] = {
        "classification": {
            "accuracy": acc,
            "f1_macro": macro_f1,
            "f1_weighted": weighted_f1,
            "confusion_matrix": cm,
        },
        "regression": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
            "per_analyte": per_analyte,
        },
    }

    df_preds = pd.DataFrame(
        {
            "path": sample_paths,
            "chemical_true": analyte_true_names,
            "chemical_pred": analyte_pred_names,
            "ppm_true": y_reg_true_raw,
            "ppm_pred": y_reg_pred_raw,
            "logvar_scaled": y_reg_logvar,
            "cls_probs": y_cls_probs,
        }
    )

    return metrics, df_preds


def main():
    parser = argparse.ArgumentParser("AI-Chemistry Publication Test Evaluator")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--meta_path", type=str, default=None, help="Optional path to .meta.json sidecar.")
    parser.add_argument("--dataset", type=str, default="10k", choices=["3k", "10k", "13k"])
    parser.add_argument("--manifests_dir", type=str, default="data/manifests")
    parser.add_argument("--manifest_csv", type=str, default=None, help="Optional path to specific CSV manifest.")
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
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    if not isinstance(state, dict):
        raise RuntimeError(f"Invalid state dict in checkpoint: {ckpt_path}")

    state = strip_state_dict_prefix(state)

    meta_path = Path(args.meta_path) if args.meta_path else ckpt_path.with_suffix(".meta.json")
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            m = json.load(f)
            meta = build_meta_from_ckpt({**ckpt, **m}, ckpt_path=ckpt_path)
    else:
        meta = build_meta_from_ckpt(ckpt, ckpt_path=ckpt_path)

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
    if args.manifest_csv:
        manifest_p = Path(args.manifest_csv).resolve()
        if not manifest_p.is_file():
            raise FileNotFoundError(f"Custom manifest file not found: {manifest_p}")
        target_df = prepare_split_frame(pd.read_csv(manifest_p))
        images_root = Path(args.images_root).resolve()
        dataset_name = f"custom_{manifest_p.stem}"
        split_name = "custom"
    else:
        images_root, train_df, val_df, test_df = load_publication_splits(
            args.manifests_dir, args.dataset, args.images_root
        )
        splits = {"train": train_df, "val": val_df, "test": test_df}
        target_df = splits[args.split]
        dataset_name = args.dataset
        split_name = args.split

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
    print(f"EVALUATION REPORT: {ckpt_path.stem} on {dataset_name.upper()} [{split_name.upper()}]")
    print("=" * 60)
    cls_res = metrics["classification"]
    reg_res = metrics["regression"]
    print(f"Classification Accuracy : {cls_res['accuracy']*100:.2f}%")
    print(f"Classification Macro-F1 : {cls_res['f1_macro']:.4f}")
    print(f"Regression MAE (ppm)    : {reg_res['mae']:.4f}")
    print(f"Regression RMSE (ppm)   : {reg_res['rmse']:.4f}")
    print(f"Regression R2 Score     : {reg_res['r2']:.4f}")
    print(f"Regression MAPE (%)     : {reg_res['mape']:.2f}%")
    print("-" * 60)

    # Save Receipts
    receipt = {
        "checkpoint": str(ckpt_path.name),
        "dataset": dataset_name,
        "split": split_name,
        "calib_train": train_calib,
        "calib_eval": eval_calib,
        "metrics": metrics,
    }

    if args.output_json:
        out_j = Path(args.output_json).resolve()
        out_j.parent.mkdir(parents=True, exist_ok=True)
        with open(out_j, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
        print(f"Saved JSON Receipt: {out_j}")

    if args.predictions_csv:
        out_c = Path(args.predictions_csv).resolve()
        out_c.parent.mkdir(parents=True, exist_ok=True)
        df_preds.to_csv(out_c, index=False)
        print(f"Saved Predictions CSV: {out_c}")


if __name__ == "__main__":
    main()