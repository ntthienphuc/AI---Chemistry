# ai_chemistry/training/test_classifier.py

import argparse
import json
import math
import os
from pathlib import Path
import csv

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# import lại từ train_classifier.py
from train_classifier import (
    ChemistryDataset,
    MultiTaskHetero,
    GreenBorderNormalizer,
    make_transforms,
    set_seed,
)


def inverse_scale_ppm(y_scaled, ppm_scale, ppm_min, ppm_max):
    """Đưa ppm về đơn vị gốc (mg/L)."""
    y_scaled = np.asarray(y_scaled, dtype=float)

    if ppm_scale == "log1p":
        return np.expm1(y_scaled)
    elif ppm_scale == "minmax":
        return y_scaled * (ppm_max - ppm_min) + ppm_min
    else:
        # "none"
        return y_scaled


def evaluate_model(model, loader, device, ppm_scale):
    model.eval()

    all_y_cls_true = []
    all_y_cls_pred = []
    all_y_reg_scaled_true = []
    all_y_reg_scaled_pred = []

    # lấy ppm_min/max từ chính dataset
    ds = loader.dataset
    ppm_min = float(ds.ppm_min)
    ppm_max = float(ds.ppm_max)

    with torch.no_grad():
        for images, chem_idx, ppm_scaled, _ in loader:
            images = images.to(device)
            chem_idx = chem_idx.to(device)
            ppm_scaled = ppm_scaled.to(device)

            logits, reg_NH4, reg_NO2, _ = model(images)

            # classification prediction
            cls_pred = logits.argmax(dim=1)

            # regression prediction: chọn head theo class dự đoán
            mu_NH4 = reg_NH4[:, 0]  # (B,)
            mu_NO2 = reg_NO2[:, 0]  # (B,)
            reg_pred_scaled = torch.where(cls_pred == 0, mu_NH4, mu_NO2)

            all_y_cls_true.append(chem_idx.cpu().numpy())
            all_y_cls_pred.append(cls_pred.cpu().numpy())

            all_y_reg_scaled_true.append(ppm_scaled.cpu().numpy())
            all_y_reg_scaled_pred.append(reg_pred_scaled.cpu().numpy())

    y_cls_true = np.concatenate(all_y_cls_true)
    y_cls_pred = np.concatenate(all_y_cls_pred)
    y_reg_scaled_true = np.concatenate(all_y_reg_scaled_true)
    y_reg_scaled_pred = np.concatenate(all_y_reg_scaled_pred)

    # đưa về ppm thật
    y_reg_true = inverse_scale_ppm(
        y_reg_scaled_true, ppm_scale, ppm_min, ppm_max
    )
    y_reg_pred = inverse_scale_ppm(
        y_reg_scaled_pred, ppm_scale, ppm_min, ppm_max
    )

    # ---- classification metrics ----
    acc = float(accuracy_score(y_cls_true, y_cls_pred))
    f1 = float(f1_score(y_cls_true, y_cls_pred, average="macro"))

    # ---- regression metrics ----
    mae = float(mean_absolute_error(y_reg_true, y_reg_pred))
    mse = float(mean_squared_error(y_reg_true, y_reg_pred))
    rmse = float(math.sqrt(mse))
    r2 = float(r2_score(y_reg_true, y_reg_pred))

    eps = 1e-8
    mape = float(
        np.mean(
            np.abs((y_reg_true - y_reg_pred)
                   / np.clip(np.abs(y_reg_true), eps, None))
            * 100.0
        )
    )

    # per-class MAE / MAPE
    per_class = {}
    for cls_idx in np.unique(y_cls_true):
        mask = y_cls_true == cls_idx
        if mask.sum() == 0:
            continue
        mae_c = float(mean_absolute_error(y_reg_true[mask], y_reg_pred[mask]))
        mape_c = float(
            np.mean(
                np.abs(
                    (y_reg_true[mask] - y_reg_pred[mask])
                    / np.clip(np.abs(y_reg_true[mask]), eps, None)
                )
                * 100.0
            )
        )
        per_class[int(cls_idx)] = {"MAE": mae_c, "MAPE": mape_c}

    logs = {
        "acc": acc,
        "f1": f1,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "per_class": per_class,
    }
    return logs


def flatten_logs_for_csv(logs, extra_info):
    row = {
        "experiment": extra_info.get("experiment", ""),
        "dataset": extra_info.get("dataset", ""),
        "split": extra_info.get("split", ""),
        "ckpt_path": extra_info.get("ckpt_path", ""),
        "timm_name": extra_info.get("timm_name", ""),
        "image_size": extra_info.get("image_size", ""),
        "seed": extra_info.get("seed", ""),
    }

    row.update(
        {
            "acc": logs["acc"],
            "f1": logs["f1"],
            "mae": logs["mae"],
            "mse": logs["mse"],
            "rmse": logs["rmse"],
            "mape": logs["mape"],
            "r2": logs["r2"],
        }
    )

    per_class = logs.get("per_class", {})
    for cls_idx, stats in per_class.items():
        prefix = f"class_{cls_idx}"
        row[f"{prefix}_MAE"] = stats.get("MAE", float("nan"))
        row[f"{prefix}_MAPE"] = stats.get("MAPE", float("nan"))

    return row


def append_row_to_csv(csv_path, row):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not csv_path.exists()
    fieldnames = list(row.keys())

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--csv_path", type=str, required=False)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--meta_path", type=str, default=None)

    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path)
    if args.meta_path is not None:
        meta_path = Path(args.meta_path)
    else:
        meta_path = ckpt_path.with_suffix(".meta.json")

    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find meta json: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    timm_name = meta["timm_name"]
    image_size = int(meta["image_size"])
    ppm_scale = meta.get("ppm_scale", "none")
    seed = int(meta.get("seed", 42))

    set_seed(seed)

    # ----- data -----
    df = pd.read_csv(args.labels_csv)

    # label encoder như lúc train
    le = LabelEncoder()
    le.fit(df["chemical"].values)

    df_split = df[df["split"] == args.split].reset_index(drop=True)

    _, test_tfms = make_transforms(image_size, train=False)
    gnorm = GreenBorderNormalizer()

    ds = ChemistryDataset(
        df_split,
        root_dir=Path(args.root_dir),
        label_encoder=le,
        ppm_scale=ppm_scale,
        transform=test_tfms,
        gnorm=gnorm,
        image_size=(image_size, image_size),
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ----- model -----
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = MultiTaskHetero(
        timm_name=timm_name,
        num_classes=len(le.classes_),
        pretrained=False,
        drop=meta.get("drop", 0.0),
        drop_path=meta.get("drop_path", 0.0),
    )
    model.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    # ----- evaluate -----
    logs = evaluate_model(model, loader, device, ppm_scale)

    print(
        f"[INFO] Test | acc {logs['acc']:.4f} | f1 {logs['f1']:.4f} | "
        f"MAE {logs['mae']:.4f} | RMSE {logs['rmse']:.4f} | "
        f"MAPE {logs['mape']:.4f} | R2 {logs['r2']:.4f}"
    )
    print("[INFO] Done. Test:", json.dumps(logs, indent=2))

    # ----- save CSV -----
    if args.csv_path:
        exp_name = args.experiment_name or ckpt_path.stem
        row = flatten_logs_for_csv(
            logs,
            extra_info={
                "experiment": exp_name,
                "dataset": args.dataset_name,
                "split": args.split,
                "ckpt_path": str(ckpt_path),
                "timm_name": timm_name,
                "image_size": image_size,
                "seed": seed,
            },
        )
        append_row_to_csv(args.csv_path, row)
        print(f"[INFO] Appended metrics to CSV: {args.csv_path}")


if __name__ == "__main__":
    main()
