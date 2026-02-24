# test_classifier.py
import argparse
import json
import math
from pathlib import Path
import csv

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

from train_classifier import (
    ChemistryDataset,
    MultiTaskHetero,
    GreenBorderNormalizer,
    IdentityNormalizer,
    make_transforms,
    set_seed,
)

def inverse_scale_ppm(y_scaled, ppm_scale, ppm_min, ppm_max):
    y_scaled = np.asarray(y_scaled, dtype=float)
    if ppm_scale == "log1p":
        return np.expm1(y_scaled)
    elif ppm_scale == "minmax":
        return y_scaled * (ppm_max - ppm_min) + ppm_min
    else:
        return y_scaled

def safe_mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred)/denom))*100.0)

def build_normalizer_from_mode(mode: str, ring_frac=0.08, inner_margin=2, min_green_pixels=300):
    mode = (mode or "greenborder").lower()
    if mode == "none":
        return IdentityNormalizer()
    if mode == "greenborder":
        return GreenBorderNormalizer(
            ring_frac=ring_frac,
            inner_margin=inner_margin,
            min_green_pixels=min_green_pixels
        )
    raise ValueError(f"Unknown calib mode: {mode}")

def evaluate_model(model, loader, device, ppm_scale, ppm_min, ppm_max):
    model.eval()

    all_y_cls_true = []
    all_y_cls_pred = []
    all_y_reg_scaled_true = []
    all_y_reg_scaled_pred = []

    with torch.no_grad():
        for images, chem_idx, ppm_scaled, _ in loader:
            images = images.to(device)
            chem_idx = chem_idx.to(device)
            ppm_scaled = ppm_scaled.to(device)

            logits, reg_NH4, reg_NO2, _ = model(images)

            cls_pred = logits.argmax(dim=1)

            mu_NH4 = reg_NH4[:, 0]
            mu_NO2 = reg_NO2[:, 0]
            reg_pred_scaled = torch.where(cls_pred == 0, mu_NH4, mu_NO2)

            all_y_cls_true.append(chem_idx.cpu().numpy())
            all_y_cls_pred.append(cls_pred.cpu().numpy())
            all_y_reg_scaled_true.append(ppm_scaled.cpu().numpy())
            all_y_reg_scaled_pred.append(reg_pred_scaled.cpu().numpy())

    y_cls_true = np.concatenate(all_y_cls_true)
    y_cls_pred = np.concatenate(all_y_cls_pred)
    y_reg_scaled_true = np.concatenate(all_y_reg_scaled_true)
    y_reg_scaled_pred = np.concatenate(all_y_reg_scaled_pred)

    y_reg_true = inverse_scale_ppm(y_reg_scaled_true, ppm_scale, ppm_min, ppm_max)
    y_reg_pred = inverse_scale_ppm(y_reg_scaled_pred, ppm_scale, ppm_min, ppm_max)

    acc = float(accuracy_score(y_cls_true, y_cls_pred))
    f1_macro = float(f1_score(y_cls_true, y_cls_pred, average="macro"))
    f1_weighted = float(f1_score(y_cls_true, y_cls_pred, average="weighted"))

    mae = float(mean_absolute_error(y_reg_true, y_reg_pred))
    mse = float(mean_squared_error(y_reg_true, y_reg_pred))
    rmse = float(math.sqrt(mse))
    r2 = float(r2_score(y_reg_true, y_reg_pred))
    mape = float(safe_mape(y_reg_true, y_reg_pred))

    per_class = {}
    eps = 1e-8
    for cls_idx in np.unique(y_cls_true):
        mask = (y_cls_true == cls_idx)
        if mask.sum() == 0:
            continue
        per_class[int(cls_idx)] = {
            "MAE": float(mean_absolute_error(y_reg_true[mask], y_reg_pred[mask])),
            "MAPE": float(np.mean(np.abs((y_reg_true[mask]-y_reg_pred[mask]) / np.clip(np.abs(y_reg_true[mask]), eps, None))) * 100.0),
        }

    logs = {
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
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
        "calib": extra_info.get("calib", ""),
        "acc": logs["acc"],
        "f1_macro": logs["f1_macro"],
        "f1_weighted": logs["f1_weighted"],
        "mae": logs["mae"],
        "mse": logs["mse"],
        "rmse": logs["rmse"],
        "mape": logs["mape"],
        "r2": logs["r2"],
    }
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

    # calibration override (nếu không set -> dùng meta)
    parser.add_argument("--calib", type=str, default=None, choices=[None, "none", "greenborder"],
                        help="Override calibration mode. If omitted, uses meta['calib'].")

    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path)
    meta_path = Path(args.meta_path) if args.meta_path else ckpt_path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find meta json: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    timm_name = meta["timm_name"]
    image_size = int(meta["image_size"])
    ppm_scale = meta.get("ppm_scale", "none")
    seed = int(meta.get("seed", 42))
    calib_mode = (args.calib if args.calib is not None else meta.get("calib", "greenborder"))

    ppm_min = meta.get("ppm_min", None)
    ppm_max = meta.get("ppm_max", None)

    set_seed(seed)

    df = pd.read_csv(args.labels_csv)
    df_split = df[df["split"] == args.split].reset_index(drop=True)

    # label encoder: dùng đúng class order đã lưu
    classes = meta.get("classes", None)
    if classes is None:
        # fallback: đọc từ ckpt nếu meta thiếu
        ckpt_tmp = torch.load(ckpt_path, map_location="cpu")
        classes = ckpt_tmp.get("classes", ["NH4","NO2"])
    le = LabelEncoder()
    le.classes_ = np.array(classes)

    # transforms
    test_tfms = make_transforms(image_size, train=False)

    # calibration
    # (nếu bạn muốn pass ring params riêng lúc test thì thêm args; hiện dùng mặc định)
    gnorm = build_normalizer_from_mode(calib_mode)

    ds = ChemistryDataset(
        df_split,
        root_dir=args.root_dir,
        chemical_encoder=le,
        ppm_scale=ppm_scale,
        ppm_min=ppm_min,
        ppm_max=ppm_max,
        transform=test_tfms,
        gnorm=gnorm
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = MultiTaskHetero(
        timm_name=timm_name,
        num_classes=len(classes),
        pretrained=False,
        drop=0.2,
        drop_path=0.1,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    logs = evaluate_model(model, loader, device, ppm_scale, ppm_min, ppm_max)

    print(
        f"[INFO] {args.split} | calib={calib_mode} | "
        f"acc {logs['acc']:.4f} | f1_macro {logs['f1_macro']:.4f} | "
        f"MAE {logs['mae']:.4f} | RMSE {logs['rmse']:.4f} | "
        f"MAPE {logs['mape']:.4f} | R2 {logs['r2']:.4f}"
    )
    print("[INFO] Done:", json.dumps(logs, indent=2))

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
                "calib": calib_mode,
            },
        )
        append_row_to_csv(args.csv_path, row)
        print(f"[INFO] Appended metrics to CSV: {args.csv_path}")

if __name__ == "__main__":
    main()