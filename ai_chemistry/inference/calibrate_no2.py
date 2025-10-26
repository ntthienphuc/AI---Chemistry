"""Simple linear calibration helper for NO2 predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def calibrate_no2(csv_path: Path, output_csv: Path, save_model: bool = False) -> Tuple[float, float]:
    df = pd.read_csv(csv_path)
    mask = (df["gt_chem"].str.upper() == "NO2") & df["gt_ppm"].notna()
    df_no2 = df.loc[mask].copy()

    if df_no2.empty:
        raise ValueError("Calibration requires at least one NO2 sample with ground-truth ppm.")

    X = df_no2["pred_ppm"].values.reshape(-1, 1)
    y = df_no2["gt_ppm"].values

    model = LinearRegression().fit(X, y)
    a, b = float(model.coef_[0]), float(model.intercept_)

    df_no2["pred_ppm_calib"] = model.predict(X)

    mae_before = mean_absolute_error(y, df_no2["pred_ppm"])
    mae_after = mean_absolute_error(y, df_no2["pred_ppm_calib"])
    r2_before = r2_score(y, df_no2["pred_ppm"])
    r2_after = r2_score(y, df_no2["pred_ppm_calib"])

    print("===== NO2 Calibration =====")
    print(f"  Fit: gt ~= {a:.3f} * pred + {b:.3f}")
    print(f"  MAE before: {mae_before:.4f} -> after: {mae_after:.4f}")
    print(f"  R^2 before: {r2_before:.4f} -> after: {r2_after:.4f}")

    df["pred_ppm_calib"] = df["pred_ppm"]
    df.loc[mask, "pred_ppm_calib"] = model.predict(df.loc[mask, "pred_ppm"].values.reshape(-1, 1))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Calibrated results saved to {output_csv}")

    if save_model:
        np.savez(output_csv.with_suffix(".no2_calibration.npz"), a=a, b=b)
        print(f"Calibration parameters saved to {output_csv.with_suffix('.no2_calibration.npz')}")

    return a, b


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Linearly calibrate NO2 predictions using ground-truth ppm.")
    parser.add_argument("--csv-path", type=Path, required=True, help="CSV produced by the inference pipeline with gt data.")
    parser.add_argument("--output-csv", type=Path, default=Path("calibrated_results.csv"))
    parser.add_argument("--save-model", action="store_true", help="Persist the calibration coefficients alongside the CSV.")
    return parser


if __name__ == "__main__":
    args = build_cli().parse_args()
    calibrate_no2(args.csv_path, args.output_csv, args.save_model)
