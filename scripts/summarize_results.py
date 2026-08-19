#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize Evaluation Receipts into Formatted Tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def main():
    parser = argparse.ArgumentParser("Summarize Evaluation JSON Receipts")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_csv", type=str, default="results/summary_table.csv")
    parser.add_argument("--output_markdown", type=str, default="results/summary_table.md")
    args = parser.parse_args()

    res_dir = Path(args.results_dir).resolve()
    json_files = list(res_dir.rglob("*.json"))

    if not json_files:
        print(f"No JSON receipt files found in {res_dir}")
        return

    records: List[Dict] = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            metrics = data.get("metrics", {})
            cls_m = metrics.get("classification", {})
            reg_m = metrics.get("regression", {})
            per_a = reg_m.get("per_analyte", {})

            rec = {
                "Receipt": jf.stem,
                "Checkpoint": data.get("checkpoint"),
                "Dataset": data.get("dataset"),
                "Split": data.get("split"),
                "Calib_Train": data.get("calib_train"),
                "Calib_Eval": data.get("calib_eval"),
                "Accuracy (%)": round(cls_m.get("accuracy", 0.0) * 100, 2),
                "Macro_F1": round(cls_m.get("f1_macro", 0.0), 4),
                "MAE (ppm)": round(reg_m.get("mae", 0.0), 4),
                "RMSE (ppm)": round(reg_m.get("rmse", 0.0), 4),
                "R2": round(reg_m.get("r2", 0.0), 4),
                "MAPE (%)": round(reg_m.get("mape", 0.0), 2),
                "NH4_MAE": round(per_a.get("NH4", {}).get("mae", 0.0), 4) if per_a.get("NH4") else None,
                "NO2_MAE": round(per_a.get("NO2", {}).get("mae", 0.0), 4) if per_a.get("NO2") else None,
            }
            records.append(rec)
        except Exception as e:
            print(f"Failed to parse {jf}: {e}")

    df = pd.DataFrame(records)
    if not df.empty:
        out_csv = Path(args.output_csv).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"Saved summary CSV: {out_csv}")

        out_md = Path(args.output_markdown).resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("# Benchmark Summary\n\n")
            f.write(df.to_markdown(index=False))
        print(f"Saved summary Markdown: {out_md}")
        print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()