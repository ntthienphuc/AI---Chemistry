"""
Clean labels CSV by dropping rows whose image files cannot be read with OpenCV.

Usage (from project root, e.g. D:\\Chemistry\\AI---Chemistry-main):

    python ai_chemistry\\data\\clean_labels.py ^
        --root-dir "D:\\Chemistry\\AI---Chemistry-main\\data_clsreg" ^
        --labels-csv "D:\\Chemistry\\AI---Chemistry-main\\data_clsreg\\labels_paper_lab10k.csv" ^
        --output-csv "D:\\Chemistry\\AI---Chemistry-main\\data_clsreg\\labels_paper_lab10k.clean.csv"

This will:
- Try to cv2.imread() each image at root_dir / path (from CSV 'path' column)
- Keep only rows whose image is successfully loaded
- Print total / bad / kept counts
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd


def clean_labels(root_dir: Path, labels_csv: Path, output_csv: Path) -> None:
    root_dir = root_dir.resolve()
    labels_csv = labels_csv.resolve()
    output_csv = output_csv.resolve()

    print(f"[INFO] Root dir: {root_dir}")
    print(f"[INFO] Input CSV: {labels_csv}")

    df = pd.read_csv(labels_csv)
    if "path" not in df.columns:
        raise KeyError("CSV must contain a 'path' column.")

    total = len(df)
    keep_indices = []
    bad_paths = []

    for idx, p in enumerate(df["path"].tolist()):
        img_path = root_dir / str(p)
        img = cv2.imread(str(img_path))
        if img is None:
            bad_paths.append(str(img_path))
        else:
            keep_indices.append(idx)

        if (idx + 1) % 1000 == 0:
            print(f"[INFO] Checked {idx+1}/{total} rows... bad so far: {len(bad_paths)}")

    df_clean = df.iloc[keep_indices].reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_csv, index=False)

    print(f"[INFO] TOTAL rows: {total}")
    print(f"[INFO] BAD images: {len(bad_paths)}")
    print(f"[INFO] KEPT rows: {len(df_clean)}")
    if bad_paths:
        print("[INFO] Example bad path (first 5):")
        for bp in bad_paths[:5]:
            print("   ", bp)


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean labels CSV by dropping unreadable images.")
    p.add_argument("--root-dir", type=Path, required=True, help="Root directory that prefixes all 'path' entries.")
    p.add_argument("--labels-csv", type=Path, required=True, help="Input labels CSV.")
    p.add_argument("--output-csv", type=Path, required=True, help="Output cleaned CSV.")
    return p


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()
    clean_labels(args.root_dir, args.labels_csv, args.output_csv)


if __name__ == "__main__":
    main()
