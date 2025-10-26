"""Generate structured CSV labels from cropped ROI images."""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass
class LabelRecord:
    path: str
    chemical: str
    ppm: float
    device: str
    timestamp: str
    split: str


CHEMICAL_PATTERN = re.compile(r"_(NH4[\+\-]?|NO2[\+\-]?|NO3[\+\-]?|NH3[\+\-]?)_", re.IGNORECASE)
PPM_PATTERN = re.compile(r"(\d+(?:\.\d+)?)(p?pm)", re.IGNORECASE)
DEVICE_PATTERN = re.compile(r"_(An|Lam)_([A-Za-z0-9]+)_")
DATETIME_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}_\d{2}_\d{2})")


def normalize_chemical(raw: str) -> str:
    raw_upper = raw.upper()
    if "NH4" in raw_upper:
        return "NH4"
    if "NO2" in raw_upper:
        return "NO2"
    return "UNK"


def parse_filename(name: str) -> Tuple[str, float, str, str]:
    ppm_match = PPM_PATTERN.search(name)
    ppm = float(ppm_match.group(1)) if ppm_match else 0.0

    chem_match = CHEMICAL_PATTERN.search(name)
    chemical = normalize_chemical(chem_match.group(1)) if chem_match else "UNK"

    dev_match = DEVICE_PATTERN.search(name)
    if dev_match:
        device = dev_match.group(2)
    else:
        device = "UNK"

    dt_match = DATETIME_PATTERN.search(name)
    timestamp = dt_match.group(1).replace("_", ":") if dt_match else "UNK"

    return chemical, ppm, device, timestamp


def iter_images(base_dir: Path, extensions: Iterable[str]) -> Iterable[Tuple[Path, str]]:
    for split_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        split_name = split_dir.name
        for ext in extensions:
            for img_path in split_dir.rglob(f"*.{ext.lstrip('.')}"):
                yield img_path, split_name


def build_labels(
    images_root: Path,
    relative_to: Optional[Path] = None,
    extensions: Iterable[str] | None = None,
) -> List[LabelRecord]:
    exts = extensions or ("jpg", "jpeg", "png")
    base = images_root.resolve()
    rel_base = (relative_to or base.parent).resolve()

    records: List[LabelRecord] = []
    for img_path, split in iter_images(base, exts):
        chemical, ppm, device, timestamp = parse_filename(img_path.name)
        if chemical not in {"NH4", "NO2"}:
            continue
        rel_path = img_path.resolve().relative_to(rel_base).as_posix()
        records.append(LabelRecord(rel_path, chemical, ppm, device, timestamp, split))
    return records


def save_labels(records: List[LabelRecord], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "chemical", "ppm", "device", "datetime", "split"])
        for record in records:
            writer.writerow([record.path, record.chemical, record.ppm, record.device, record.timestamp, record.split])


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate labels.csv from cropped images.")
    parser.add_argument("--images-root", type=Path, required=True, help="Root folder containing split images.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Destination CSV file.")
    parser.add_argument(
        "--relative-to",
        type=Path,
        default=None,
        help="Optional base path used when writing relative paths into the CSV.",
    )
    return parser


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()
    records = build_labels(args.images_root, relative_to=args.relative_to)
    save_labels(records, args.output_csv)
    print(f"labels.csv created at {args.output_csv} with {len(records)} entries.")


if __name__ == "__main__":
    main()
