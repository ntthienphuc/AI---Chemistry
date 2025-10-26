"""Dataset splitting utilities for the chemistry project."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

SplitRatio = Dict[str, float]


def validate_ratios(ratios: SplitRatio) -> None:
    total = sum(ratios.values())
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")


def choose_split(index: int, cumulative_counts: Dict[str, int]) -> str:
    offset = 0
    for split, count in cumulative_counts.items():
        offset += count
        if index < offset:
            return split
    # Fallback for rounding issues.
    return next(reversed(cumulative_counts))


def compute_split_counts(num_items: int, ratios: SplitRatio) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    remaining = num_items
    split_items = list(ratios.items())
    for idx, (split, ratio) in enumerate(split_items):
        if idx == len(split_items) - 1:
            counts[split] = remaining
        else:
            count = int(round(num_items * ratio))
            counts[split] = count
            remaining -= count
    return counts


def iter_image_files(root_dir: Path, extensions: Iterable[str]) -> Iterator[Tuple[Path, Tuple[str, ...]]]:
    for chem_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        for device_dir in sorted(p for p in chem_dir.iterdir() if p.is_dir()):
            for round_dir in sorted(p for p in device_dir.iterdir() if p.is_dir()):
                for ext in extensions:
                    pattern = f"*.{ext.lstrip('.')}"
                    for img_path in round_dir.glob(pattern):
                        yield img_path, (chem_dir.name, device_dir.name, round_dir.name)


def split_dataset(
    source_root: Path,
    output_root: Path,
    ratios: SplitRatio | None = None,
    seed: int | None = None,
    extensions: Iterable[str] | None = None,
) -> None:
    if ratios is None:
        ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    if extensions is None:
        extensions = ("jpg", "jpeg", "png")

    validate_ratios(ratios)
    output_root.mkdir(parents=True, exist_ok=True)

    items = list(iter_image_files(source_root, extensions))
    if seed is not None:
        random.seed(seed)
    random.shuffle(items)

    counts = compute_split_counts(len(items), ratios)
    cumulative = {}
    running = 0
    for split, count in counts.items():
        running += count
        cumulative[split] = running

    for idx, (img_path, (chem, device, round_name)) in enumerate(items):
        split = choose_split(idx, cumulative)
        dest_dir = output_root / split / chem / device / round_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest_dir / img_path.name)


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split NH4/NO2 dataset into train/val/test folders.")
    parser.add_argument("--source-root", type=Path, required=True, help="Root folder containing NH4 and NO2 directories.")
    parser.add_argument("--output-root", type=Path, required=True, help="Destination folder for split dataset.")
    parser.add_argument("--train", type=float, default=0.7, help="Proportion of images for the train split.")
    parser.add_argument("--val", type=float, default=0.15, help="Proportion of images for the validation split.")
    parser.add_argument("--test", type=float, default=0.15, help="Proportion of images for the test split.")
    parser.add_argument("--seed", type=int, default=150, help="Random seed for reproducible shuffling.")
    return parser


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()
    ratios = {"train": args.train, "val": args.val, "test": args.test}
    split_dataset(args.source_root, args.output_root, ratios=ratios, seed=args.seed)


if __name__ == "__main__":
    main()
