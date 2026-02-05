"""
Group-aware dataset split (prevents leakage across near-duplicate images).

Default expected structure (same as current project):
  source_root/
    NH4/
      <device>/
        <round_or_session>/
          *.jpg|png
    NO2/
      <device>/
        <round_or_session>/
          *.jpg|png

BUT this script is robust to flatter layouts too:
- If images are directly under NH4/ or NO2/, it will use device="device0", round="round0".
- If images are under NH4/<device>/, it will use round="round0".
- If there are deeper levels, it will pack remaining folders into round name with "__".

Output structure:
  output_root/
    train/<chem>/<device>/<round>/image.jpg
    val/<chem>/<device>/<round>/image.jpg
    test/<chem>/<device>/<round>/image.jpg

Key features:
- Split by GROUP (chem + device + round) instead of individual images.
- Optional manifest JSON: freeze split for reproducibility.
- Optional holdout devices: force selected device(s) into TEST for cross-device protocol.
- Optional hardlinks to save disk (same drive), falls back to copy if unsupported.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

IMG_EXTS = ("jpg", "jpeg", "png")


@dataclass(frozen=True)
class GroupKey:
    chem: str
    device: str
    round_name: str

    def to_id(self) -> str:
        return f"{self.chem}|{self.device}|{self.round_name}"

    @staticmethod
    def from_id(s: str) -> "GroupKey":
        chem, device, round_name = s.split("|", 2)
        return GroupKey(chem=chem, device=device, round_name=round_name)


def _link_or_copy(src: Path, dst: Path, use_hardlinks: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if use_hardlinks:
        try:
            if dst.exists():
                return
            os.link(src, dst)  # hardlink
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def iter_image_files(source_root: Path, extensions: Iterable[str]) -> List[Tuple[Path, GroupKey]]:
    """
    Return list of (image_path, group_key).
    Grouping rule: (chem, device, round_name) derived from relative path under chem_dir.
    """
    items: List[Tuple[Path, GroupKey]] = []
    for chem_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        chem = chem_dir.name
        for ext in extensions:
            for img_path in chem_dir.rglob(f"*.{ext}"):
                rel = img_path.relative_to(chem_dir)
                parts = rel.parts[:-1]  # folders only
                if len(parts) >= 2:
                    device = parts[0]
                    round_name = "__".join(parts[1:])  # keep deeper hierarchy stable
                elif len(parts) == 1:
                    device = parts[0]
                    round_name = "round0"
                else:
                    device = "device0"
                    round_name = "round0"
                items.append((img_path, GroupKey(chem=chem, device=device, round_name=round_name)))
    return items


def validate_ratios(train: float, val: float, test: float) -> None:
    s = train + val + test
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"train+val+test must sum to 1.0, got {s:.6f}")


def build_groups(items: List[Tuple[Path, GroupKey]]) -> Dict[GroupKey, List[Path]]:
    groups: Dict[GroupKey, List[Path]] = {}
    for img_path, key in items:
        groups.setdefault(key, []).append(img_path)
    # sort within group for determinism
    for k in groups:
        groups[k] = sorted(groups[k])
    return groups


def compute_targets(total_images: int, train: float, val: float, test: float) -> Dict[str, int]:
    train_n = int(round(total_images * train))
    val_n = int(round(total_images * val))
    test_n = max(0, total_images - train_n - val_n)
    return {"train": train_n, "val": val_n, "test": test_n}


def assign_per_chem(
    groups: List[Tuple[GroupKey, List[Path]]],
    train: float,
    val: float,
    test: float,
    rng: random.Random,
) -> Dict[str, List[GroupKey]]:
    """Greedy group assignment to roughly match split ratios (by image count)."""
    total = sum(len(v) for _, v in groups)
    targets = compute_targets(total, train, val, test)

    rng.shuffle(groups)

    out: Dict[str, List[GroupKey]] = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for key, imgs in groups:
        if counts["train"] < targets["train"]:
            split = "train"
        elif counts["val"] < targets["val"]:
            split = "val"
        else:
            split = "test"
        out[split].append(key)
        counts[split] += len(imgs)

    return out


def build_split_mapping(
    grouped: Dict[GroupKey, List[Path]],
    train: float,
    val: float,
    test: float,
    seed: int,
    holdout_devices: List[str] | None = None,
) -> Dict[str, List[GroupKey]]:
    rng = random.Random(seed)
    holdout_devices = [d.strip() for d in (holdout_devices or []) if d.strip()]

    forced_test: List[GroupKey] = []
    remaining: Dict[GroupKey, List[Path]] = {}

    for k, imgs in grouped.items():
        if k.device in holdout_devices:
            forced_test.append(k)
        else:
            remaining[k] = imgs

    # keep balance per chemical
    by_chem: Dict[str, List[Tuple[GroupKey, List[Path]]]] = {}
    for k, imgs in remaining.items():
        by_chem.setdefault(k.chem, []).append((k, imgs))

    mapping: Dict[str, List[GroupKey]] = {"train": [], "val": [], "test": []}
    for chem, items in by_chem.items():
        sub = assign_per_chem(items, train, val, test, rng)
        mapping["train"].extend(sub["train"])
        mapping["val"].extend(sub["val"])
        mapping["test"].extend(sub["test"])

    mapping["test"].extend(forced_test)
    return mapping


def save_manifest(path: Path, mapping: Dict[str, List[GroupKey]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {split: [k.to_id() for k in keys] for split, keys in mapping.items()}
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> Dict[str, List[GroupKey]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return {split: [GroupKey.from_id(s) for s in ids] for split, ids in obj.items()}


def write_output(
    grouped: Dict[GroupKey, List[Path]],
    mapping: Dict[str, List[GroupKey]],
    output_root: Path,
    use_hardlinks: bool,
) -> None:
    split_of: Dict[GroupKey, str] = {}
    for split, keys in mapping.items():
        for k in keys:
            split_of[k] = split

    for k, imgs in grouped.items():
        split = split_of.get(k)
        if split is None:
            continue
        dst_dir = output_root / split / k.chem / k.device / k.round_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in imgs:
            _link_or_copy(src, dst_dir / src.name, use_hardlinks=use_hardlinks)


def summarize(grouped: Dict[GroupKey, List[Path]], mapping: Dict[str, List[GroupKey]]) -> None:
    def count_imgs(keys: List[GroupKey]) -> int:
        return sum(len(grouped.get(k, [])) for k in keys)

    print("===== Split summary (by images) =====")
    for split in ["train", "val", "test"]:
        keys = mapping.get(split, [])
        print(f"{split:>5}: groups={len(keys):4d}  images={count_imgs(keys):6d}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Group-aware split (prevents leakage across sessions/rounds).")
    ap.add_argument("--source-root", type=Path, required=True, help="Root folder containing NH4 and NO2 directories.")
    ap.add_argument("--output-root", type=Path, required=True, help="Destination folder for split dataset.")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=150)
    ap.add_argument("--manifest-json", type=Path, default=None,
                    help="If provided: save/load split manifest for reproducibility.")
    ap.add_argument("--holdout-devices", nargs="*", default=None,
                    help="Optional device name(s) forced into TEST (cross-device protocol).")
    ap.add_argument("--use-hardlinks", action="store_true",
                    help="Use hardlinks instead of copying (same drive). Falls back to copy if unsupported.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"source-root not found: {source_root}")

    validate_ratios(args.train, args.val, args.test)

    items = iter_image_files(source_root, IMG_EXTS)
    if len(items) == 0:
        raise ValueError("No images found. Check your folder structure and extensions (jpg/jpeg/png).")

    grouped = build_groups(items)

    if args.manifest_json and args.manifest_json.exists():
        mapping = load_manifest(args.manifest_json)
        print(f"[INFO] Loaded manifest: {args.manifest_json}")
    else:
        mapping = build_split_mapping(
            grouped=grouped,
            train=args.train,
            val=args.val,
            test=args.test,
            seed=args.seed,
            holdout_devices=args.holdout_devices,
        )
        if args.manifest_json:
            save_manifest(args.manifest_json, mapping)
            print(f"[INFO] Saved manifest: {args.manifest_json}")

    output_root.mkdir(parents=True, exist_ok=True)
    write_output(grouped, mapping, output_root, use_hardlinks=args.use_hardlinks)
    summarize(grouped, mapping)


if __name__ == "__main__":
    main()
