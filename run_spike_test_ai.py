from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from api.config import DEVICE as API_DEVICE
from api.config import MODEL_ZOO_EXPLICIT, YOLO_WEIGHTS
from api.predictor import LoadedPredictor
from api.roi import GreenRoiConfig, YoloRoiConfig, crop_roi_auto


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CALIB_ALIASES = {
    "green": "greenborder",
    "greenborder": "greenborder",
    "none": "none",
}


@dataclass(frozen=True)
class SpikeSample:
    path: Path
    chemical: str
    water_type: str
    device: str
    nominal_ppm: float
    target_ppm: float
    target_ppm_source: str


@dataclass(frozen=True)
class ModelEntry:
    model_key: str
    ckpt_path: Path
    meta_path: Optional[Path]
    train_calib: str


def normalize_calib_mode(mode: str) -> str:
    key = (mode or "").strip().lower()
    if key not in CALIB_ALIASES:
        valid = ", ".join(sorted(CALIB_ALIASES))
        raise ValueError(f"Unknown calib mode '{mode}'. Valid: {valid}")
    return CALIB_ALIASES[key]


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    if math.isfinite(out):
        return out
    return float("nan")


def parse_nominal_ppm(path: Path) -> Optional[float]:
    match = re.search(r"_(\d+(?:[.,]\d+)?)\s*ppm_", path.name, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def parse_device(path: Path) -> str:
    match = re.search(r"_Validation_([^_]+)_", path.name, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def load_note_values(folder: Path) -> List[float]:
    values: List[float] = []
    for note_path in sorted(folder.glob("*.txt")):
        try:
            for line in note_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    values.append(float(line.replace(",", ".")))
                except ValueError:
                    pass
        except UnicodeDecodeError:
            for line in note_path.read_text(errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    values.append(float(line.replace(",", ".")))
                except ValueError:
                    pass
    return values


def build_note_target_map(folder: Path, image_paths: Sequence[Path]) -> Dict[float, float]:
    note_values = load_note_values(folder)
    nominal_values = sorted(
        {
            nominal
            for path in image_paths
            for nominal in [parse_nominal_ppm(path)]
            if nominal is not None
        }
    )
    if len(note_values) != len(nominal_values):
        return {}
    return dict(zip(nominal_values, note_values))


def discover_spike_samples(spike_root: Path) -> List[SpikeSample]:
    if not spike_root.exists():
        raise FileNotFoundError(f"Spike root not found: {spike_root}")

    images_by_folder: Dict[Path, List[Path]] = {}
    for path in sorted(spike_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images_by_folder.setdefault(path.parent, []).append(path)

    samples: List[SpikeSample] = []
    for folder, image_paths in sorted(images_by_folder.items(), key=lambda item: str(item[0]).lower()):
        note_target_map = build_note_target_map(folder, image_paths)
        for path in sorted(image_paths):
            rel_parts = path.relative_to(spike_root).parts
            chemical = rel_parts[0] if len(rel_parts) >= 1 else ""
            water_type = rel_parts[1] if len(rel_parts) >= 2 else path.parent.name
            nominal = parse_nominal_ppm(path)
            if nominal is None:
                print(f"[WARN] Skip image without ppm in filename: {path}", file=sys.stderr)
                continue
            if nominal in note_target_map:
                target_ppm = note_target_map[nominal]
                target_source = "note"
            else:
                target_ppm = nominal
                target_source = "filename"
            samples.append(
                SpikeSample(
                    path=path,
                    chemical=chemical.upper(),
                    water_type=water_type,
                    device=parse_device(path),
                    nominal_ppm=nominal,
                    target_ppm=target_ppm,
                    target_ppm_source=target_source,
                )
            )

    if not samples:
        raise RuntimeError(f"No spike images found under: {spike_root}")
    return samples


def build_api_key_by_ckpt() -> Dict[Path, str]:
    out: Dict[Path, str] = {}
    for key, spec in MODEL_ZOO_EXPLICIT.items():
        try:
            out[spec.ckpt_path().resolve()] = key
        except Exception:
            continue
    return out


def infer_train_calib(path: Path) -> str:
    stem = path.stem.lower()
    if stem.endswith("_green"):
        return "green"
    if stem.endswith("_none"):
        return "none"
    return ""


def discover_models(weights_dir: Path, selected: Sequence[str]) -> List[ModelEntry]:
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights dir not found: {weights_dir}")

    api_key_by_ckpt = build_api_key_by_ckpt()
    entries: List[ModelEntry] = []
    for ckpt_path in sorted(weights_dir.rglob("*.pt")):
        if ckpt_path.name.lower() == "best.pt":
            continue
        ckpt_resolved = ckpt_path.resolve()
        model_key = api_key_by_ckpt.get(ckpt_resolved)
        if not model_key:
            model_key = ckpt_path.relative_to(weights_dir).with_suffix("").as_posix().replace("/", "__")
        meta_path = ckpt_path.with_suffix(".meta.json")
        entries.append(
            ModelEntry(
                model_key=model_key,
                ckpt_path=ckpt_path,
                meta_path=meta_path if meta_path.exists() else None,
                train_calib=infer_train_calib(ckpt_path),
            )
        )

    if selected and "all" not in {item.lower() for item in selected}:
        selected_set = {item.lower() for item in selected}
        entries = [
            entry
            for entry in entries
            if entry.model_key.lower() in selected_set
            or entry.ckpt_path.stem.lower() in selected_set
            or entry.ckpt_path.name.lower() in selected_set
        ]

    if not entries:
        raise RuntimeError("No model checkpoints selected.")
    return entries


def load_yolo_or_none(roi_mode: str, yolo_weights: Path) -> Optional[YOLO]:
    roi_mode = (roi_mode or "auto").strip().lower()
    if roi_mode in {"center", "green"}:
        return None
    if not yolo_weights.exists():
        if roi_mode == "yolo":
            raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")
        print(f"[WARN] YOLO weights not found, auto mode will use green/center fallback: {yolo_weights}")
        return None
    return YOLO(str(yolo_weights))


def metric_mean(values: Sequence[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def metric_rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    pairs = [(a, b) for a, b in zip(y_true, y_pred) if math.isfinite(a) and math.isfinite(b)]
    if not pairs:
        return float("nan")
    return math.sqrt(sum((a - b) ** 2 for a, b in pairs) / len(pairs))


def metric_r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    pairs = [(a, b) for a, b in zip(y_true, y_pred) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 2:
        return float("nan")
    ys = [a for a, _ in pairs]
    mean_y = sum(ys) / len(ys)
    ss_tot = sum((a - mean_y) ** 2 for a in ys)
    if ss_tot <= 0:
        return float("nan")
    ss_res = sum((a - b) ** 2 for a, b in pairs)
    return float(1.0 - ss_res / ss_tot)


def compute_summary(rows: Sequence[Dict[str, Any]], extra: Dict[str, Any]) -> Dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    y_true = [safe_float(row.get("target_ppm")) for row in ok_rows]
    y_pred = [safe_float(row.get("pred_ppm")) for row in ok_rows]
    abs_errors = [abs(a - b) for a, b in zip(y_true, y_pred) if math.isfinite(a) and math.isfinite(b)]
    sq_errors = [(a - b) ** 2 for a, b in zip(y_true, y_pred) if math.isfinite(a) and math.isfinite(b)]
    mape_values = [
        abs(a - b) / max(abs(a), 1e-8) * 100.0
        for a, b in zip(y_true, y_pred)
        if math.isfinite(a) and math.isfinite(b)
    ]

    class_total = len(ok_rows)
    class_correct = sum(
        1
        for row in ok_rows
        if str(row.get("chemical_true", "")).upper() == str(row.get("pred_chemical", "")).upper()
    )

    summary = dict(extra)
    summary.update(
        {
            "n_images": len(rows),
            "n_ok": len(ok_rows),
            "n_error": len(rows) - len(ok_rows),
            "class_acc": float(class_correct / class_total) if class_total else float("nan"),
            "mae": metric_mean(abs_errors),
            "mse": metric_mean(sq_errors),
            "rmse": metric_rmse(y_true, y_pred),
            "mape": metric_mean(mape_values),
            "r2": metric_r2(y_true, y_pred),
        }
    )
    return summary


def compute_group_summaries(rows: Sequence[Dict[str, Any]], extra: Dict[str, Any]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("chemical_true", "")),
            str(row.get("water_type", "")),
            str(row.get("target_ppm", "")),
        )
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for (chemical, water_type, target_ppm), group_rows in sorted(groups.items()):
        summary = compute_summary(
            group_rows,
            {
                **extra,
                "chemical_true": chemical,
                "water_type": water_type,
                "target_ppm": target_ppm,
            },
        )
        out.append(summary)
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def csv_safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def predict_one(
    sample: SpikeSample,
    predictor: LoadedPredictor,
    yolo: Optional[YOLO],
    roi_mode: str,
    yolo_cfg: YoloRoiConfig,
    green_cfg: GreenRoiConfig,
    debug_raw: bool,
) -> Dict[str, Any]:
    started = time.perf_counter()
    base_row: Dict[str, Any] = {
        "image_path": str(sample.path),
        "chemical_true": sample.chemical,
        "water_type": sample.water_type,
        "device": sample.device,
        "nominal_ppm": sample.nominal_ppm,
        "target_ppm": sample.target_ppm,
        "target_ppm_source": sample.target_ppm_source,
    }

    img = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
    if img is None:
        return {
            **base_row,
            "status": "error",
            "error": "cv2.imread failed",
            "elapsed_sec": time.perf_counter() - started,
        }

    try:
        roi_bgr, roi_source, bbox, padding, imgsz = crop_roi_auto(
            yolo=yolo,
            image_bgr=img,
            yolo_cfg=yolo_cfg,
            green_cfg=green_cfg,
            mode=roi_mode,
        )
        pred = predictor.predict(roi_bgr)
    except Exception as exc:
        return {
            **base_row,
            "status": "error",
            "error": repr(exc),
            "elapsed_sec": time.perf_counter() - started,
        }

    abs_error = abs(float(sample.target_ppm) - float(pred.ppm))
    row: Dict[str, Any] = {
        **base_row,
        "status": "ok",
        "error": "",
        "pred_chemical": pred.chemical,
        "pred_confidence": pred.chemical_conf,
        "pred_ppm": pred.ppm,
        "abs_error": abs_error,
        "ape": abs_error / max(abs(float(sample.target_ppm)), 1e-8) * 100.0,
        "ppm_ci95_low": pred.ppm_ci95[0] if pred.ppm_ci95 else "",
        "ppm_ci95_high": pred.ppm_ci95[1] if pred.ppm_ci95 else "",
        "ppm_sigma": pred.ppm_sigma if pred.ppm_sigma is not None else "",
        "roi_source": roi_source,
        "roi_bbox_x0": bbox[0],
        "roi_bbox_y0": bbox[1],
        "roi_bbox_x1": bbox[2],
        "roi_bbox_y1": bbox[3],
        "roi_padding": padding,
        "roi_imgsz": imgsz,
        "elapsed_sec": time.perf_counter() - started,
    }
    if debug_raw:
        row["raw_json"] = json.dumps(pred.raw, ensure_ascii=False)
    return row


def run_model_calib(
    model: ModelEntry,
    calib_mode: str,
    samples: Sequence[SpikeSample],
    yolo: Optional[YOLO],
    args: argparse.Namespace,
    predictions_dir: Path,
) -> Tuple[List[Dict[str, Any]], bool]:
    run_name = f"{model.model_key}__runtime_{calib_mode}"
    pred_path = predictions_dir / f"{csv_safe_name(run_name)}.csv"
    if pred_path.exists() and not args.overwrite:
        rows = read_csv_rows(pred_path)
        if len(rows) == len(samples):
            print(f"[SKIP] {run_name}: found complete prediction CSV ({len(rows)} rows)")
            return rows, True
        print(f"[INFO] {run_name}: existing CSV has {len(rows)} rows, rerunning because sample count is {len(samples)}")

    print(f"[LOAD] {run_name}")
    predictor = LoadedPredictor(
        ckpt_path=model.ckpt_path,
        meta_path=model.meta_path,
        device=args.device,
        calib_mode=calib_mode,
    )

    yolo_cfg = YoloRoiConfig(padding=args.yolo_padding, conf=args.yolo_conf, imgsz=args.yolo_imgsz)
    green_cfg = GreenRoiConfig(
        ratio_center=args.green_ratio_center,
        hsv_lower=(35, 40, 40),
        hsv_upper=(95, 255, 255),
        pad_ratio=args.green_pad_ratio,
        min_area_ratio=args.green_min_area_ratio,
    )

    rows: List[Dict[str, Any]] = []
    total = len(samples)
    for idx, sample in enumerate(samples, start=1):
        row = predict_one(
            sample=sample,
            predictor=predictor,
            yolo=yolo,
            roi_mode=args.roi_mode,
            yolo_cfg=yolo_cfg,
            green_cfg=green_cfg,
            debug_raw=args.debug_raw,
        )
        row.update(
            {
                "model_key": model.model_key,
                "train_calib": model.train_calib,
                "runtime_calib_mode": calib_mode,
                "ckpt_path": str(model.ckpt_path),
                "meta_path": str(model.meta_path or ""),
            }
        )
        rows.append(row)

        if idx == 1 or idx == total or idx % args.progress_every == 0:
            ok_count = sum(1 for item in rows if item.get("status") == "ok")
            print(f"[RUN] {run_name}: {idx}/{total} images, ok={ok_count}, error={idx - ok_count}")

    write_csv(pred_path, rows)

    del predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Spike_test_AI evaluation over all checkpoints in weights with greenborder and none calibration."
    )
    parser.add_argument("--spike-root", type=Path, default=Path("Spike_test_AI"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--out-dir", type=Path, default=Path("Results") / "spike_test_ai")
    parser.add_argument("--models", nargs="*", default=["all"], help="Model keys/stems to run, or all.")
    parser.add_argument("--calib-modes", nargs="+", default=["green", "none"], help="Runtime calib modes: green/greenborder/none.")
    parser.add_argument("--roi-mode", choices=["auto", "yolo", "green", "center"], default="auto")
    parser.add_argument("--device", default=API_DEVICE, help="cuda/cpu/mps. Defaults to api.config.DEVICE.")
    parser.add_argument("--yolo-weights", type=Path, default=YOLO_WEIGHTS)
    parser.add_argument("--yolo-padding", type=float, default=0.10)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--green-ratio-center", type=float, default=0.75)
    parser.add_argument("--green-pad-ratio", type=float, default=0.15)
    parser.add_argument("--green-min-area-ratio", type=float, default=0.002)
    parser.add_argument("--limit-models", type=int, default=0, help="Debug: run only first N discovered models.")
    parser.add_argument("--limit-images", type=int, default=0, help="Debug: run only first N spike images.")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--debug-raw", action="store_true", help="Store predictor raw JSON per image.")
    parser.add_argument("--overwrite", action="store_true", help="Rerun even when a complete prediction CSV exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.spike_root = args.spike_root.resolve()
    args.weights_dir = args.weights_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.yolo_weights = args.yolo_weights.resolve()
    args.calib_modes = [normalize_calib_mode(mode) for mode in args.calib_modes]

    samples = discover_spike_samples(args.spike_root)
    if args.limit_images > 0:
        samples = samples[: args.limit_images]

    models = discover_models(args.weights_dir, args.models)
    if args.limit_models > 0:
        models = models[: args.limit_models]

    expected_runs = len(models) * len(args.calib_modes)
    print(f"[INFO] Spike images: {len(samples)}")
    print(f"[INFO] Checkpoints: {len(models)}")
    print(f"[INFO] Runtime calib modes: {', '.join(args.calib_modes)}")
    print(f"[INFO] Planned runs: {expected_runs}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = args.out_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        args.out_dir / "samples_manifest.csv",
        [
            {
                "image_path": str(sample.path),
                "chemical_true": sample.chemical,
                "water_type": sample.water_type,
                "device": sample.device,
                "nominal_ppm": sample.nominal_ppm,
                "target_ppm": sample.target_ppm,
                "target_ppm_source": sample.target_ppm_source,
            }
            for sample in samples
        ],
    )

    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "spike_root": str(args.spike_root),
                "weights_dir": str(args.weights_dir),
                "out_dir": str(args.out_dir),
                "roi_mode": args.roi_mode,
                "device": args.device,
                "yolo_weights": str(args.yolo_weights),
                "n_images": len(samples),
                "n_models": len(models),
                "calib_modes": args.calib_modes,
                "planned_runs": expected_runs,
                "models": [
                    {
                        "model_key": model.model_key,
                        "ckpt_path": str(model.ckpt_path),
                        "meta_path": str(model.meta_path or ""),
                        "train_calib": model.train_calib,
                    }
                    for model in models
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    yolo = load_yolo_or_none(args.roi_mode, args.yolo_weights)

    run_summaries: List[Dict[str, Any]] = []
    group_summaries: List[Dict[str, Any]] = []
    started_all = time.perf_counter()

    for model in models:
        for calib_mode in args.calib_modes:
            run_started = time.perf_counter()
            rows, skipped = run_model_calib(
                model=model,
                calib_mode=calib_mode,
                samples=samples,
                yolo=yolo,
                args=args,
                predictions_dir=predictions_dir,
            )
            elapsed = time.perf_counter() - run_started
            extra = {
                "model_key": model.model_key,
                "train_calib": model.train_calib,
                "runtime_calib_mode": calib_mode,
                "ckpt_path": str(model.ckpt_path),
                "meta_path": str(model.meta_path or ""),
                "roi_mode": args.roi_mode,
                "skipped_existing": bool(skipped),
                "elapsed_sec": elapsed,
            }
            summary = compute_summary(rows, extra)
            run_summaries.append(summary)
            group_summaries.extend(compute_group_summaries(rows, extra))
            print(
                "[DONE] "
                f"{model.model_key} / {calib_mode}: "
                f"acc={summary['class_acc']:.4f} "
                f"mae={summary['mae']:.4f} "
                f"rmse={summary['rmse']:.4f} "
                f"mape={summary['mape']:.2f}% "
                f"errors={summary['n_error']}"
            )

            write_csv(args.out_dir / "summary_by_run.csv", run_summaries)
            write_csv(args.out_dir / "summary_by_group.csv", group_summaries)

    run_summaries_sorted = sorted(
        run_summaries,
        key=lambda row: (
            safe_float(row.get("mae")) if math.isfinite(safe_float(row.get("mae"))) else float("inf"),
            -(safe_float(row.get("class_acc")) if math.isfinite(safe_float(row.get("class_acc"))) else -1.0),
        ),
    )
    write_csv(args.out_dir / "leaderboard_by_mae.csv", run_summaries_sorted)

    total_elapsed = time.perf_counter() - started_all
    print(f"[INFO] All done in {total_elapsed / 60.0:.2f} minutes")
    print(f"[INFO] Summary: {args.out_dir / 'summary_by_run.csv'}")
    print(f"[INFO] Leaderboard: {args.out_dir / 'leaderboard_by_mae.csv'}")
    print(f"[INFO] Per-image predictions: {predictions_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
