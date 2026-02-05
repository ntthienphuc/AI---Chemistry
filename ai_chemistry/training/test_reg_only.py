# ai_chemistry/training/test_reg_only.py
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import timm
import torch.nn as nn

from gb_utils import (
    set_seed, make_transforms,
    GreenBorderNormalizer, IdentityNormalizer,
    inverse_scale_ppm, reg_metrics,
    load_meta
)


class StripRegDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: Path, tfm, normalizer, ppm_scale, ppm_min, ppm_max):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.tfm = tfm
        self.norm = normalizer
        self.ppm_scale = ppm_scale
        self.ppm_min = ppm_min
        self.ppm_max = ppm_max

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel = str(row["path"])
        img_path = self.root_dir / rel
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        rgb01 = self.norm(img)
        x = self.tfm(image=rgb01)["image"]
        ppm = float(row["ppm"])
        return x, ppm, rel


class HeteroRegressor(nn.Module):
    def __init__(self, timm_name: str, pretrained: bool = False, drop: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                feat_dim = self.backbone(dummy).shape[-1]
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        f = self.backbone(x)
        out = self.head(f)
        mu = out[:, 0]
        log_var = out[:, 1].clamp(-10, 10)
        return mu, log_var


@torch.no_grad()
def evaluate(model, loader, device, ppm_scale, ppm_min, ppm_max):
    model.eval()
    y_true_ppm, y_pred_ppm = [], []

    for x, ppm, rel in loader:
        x = x.to(device)
        mu, log_var = model(x)
        mu = mu.detach().cpu().numpy()
        pred_ppm = inverse_scale_ppm(mu, ppm_scale, ppm_min, ppm_max)

        y_true_ppm.append(ppm.numpy())
        y_pred_ppm.append(pred_ppm)

    y_true_ppm = np.concatenate(y_true_ppm)
    y_pred_ppm = np.concatenate(y_pred_ppm)
    return reg_metrics(y_true_ppm, y_pred_ppm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, required=True)
    p.add_argument("--labels_csv", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--meta_path", type=str, default=None)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt_path)
    meta_path = Path(args.meta_path) if args.meta_path else (ckpt_path.parent / "best.meta.json")
    meta = load_meta(meta_path)

    set_seed(int(meta.get("seed", 42)))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.labels_csv)
    df = df[(df["chemical"] == meta["chemical"]) & (df["split"] == args.split)].reset_index(drop=True)

    tfm = make_transforms(int(meta["image_size"]), train=False)
    calib = meta.get("calibration", "green")
    normalizer = GreenBorderNormalizer() if calib == "green" else IdentityNormalizer()

    ds = StripRegDataset(df, Path(args.root_dir), tfm, normalizer,
                         meta.get("ppm_scale", "log1p"), meta.get("ppm_min", None), meta.get("ppm_max", None))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = HeteroRegressor(meta["timm_name"], pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    logs = evaluate(model, loader, device, meta.get("ppm_scale", "log1p"), meta.get("ppm_min", None), meta.get("ppm_max", None))
    print(json.dumps(logs, indent=2))


if __name__ == "__main__":
    main()
