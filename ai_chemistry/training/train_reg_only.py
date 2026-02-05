# ai_chemistry/training/train_reg_only.py
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import timm
from tqdm import tqdm

from gb_utils import (
    set_seed, make_transforms,
    GreenBorderNormalizer, IdentityNormalizer,
    scale_ppm, inverse_scale_ppm,
    reg_metrics, save_meta
)

torch.backends.cudnn.benchmark = True


class StripRegDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: Path, tfm, normalizer,
                 ppm_scale: str, ppm_min: float | None, ppm_max: float | None):
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
        y = float(scale_ppm(ppm, self.ppm_scale, self.ppm_min, self.ppm_max))
        return x, y, ppm, rel


class HeteroRegressor(nn.Module):
    """Outputs (mu, log_var) in scaled space."""
    def __init__(self, timm_name: str, pretrained: bool = True, drop: float = 0.2):
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
        out = self.head(f)          # (B,2)
        mu = out[:, 0]
        log_var = out[:, 1].clamp(-10, 10)
        return mu, log_var


def gaussian_nll(mu, log_var, target):
    inv_var = torch.exp(-log_var).clamp(max=1e6)
    return 0.5 * (inv_var * (target - mu) ** 2 + log_var)


def build_scheduler(optimizer, warmup_epochs: int, cosine_epochs: int):
    total = warmup_epochs + cosine_epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        t = (epoch - warmup_epochs) / max(1, cosine_epochs)
        return 0.5 * (1 + math.cos(math.pi * t))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), total


@torch.no_grad()
def evaluate(model, loader, device, ppm_scale, ppm_min, ppm_max):
    model.eval()
    y_true_scaled, y_pred_scaled = [], []
    y_true_ppm = []

    for x, y, ppm, rel in loader:
        x = x.to(device)
        y = y.to(device)

        mu, log_var = model(x)
        y_pred_scaled.append(mu.detach().cpu().numpy())
        y_true_scaled.append(y.detach().cpu().numpy())
        y_true_ppm.append(ppm.numpy())

    y_true_scaled = np.concatenate(y_true_scaled)
    y_pred_scaled = np.concatenate(y_pred_scaled)
    y_true_ppm = np.concatenate(y_true_ppm)

    y_pred_ppm = inverse_scale_ppm(y_pred_scaled, ppm_scale, ppm_min, ppm_max)
    out = reg_metrics(y_true_ppm, y_pred_ppm)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, required=True)
    p.add_argument("--labels_csv", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="runs_reg")

    p.add_argument("--chemical", type=str, required=True, choices=["NH4", "NO2"])
    p.add_argument("--timm_name", type=str, required=True)

    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--ppm_scale", type=str, default="log1p", choices=["log1p", "minmax", "none"])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)

    p.add_argument("--calibration", type=str, choices=["green", "none"], default="green")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    root_dir = Path(args.root_dir)
    df = pd.read_csv(args.labels_csv)

    # filter by chemical
    df = df[df["chemical"] == args.chemical].reset_index(drop=True)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    # scale stats from TRAIN only (important!)
    if args.ppm_scale == "minmax":
        ppm_min = float(train_df["ppm"].min())
        ppm_max = float(train_df["ppm"].max())
    else:
        ppm_min, ppm_max = None, None

    tfm_train = make_transforms(args.image_size, train=True)
    tfm_eval  = make_transforms(args.image_size, train=False)
    normalizer = GreenBorderNormalizer() if args.calibration == "green" else IdentityNormalizer()

    train_ds = StripRegDataset(train_df, root_dir, tfm_train, normalizer, args.ppm_scale, ppm_min, ppm_max)
    val_ds   = StripRegDataset(val_df,   root_dir, tfm_eval,  normalizer, args.ppm_scale, ppm_min, ppm_max)
    test_ds  = StripRegDataset(test_df,  root_dir, tfm_eval,  normalizer, args.ppm_scale, ppm_min, ppm_max)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = HeteroRegressor(args.timm_name, pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler, total_epochs = build_scheduler(optimizer, args.warmup_epochs, args.epochs)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_dir = Path(args.save_dir) / f"{args.chemical}_{args.timm_name.replace('/','_')}_cal-{args.calibration}_seed{args.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "best.pt"
    meta_path = save_dir / "best.meta.json"

    best_val = float("inf")
    bad = 0

    for epoch in range(total_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
        for x, y, ppm, rel in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                mu, log_var = model(x)
                loss = gaussian_nll(mu, log_var, y).mean()

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(loss=float(loss.detach().cpu()))

        scheduler.step()

        val_logs = evaluate(model, val_loader, device, args.ppm_scale, ppm_min, ppm_max)
        val_score = val_logs["mae"]
        print(f"[VAL] epoch={epoch+1} MAE={val_logs['mae']:.4f} RMSE={val_logs['rmse']:.4f} R2={val_logs['r2']:.4f}")

        if val_score < best_val - 1e-6:
            best_val = val_score
            bad = 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch+1}, ckpt_path)
            meta = {
                "task": "reg_only",
                "chemical": args.chemical,
                "timm_name": args.timm_name,
                "image_size": args.image_size,
                "ppm_scale": args.ppm_scale,
                "ppm_min": ppm_min,
                "ppm_max": ppm_max,
                "calibration": args.calibration,
                "seed": args.seed,
            }
            save_meta(meta_path, meta)
        else:
            bad += 1
            if bad >= args.patience:
                print("[INFO] Early stopping triggered.")
                break

    # final test
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    test_logs = evaluate(model, test_loader, device, args.ppm_scale, ppm_min, ppm_max)
    print("[TEST]", json.dumps(test_logs, indent=2))


if __name__ == "__main__":
    main()
