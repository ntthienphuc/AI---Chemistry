# ai_chemistry/training/train_cls_only.py
import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import timm
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from gb_utils import (
    set_seed, make_transforms,
    GreenBorderNormalizer, IdentityNormalizer,
    cls_metrics, extract_device_from_path,
    save_meta
)

torch.backends.cudnn.benchmark = True


class StripClsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: Path, le: LabelEncoder, tfm, normalizer, image_size: int):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.le = le
        self.tfm = tfm
        self.norm = normalizer
        self.image_size = image_size

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel = str(row["path"])
        img_path = self.root_dir / rel
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        rgb01 = self.norm(img)  # RGB [0,1]
        out = self.tfm(image=rgb01)
        x = out["image"]

        y = int(self.le.transform([row["chemical"]])[0])
        device = row["device"] if "device" in row else extract_device_from_path(rel)
        return x, y, rel, device


class ClassifierModel(nn.Module):
    def __init__(self, timm_name: str, num_classes: int = 2, pretrained: bool = True, drop: float = 0.2, drop_path: float = 0.1):
        super().__init__()
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,       # feature extractor
        )
        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            # fallback: run a dummy
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                feat_dim = self.backbone(dummy).shape[-1]
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        f = self.backbone(x)
        return self.head(f)


def build_scheduler(optimizer, warmup_epochs: int, cosine_epochs: int):
    total = warmup_epochs + cosine_epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        t = (epoch - warmup_epochs) / max(1, cosine_epochs)
        return 0.5 * (1 + math.cos(math.pi * t))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    per_device = {}

    for x, y, rel, dev in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()

        y_np = y.numpy()
        y_true.append(y_np)
        y_pred.append(pred)

        dev_list = list(dev)
        for i in range(len(dev_list)):
            d = dev_list[i]
            per_device.setdefault(d, {"y": [], "p": []})
            per_device[d]["y"].append(int(y_np[i]))
            per_device[d]["p"].append(int(pred[i]))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    out = cls_metrics(y_true, y_pred)

    # per-device
    out["per_device"] = {}
    for d, pack in per_device.items():
        out["per_device"][d] = cls_metrics(pack["y"], pack["p"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, required=True)
    p.add_argument("--labels_csv", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="runs_cls")

    p.add_argument("--timm_name", type=str, required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)

    # ablation
    p.add_argument("--calibration", type=str, choices=["green", "none"], default="green")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    root_dir = Path(args.root_dir)
    df = pd.read_csv(args.labels_csv)

    # label encoder (stable)
    le = LabelEncoder()
    le.fit(sorted(df["chemical"].unique().tolist()))
    num_classes = len(le.classes_)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    tfm_train = make_transforms(args.image_size, train=True)
    tfm_eval  = make_transforms(args.image_size, train=False)

    normalizer = GreenBorderNormalizer() if args.calibration == "green" else IdentityNormalizer()

    train_ds = StripClsDataset(train_df, root_dir, le, tfm_train, normalizer, args.image_size)
    val_ds   = StripClsDataset(val_df,   root_dir, le, tfm_eval,  normalizer, args.image_size)
    test_ds  = StripClsDataset(test_df,  root_dir, le, tfm_eval,  normalizer, args.image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = ClassifierModel(args.timm_name, num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler, total_epochs = build_scheduler(optimizer, args.warmup_epochs, args.epochs)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_dir = Path(args.save_dir) / f"{args.timm_name.replace('/','_')}_cal-{args.calibration}_seed{args.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "best.pt"
    meta_path = save_dir / "best.meta.json"

    best_val = -1.0
    bad = 0

    for epoch in range(total_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
        for x, y, _, _ in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(loss=float(loss.detach().cpu()))

        scheduler.step()

        val_logs = evaluate(model, val_loader, device)
        val_score = val_logs["acc"]  # optimize accuracy for paper-1

        print(f"[VAL] epoch={epoch+1} acc={val_logs['acc']:.4f} f1={val_logs['f1_macro']:.4f}")

        if val_score > best_val + 1e-6:
            best_val = val_score
            bad = 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch+1}, ckpt_path)
            meta = {
                "timm_name": args.timm_name,
                "image_size": args.image_size,
                "calibration": args.calibration,
                "seed": args.seed,
                "classes": le.classes_.tolist(),
            }
            save_meta(meta_path, meta)
        else:
            bad += 1
            if bad >= args.patience:
                print("[INFO] Early stopping triggered.")
                break

    # final test (best checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    test_logs = evaluate(model, test_loader, device)
    print("[TEST]", json.dumps(test_logs, indent=2))


if __name__ == "__main__":
    main()
