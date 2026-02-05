# ai_chemistry/training/test_cls_only.py
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
from sklearn.preprocessing import LabelEncoder

from gb_utils import (
    set_seed, make_transforms,
    GreenBorderNormalizer, IdentityNormalizer,
    cls_metrics, extract_device_from_path,
    load_meta
)


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

        rgb01 = self.norm(img)
        x = self.tfm(image=rgb01)["image"]
        y = int(self.le.transform([row["chemical"]])[0])
        device = row["device"] if "device" in row else extract_device_from_path(rel)
        return x, y, rel, device


class ClassifierModel(nn.Module):
    def __init__(self, timm_name: str, num_classes: int = 2, pretrained: bool = False, drop: float = 0.2):
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
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        f = self.backbone(x)
        return self.head(f)


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

    out["per_device"] = {d: cls_metrics(v["y"], v["p"]) for d, v in per_device.items()}
    return out


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
    df = df[df["split"] == args.split].reset_index(drop=True)

    le = LabelEncoder()
    le.fit(meta["classes"])

    tfm = make_transforms(int(meta["image_size"]), train=False)
    calib = meta.get("calibration", "green")
    normalizer = GreenBorderNormalizer() if calib == "green" else IdentityNormalizer()

    ds = StripClsDataset(df, Path(args.root_dir), le, tfm, normalizer, int(meta["image_size"]))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = ClassifierModel(meta["timm_name"], num_classes=len(le.classes_), pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    logs = evaluate(model, loader, device)
    print(json.dumps(logs, indent=2))


if __name__ == "__main__":
    main()
