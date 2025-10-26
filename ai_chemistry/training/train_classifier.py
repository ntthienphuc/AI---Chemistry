import os, json, math, argparse, logging
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
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from albumentations import (
    Compose, Resize, Normalize, ToTensorV2,
    Rotate, Affine, RandomBrightnessContrast, GaussianBlur, HueSaturationValue
)
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ===================== Utils =====================
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD  = (0.229, 0.224, 0.225)

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def safe_mape(y_true, y_pred, eps=1e-6):
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

def inverse_scale(ppm_scaled, ppm_scale, ppm_min=None, ppm_max=None):
    if ppm_scale == 'log1p':
        return math.expm1(ppm_scaled)
    elif ppm_scale == 'minmax':
        return ppm_scaled * (ppm_max - ppm_min) + ppm_min
    return ppm_scaled

def srgb_to_linear(x):
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1+a)) ** 2.4)

def linear_to_srgb(x):
    a = 0.055
    return np.where(x <= 0.0031308, 12.92*x, (1+a)*(x ** (1/2.4)) - a)

# ================= Green Border Normalizer =================
class GreenBorderNormalizer:
    def __init__(self,
                 hsv_lower=(35, 40, 40), hsv_upper=(95, 255, 255),
                 ring_frac=0.08, inner_margin=2, min_green_pixels=300,
                 epsilon=1e-6, gamma=1.0):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.ring_frac = float(ring_frac)
        self.inner_margin = int(inner_margin)
        self.min_green_pixels = int(min_green_pixels)
        self.eps = float(epsilon)
        self.gamma = float(gamma)

    def _to_rgb01(self, img_bgr):
        if img_bgr.dtype not in (np.float32, np.float64):
            img_bgr = img_bgr.astype(np.float32) / 255.0
        return np.clip(img_bgr[..., ::-1], 0.0, 1.0)

    def _ring_mask(self, h, w, ring_px):
        m = np.zeros((h, w), dtype=np.uint8)
        m[:ring_px, :] = 255; m[-ring_px:, :] = 255
        m[:, :ring_px] = 255; m[:, -ring_px:] = 255
        im = self.inner_margin
        if 2*im < h and 2*im < w:
            m[im:-im, im:-im] = np.where(m[im:-im, im:-im] > 0, 0, m[im:-im, im:-im])
        return m

    def __call__(self, image_bgr):
        rgb = self._to_rgb01(image_bgr)
        h, w = rgb.shape[:2]
        ring_px = max(2, int(min(h, w) * self.ring_frac))
        ring = self._ring_mask(h, w, ring_px)

        img_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.bitwise_and(mask, ring)

        green_pixels = mask > 0
        if green_pixels.sum() < self.min_green_pixels:
            mask = ring; green_pixels = mask > 0

        if green_pixels.sum() == 0:
            mean_border = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        else:
            lin = srgb_to_linear(rgb)
            mean_border = lin[green_pixels].mean(axis=0).astype(np.float32)
            mean_border = np.clip(mean_border, 0.05, 1.0)

        lin = srgb_to_linear(rgb)
        norm_lin = lin / (mean_border[None, None, :] + self.eps)
        norm_lin = np.clip(norm_lin, 0.0, 1.0)
        norm = linear_to_srgb(norm_lin)
        return np.clip(norm, 0.0, 1.0).astype(np.float32)

# ================= Dataset =================
class ChemistryDataset(Dataset):
    def __init__(self, df, root_dir, chemical_encoder, ppm_scale='log1p',
                 transform=None, gnorm=None, target_size=(224, 224)):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.ppm_scale = ppm_scale
        self.transform = transform
        self.gnorm = gnorm or GreenBorderNormalizer()
        self.le = chemical_encoder
        self.target_size = target_size

        if self.ppm_scale == 'minmax':
            self.ppm_min = float(df['ppm'].min())
            self.ppm_max = float(df['ppm'].max())
        else:
            self.ppm_min, self.ppm_max = None, None

    def __len__(self): return len(self.df)

    def _scale_ppm(self, ppm):
        if self.ppm_scale == 'log1p':
            return math.log1p(ppm)
        elif self.ppm_scale == 'minmax':
            return (ppm - self.ppm_min) / (self.ppm_max - self.ppm_min + 1e-12)
        return ppm

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row['path']
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        img_norm = self.gnorm(img)  # RGB 0..1
        img_norm = cv2.resize(img_norm, self.target_size, interpolation=cv2.INTER_AREA)

        if self.transform is not None:
            out = self.transform(image=img_norm)
            img_t = out["image"]
        else:
            import torchvision.transforms as T
            to_tensor = T.Compose([T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD)])
            img_t = to_tensor(img_norm)

        chem_idx = int(self.le.transform([row['chemical']])[0])
        ppm_scaled = float(self._scale_ppm(float(row['ppm'])))
        return img_t, chem_idx, ppm_scaled, str(img_path)

# ================= Model (Two heads, heteroscedastic) =================
class MultiTaskHetero(nn.Module):
    """
    - Backbone timm (configurable by --timm_name)
    - head_cls: logits 2 lá»›p
    - head_reg_NH4/NO2: má»—i head tráº£ (mu, log_var)
    """
    def __init__(self, timm_name='efficientnetv2_s', num_classes=2, pretrained=True,
                 drop=0.2, drop_path=0.1):
        super().__init__()
        self.backbone = timm.create_model(
            timm_name, pretrained=pretrained, num_classes=0,
            drop_rate=drop, drop_path_rate=drop_path
        )
        feat_dim = getattr(self.backbone, 'num_features', None)
        if feat_dim is None:
            # fallback: try feature_info
            feat_dim = self.backbone.feature_info[-1]['num_chs']

        self.head_cls = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(512, num_classes)
        )
        # Each outputs (mu, log_var)
        self.head_reg_NH4 = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(512, 2)
        )
        self.head_reg_NO2 = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(512, 2)
        )

    def forward(self, x):
        feats = self.backbone(x)
        cls_out = self.head_cls(feats)
        reg_NH4 = self.head_reg_NH4(feats)  # (B,2) -> mu, log_var
        reg_NO2 = self.head_reg_NO2(feats)  # (B,2)
        return cls_out, reg_NH4, reg_NO2, feats

# ================= Losses =================
class GaussianNLLLossPerSample(nn.Module):
    """Return per-sample loss; use mean outside (for reweight/EMA etc.)"""
    def __init__(self):
        super().__init__()
    def forward(self, mu, log_var, target):
        # 0.5*(exp(-log_var) * (y-mu)^2 + log_var)
        inv_var = torch.exp(-log_var).clamp(max=1e6)
        return 0.5 * (inv_var * (target - mu)**2 + log_var)

def focal_loss(logits, target, alpha=0.75, gamma=2.0):
    # target: (B,) int64
    ce = nn.functional.cross_entropy(logits, target, reduction='none', label_smoothing=0.0)
    pt = torch.exp(-ce)
    return (alpha * (1-pt)**gamma * ce).mean()

# ================= Transforms =================
def make_transforms(image_size=224, train=True):
    if train:
        return Compose([
            Resize(image_size, image_size, interpolation=cv2.INTER_AREA),
            Rotate(limit=10, p=0.5),
            Affine(scale=(0.97, 1.03), translate_percent=0.02, shear=4, p=0.5),
            HueSaturationValue(hue_shift_limit=5, sat_shift_limit=12, val_shift_limit=6, p=0.4),
            RandomBrightnessContrast(brightness_limit=0.06, contrast_limit=0.06, p=0.3),
            GaussianBlur(blur_limit=(3, 3), p=0.15),
            Normalize(mean=IMNET_MEAN, std=IMNET_STD, max_pixel_value=1.0),
            ToTensorV2()
        ])
    else:
        return Compose([
            Resize(image_size, image_size, interpolation=cv2.INTER_AREA),
            Normalize(mean=IMNET_MEAN, std=IMNET_STD, max_pixel_value=1.0),
            ToTensorV2()
        ])

# ================= EMA =================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = v.detach().clone()
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
    def apply_to(self, model):
        state = model.state_dict()
        for k in self.shadow:
            if k in state:
                state[k].copy_(self.shadow[k])

# ================= Evaluate =================
@torch.no_grad()
def evaluate(model, loader, device, ppm_scale, ppm_min, ppm_max, prefix="Eval"):
    model.eval()
    y_true_cls, y_pred_cls = [], []
    y_true_ppm_s, y_pred_ppm_s = [], []

    per_class_true = {0: [], 1: []}
    per_class_pred = {0: [], 1: []}

    for images, cls_labels, reg_labels, _ in tqdm(loader, desc=prefix):
        images = images.to(device, non_blocking=True)
        cls_labels = cls_labels.to(device, non_blocking=True)
        reg_labels = reg_labels.to(device, non_blocking=True).float()

        cls_out, rNH4, rNO2, _ = model(images)
        pred_cls = torch.argmax(cls_out, dim=1)  # (B,)

        # chá»n mu theo lá»›p Dá»° ÄOÃN
        mu_NH4 = rNH4[:, 0]; mu_NO2 = rNO2[:, 0]
        mu_heads = torch.stack([mu_NH4, mu_NO2], dim=1)  # (B,2)
        reg_pred_scaled = mu_heads.gather(1, pred_cls.view(-1,1)).squeeze(1)

        y_pred_cls.extend(pred_cls.detach().cpu().numpy().tolist())
        y_true_cls.extend(cls_labels.detach().cpu().numpy().tolist())
        y_pred_ppm_s.extend(reg_pred_scaled.detach().cpu().numpy().tolist())
        y_true_ppm_s.extend(reg_labels.detach().cpu().numpy().tolist())

        # per-class theo NHÃƒN THáº¬T
        cls_true_np = cls_labels.detach().cpu().numpy()
        reg_true_np = reg_labels.detach().cpu().numpy()
        reg_pred_np = reg_pred_scaled.detach().cpu().numpy()
        for i, gt in enumerate(cls_true_np):
            per_class_true[gt].append(reg_true_np[i])
            per_class_pred[gt].append(reg_pred_np[i])

    acc = accuracy_score(y_true_cls, y_pred_cls)
    f1  = f1_score(y_true_cls, y_pred_cls, average='weighted')

    inv = lambda arr: [inverse_scale(v, ppm_scale, ppm_min, ppm_max) for v in arr]
    y_true_ppm = inv(y_true_ppm_s)
    y_pred_ppm = inv(y_pred_ppm_s)
    mae  = mean_absolute_error(y_true_ppm, y_pred_ppm)
    mape = safe_mape(y_true_ppm, y_pred_ppm)

    per_class_metrics = {}
    for k in [0,1]:
        if len(per_class_true[k]) == 0:
            per_class_metrics[k] = {"MAE": None, "MAPE": None}
        else:
            t = inv(per_class_true[k]); p = inv(per_class_pred[k])
            per_class_metrics[k] = {
                "MAE": float(mean_absolute_error(t, p)),
                "MAPE": float(safe_mape(t, p))
            }

    logs = {
        'acc': float(acc), 'f1': float(f1),
        'mae': float(mae), 'mape': float(mape),
        'per_class': per_class_metrics
    }
    logging.info(f"{prefix} | acc {acc:.4f} | f1 {f1:.4f} | MAE {mae:.4f} | MAPE {mape:.4f}")
    return logs

# ================= Train =================
def train(args):
    set_seed(args.seed)

    # ----- data -----
    labels = pd.read_csv(args.labels_csv)
    labels = labels[labels['chemical'].isin(['NH4','NO2'])].copy()
    assert len(labels) > 0, "labels.csv khÃ´ng cÃ³ máº«u NH4/NO2!"

    train_df = labels[labels['split']=='train'].reset_index(drop=True)
    val_df   = labels[labels['split']=='val'].reset_index(drop=True)
    test_df  = labels[labels['split']=='test'].reset_index(drop=True)

    le = LabelEncoder().fit(train_df['chemical'])
    classes = list(le.classes_)  # ['NH4','NO2']; giáº£ Ä‘á»‹nh NH4=0, NO2=1
    num_classes = len(classes)

    ppm_min = ppm_max = None
    if args.ppm_scale == 'minmax':
        ppm_min, ppm_max = float(train_df['ppm'].min()), float(train_df['ppm'].max())

    gnorm = GreenBorderNormalizer(ring_frac=args.ring_frac,
                                  inner_margin=args.inner_margin,
                                  min_green_pixels=args.min_green_pixels)

    t_train = make_transforms(args.image_size, train=True)
    t_eval  = make_transforms(args.image_size, train=False)

    train_ds = ChemistryDataset(train_df, args.root_dir, le, args.ppm_scale, t_train, gnorm, (args.image_size, args.image_size))
    val_ds   = ChemistryDataset(val_df,   args.root_dir, le, args.ppm_scale, t_eval,  gnorm, (args.image_size, args.image_size))
    test_ds  = ChemistryDataset(test_df,  args.root_dir, le, args.ppm_scale, t_eval,  gnorm, (args.image_size, args.image_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    logging.info(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    logging.info(f"Classes (label encoder order): {classes}")

    # ----- model -----
    model = MultiTaskHetero(
        timm_name=args.timm_name, num_classes=num_classes,
        pretrained=True, drop=0.2, drop_path=0.1
    ).to(args.device)

    # freeze backbone (warmup)
    for p in model.backbone.parameters():
        p.requires_grad = False

    # ----- losses -----
    if args.focal:
        crit_cls = None  # dÃ¹ng focal_loss() thá»§ cÃ´ng
    else:
        crit_cls = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    gauss_ps = GaussianNLLLossPerSample()

    # ----- optim, sched, amp, ema -----
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=1e-4)

    total_epochs = args.warmup_epochs + args.epochs
    def lr_lambda(epoch):
        # cosine with warmup (per-epoch)
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(1, args.warmup_epochs)
        t = (epoch - args.warmup_epochs) / max(1, args.epochs)
        return 0.5 * (1 + math.cos(math.pi * t))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    use_amp = args.device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    best_val = float('inf'); patience = 0
    score_hist = []

    # ----- training loop -----
    for epoch in range(total_epochs):
        model.train()
        if epoch == args.warmup_epochs:
            # unfreeze backbone
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        running_cls, running_reg = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} ({'warmup' if epoch<args.warmup_epochs else 'main'})")
        for images, cls_labels, reg_labels, _ in pbar:
            images = images.to(args.device, non_blocking=True)
            cls_labels = cls_labels.to(args.device, non_blocking=True)
            reg_labels = reg_labels.to(args.device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                cls_out, rNH4, rNO2, _ = model(images)
                # hetero: láº¥y theo NHÃƒN THáº¬T Ä‘á»ƒ tÃ­nh loss
                mu_NH4, lv_NH4 = rNH4[:,0], rNH4[:,1]
                mu_NO2, lv_NO2 = rNO2[:,0], rNO2[:,1]
                mu_heads = torch.stack([mu_NH4, mu_NO2], dim=1)  # (B,2)
                lv_heads = torch.stack([lv_NH4, lv_NO2], dim=1)  # (B,2)
                mu_true = mu_heads.gather(1, cls_labels.view(-1,1)).squeeze(1)
                lv_true = lv_heads.gather(1, cls_labels.view(-1,1)).squeeze(1)

                # cls loss
                if args.focal:
                    l_cls = focal_loss(cls_out, cls_labels, alpha=args.focal_alpha, gamma=args.focal_gamma)
                else:
                    l_cls = crit_cls(cls_out, cls_labels)

                # reg loss (per-sample -> mean)
                l_reg_ps = gauss_ps(mu_true, lv_true, reg_labels)  # (B,)
                l_reg = l_reg_ps.mean()

                loss = l_cls + args.lambda_reg * l_reg

            scaler.scale(loss).backward()
            # grad clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            running_cls += float(l_cls.item())
            running_reg += float(l_reg.item())
            pbar.set_postfix(cls=f"{running_cls/(pbar.n or 1):.4f}",
                             reg=f"{running_reg/(pbar.n or 1):.4f}")

        logging.info(f"Train | Cls {running_cls/len(train_loader):.4f} | Reg {running_reg/len(train_loader):.4f}")
        scheduler.step()

        # ---- validate (dÃ¹ng EMA weights náº¿u cÃ³) ----
        if ema is not None:
            # backup & apply ema
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.apply_to(model)

        val_logs = evaluate(model, val_loader, args.device, args.ppm_scale, ppm_min, ppm_max, prefix="Val")

        if ema is not None:
            # restore
            model.load_state_dict(backup, strict=False)

        # val_score: Æ°u tiÃªn reg + classification
        # (moving average Ä‘á»ƒ mÆ°á»£t hÆ¡n)
        val_score = (1.0 - val_logs['acc']) + val_logs['mae']
        score_hist.append(val_score)
        smooth = sum(score_hist[-3:]) / min(3, len(score_hist))

        if smooth < best_val - 1e-6:
            best_val = smooth; patience = 0
            Path(os.path.dirname(args.save_path) or ".").mkdir(parents=True, exist_ok=True)
            # save EMA weights náº¿u cÃ³
            if ema is not None:
                ema.apply_to(model)
            torch.save({
                'state_dict': model.state_dict(),
                'classes': classes,
                'ppm_scale': args.ppm_scale,
                'ppm_min': ppm_min, 'ppm_max': ppm_max,
                'image_size': args.image_size,
                'timm_name': args.timm_name,
                'two_reg_heads': True,
                'heteroscedastic': True
            }, args.save_path)
            logging.info("Saved best to %s", args.save_path)
            # restore training weights (khÃ´ng cáº§n náº¿u ema.apply_to dÃ¹ng copy)
            if ema is not None:
                model.load_state_dict(backup, strict=False)
        else:
            patience += 1
            if patience >= args.patience:
                logging.info("Early stopping.")
                break

    # ----- test -----
    ckpt = torch.load(args.save_path, map_location=args.device)
    model.load_state_dict(ckpt['state_dict'])
    test_logs = evaluate(model, test_loader, args.device, args.ppm_scale, ppm_min, ppm_max, prefix="Test")
    logging.info(f"Done. Test: {json.dumps(test_logs, indent=2)}")

    # save meta json
    meta = {
        'classes': classes,
        'ppm_scale': args.ppm_scale,
        'ppm_min': ppm_min, 'ppm_max': ppm_max,
        'image_size': args.image_size,
        'timm_name': args.timm_name,
        'two_reg_heads': True, 'heteroscedastic': True
    }
    with open(Path(args.save_path).with_suffix('.meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved meta to {Path(args.save_path).with_suffix('.meta.json')}")

# ================= CLI =================
def parse_args():
    ap = argparse.ArgumentParser("Train NH4/NO2 two-heads heteroscedastic + CosineLR+Warmup + EMA + GradClip + MA-earlystop")
    # data
    ap.add_argument('--root_dir',   type=str, default='data_clsreg/images')
    ap.add_argument('--labels_csv', type=str, default='data_clsreg/labels.csv')
    ap.add_argument('--image_size', type=int, default=224)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=2)
    # model/backbone
    ap.add_argument('--timm_name', type=str, default='efficientnetv2_s',
                    help="e.g. efficientnetv2_s | convnextv2_tiny.fcmae_ft_in22k_in1k | regnety_008 | nfnet_f0.dm_in1k")
    # train
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--warmup_epochs', type=int, default=5)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--lambda_reg', type=float, default=1.0)
    ap.add_argument('--ppm_scale', type=str, default='log1p', choices=['log1p','minmax','none'])
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save_path', type=str, default='training/weights/twoheads_hetero_best.pt')
    ap.add_argument('--seed', type=int, default=42)
    # green norm
    ap.add_argument('--ring_frac', type=float, default=0.08)
    ap.add_argument('--inner_margin', type=int, default=2)
    ap.add_argument('--min_green_pixels', type=int, default=300)
    # cls loss choices
    ap.add_argument('--focal', action='store_true', help='use focal loss for classification')
    ap.add_argument('--focal_alpha', type=float, default=0.75)
    ap.add_argument('--focal_gamma', type=float, default=2.0)
    ap.add_argument('--label_smoothing', type=float, default=0.05)
    # ema & clip
    ap.add_argument('--ema_decay', type=float, default=0.999)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    train(args)

