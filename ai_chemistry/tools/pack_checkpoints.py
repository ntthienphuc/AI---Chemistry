# pack_pt_from_dirs.py (update)
import argparse, os, sys, shutil
from pathlib import Path
import torch

def is_ckpt_dir(d: Path) -> bool:
    return (d / "data.pkl").exists() and ((d / ".data").exists() or (d / "data").exists())

def safe_torch_load(pkl_path: Path):
    # PyTorch 2.6+: default weights_only=True. Một số checkpoint cũ cần False.
    try:
        return torch.load(pkl_path, map_location="cpu")  # weights_only=True by default in 2.6
    except Exception as e1:
        print(f"[WARN] weights_only=True failed for {pkl_path.name}: {e1}")
        print("[WARN] Retrying with weights_only=False (chỉ dùng khi file là nguồn tin cậy).")
        return torch.load(pkl_path, map_location="cpu", weights_only=False)

def convert_dir(d: Path, outdir: Path):
    data_pkl = d / "data.pkl"
    name = d.name
    outdir.mkdir(parents=True, exist_ok=True)
    out_pt = outdir / f"{name}.pt"

    print(f"[INFO] Converting: {d} -> {out_pt}")
    obj = safe_torch_load(data_pkl)
    torch.save(obj, out_pt)
    print(f"[OK]  Saved: {out_pt} ({out_pt.stat().st_size/1e6:.1f} MB)")

    # copy meta nếu có
    cand_meta = d.parent / f"{name}.meta.json"
    if cand_meta.exists():
        shutil.copy2(cand_meta, outdir / cand_meta.name)
        print(f"[OK]  Copied meta: {cand_meta.name}")
    else:
        for m in d.parent.glob("*.meta.json"):
            if name in m.stem:
                shutil.copy2(m, outdir / m.name)
                print(f"[OK]  Copied meta (match): {m.name}")
                break

def main():
    ap = argparse.ArgumentParser("Pack folder checkpoints (data.pkl/.data/...) back to .pt")
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Một hoặc nhiều thư mục gốc (quét đệ quy).")
    ap.add_argument("--outdir", type=str, default=None,
                    help="Thư mục xuất .pt (mặc định: cạnh thư mục checkpoint).")
    args = ap.parse_args()

    roots = [Path(p).resolve() for p in args.roots]
    outdir_global = Path(args.outdir).resolve() if args.outdir else None

    any_found = False
    for root in roots:
        if not root.exists():
            print(f"[WARN] Root not found: {root}")
            continue

        if root.is_dir() and is_ckpt_dir(root):
            any_found = True
            convert_dir(root, outdir_global or root.parent)
            continue

        for d in root.rglob("*"):
            if d.is_dir() and is_ckpt_dir(d):
                any_found = True
                convert_dir(d, outdir_global or d.parent)

    if not any_found:
        print("[ERR] Không tìm được thư mục checkpoint nào có data.pkl/.data.")
        sys.exit(1)

    print("[DONE] All conversions finished.")

if __name__ == "__main__":
    main()
