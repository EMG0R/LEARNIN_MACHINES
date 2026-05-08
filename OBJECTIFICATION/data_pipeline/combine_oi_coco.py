"""Combine OpenImages V7 + COCO 2017 prepared data into a unified data dir.

After running:
  - prepare_masks (for OI side, into data_v3/)
  - coco_to_masks (for COCO side, into data_coco/)

This script symlinks both sources into a single data_v5/ tree so train.py
reads them as one dataset (DATA_ROOT=data_v5).

OI image_ids (16-char hex like 'abc123def4567890') and COCO file IDs
('000000000139') don't collide, so they coexist in the same dir without
conflict.

Usage:
    python -m OBJECTIFICATION.data_pipeline.combine_oi_coco
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OI_ROOT   = ROOT / "data_v3"
COCO_ROOT = Path(os.environ.get("COCO_DATA_ROOT", str(ROOT / "data_coco")))
OUT_ROOT  = ROOT / "data_v5"


def _link_all(src_dir: Path, dst_dir: Path, ext: str):
    """Create relative symlinks from src_dir/*ext into dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src_dir.glob(f"*{ext}"):
        link = dst_dir / p.name
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(p.resolve())
        n += 1
    return n


def main():
    for split in ("train", "val"):
        for kind, ext in (("images", ".jpg"), ("masks", ".png")):
            dst = OUT_ROOT / split / kind
            n_oi = _link_all(OI_ROOT / split / kind, dst, ext)
            n_co = _link_all(COCO_ROOT / split / kind, dst, ext)
            total = sum(1 for _ in dst.glob(f"*{ext}"))
            print(f"  {split}/{kind:6s}  +{n_oi} OI  +{n_co} COCO  = {total} total")
    print(f"done. point training at: DATA_ROOT={OUT_ROOT}")


if __name__ == "__main__":
    main()
