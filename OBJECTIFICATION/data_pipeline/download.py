"""Filter OpenImages V7 segmentation annotations to our 23 merged
classes, then download only the matching images and instance masks.

Usage (from OBJECTIFICATION/data_pipeline):
    python download.py --split train --max-per-class 3000
    python download.py --split val --max-per-class 200

Resumable: skips files that already exist on disk.

URL notes (patched 2026-04-24):
- Seg annotation CSVs: use full split name (val→validation) via SPLIT_URL_NAME.
- Mask ZIPs: the flat /v5/{split_url}-masks-{hex}.zip pattern returns 403.
  Working URL uses the subdir form:
  https://storage.googleapis.com/openimages/v5/{split_url}-masks/{split_url}-masks-{hex}.zip
- Images: S3 URL uses full split name (validation, not val); IMG_URL_FMT uses {split_url}.
"""
import argparse
import csv
import json
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import os
ROOT = Path(__file__).resolve().parent.parent
# ANNOT_DIR holds the seg-annotation CSVs and the per-hex mask zips. These
# are downloaded once and reusable across runs — defaults to the original
# (potentially root-owned) location so we don't re-download ~5 GB of zips.
ANNOT_DIR = Path(os.environ.get(
    "OBJ_ANNOT_DIR",
    str(ROOT / "shared" / "datasets" / "openimages_v7" / "annotations"),
))
# DATA_DIR is where per-split images, instance_masks, and instance_index.json
# get written. OBJ_DATA_ROOT lets you point this at a user-owned location to
# bypass sudo issues with old root-owned downloads.
DATA_DIR  = Path(os.environ.get(
    "OBJ_DATA_ROOT",
    str(ROOT / "shared" / "datasets" / "openimages_v7"),
))
CLASS_MAP_PATH = ROOT / "seg" / "class_map.json"

SPLIT_URL_NAME = {"train": "train", "val": "validation", "test": "test"}

# OI V7 segmentation annotations (CSV per split)
SEG_ANNOT_URL = {
    "train": "https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv",
    "val":   "https://storage.googleapis.com/openimages/v5/validation-annotations-object-segmentation.csv",
    "test":  "https://storage.googleapis.com/openimages/v5/test-annotations-object-segmentation.csv",
}

# Image URLs follow this pattern (S3, public); split path uses full name (validation, not val)
IMG_URL_FMT = "https://s3.amazonaws.com/open-images-dataset/{split_url}/{image_id}.jpg"

# Mask ZIPs: subdir form (patched — flat /v5/{split_url}-masks-{hex}.zip returns 403)
MASK_ZIP_URL_FMT = (
    "https://storage.googleapis.com/openimages/v5/"
    "{split_url}-masks/{split_url}-masks-{hex}.zip"
)


def download_file(url: str, dest: Path):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as e:
            last_err = e
            if dest.exists():
                dest.unlink()
            if attempt < 3:
                time.sleep(1.5 * (2 ** attempt))  # 1.5s, 3s, 6s
    print(f"  FAIL {url}: {last_err}", file=sys.stderr)


def load_class_map():
    with open(CLASS_MAP_PATH) as f:
        m = json.load(f)
    return m["by_mid"]  # mid -> class_id


def filter_annotations(split: str, target_mids: set, max_per_class: int):
    """Return: dict image_id -> list of (mask_path_in_zip, mid)."""
    split_url = SPLIT_URL_NAME[split]
    csv_filename = f"{split_url}-annotations-object-segmentation.csv"
    csv_path = ANNOT_DIR / csv_filename
    if not csv_path.exists():
        url = SEG_ANNOT_URL[split]
        print(f"downloading {url} -> {csv_path}")
        download_file(url, csv_path)

    per_class_count = defaultdict(int)
    by_image = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row["LabelName"]
            if mid not in target_mids:
                continue
            if per_class_count[mid] >= max_per_class:
                continue
            image_id = row["ImageID"]
            mask_path = row["MaskPath"]  # e.g. "0/abc123_m01g317_xyz.png"
            by_image[image_id].append((mask_path, mid))
            per_class_count[mid] += 1
    print(f"  filtered {sum(len(v) for v in by_image.values())} masks across "
          f"{len(by_image)} unique images")
    return by_image


def fetch_mask_zip(split: str, hex_char: str) -> Path:
    """Download the mask zip for a hex bucket and return its path."""
    split_url = SPLIT_URL_NAME[split]
    url = MASK_ZIP_URL_FMT.format(split_url=split_url, hex=hex_char)
    dest = ANNOT_DIR / "mask_zips" / f"{split_url}-masks-{hex_char}.zip"
    if not dest.exists():
        print(f"  downloading mask zip {hex_char}...")
        download_file(url, dest)
    return dest


def extract_target_masks_from_zip(zip_path: Path, wanted_paths: set, out_dir: Path):
    """Extract only the masks we want from a zip."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"  SKIP extract — zip not found: {zip_path}", file=sys.stderr)
        return
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name in wanted_paths:
                target = out_dir / Path(name).name
                if target.exists():
                    continue
                with zf.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())


def download_images(split: str, image_ids, out_dir: Path, workers: int = 8):
    out_dir.mkdir(parents=True, exist_ok=True)
    split_url = SPLIT_URL_NAME[split]
    pending = []
    for image_id in image_ids:
        dest = out_dir / f"{image_id}.jpg"
        if dest.exists():
            continue
        pending.append((image_id, dest))
    if not pending:
        return
    print(f"  fetching {len(pending)} images with {workers} workers...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(download_file, IMG_URL_FMT.format(split_url=split_url, image_id=iid), dest): iid
            for iid, dest in pending
        }
        for i, fut in enumerate(as_completed(futs), 1):
            if i % 200 == 0:
                print(f"    {i}/{len(pending)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--max-per-class", type=int, default=3000)
    args = ap.parse_args()

    mid_to_class = load_class_map()
    target_mids = set(mid_to_class.keys())

    by_image = filter_annotations(args.split, target_mids, args.max_per_class)

    # Group required mask-paths by hex-bucket (first char of image_id)
    mask_paths_by_hex = defaultdict(set)
    for image_id, instances in by_image.items():
        h = image_id[0].lower()
        for mp, _ in instances:
            mask_paths_by_hex[h].add(mp)

    instance_dir = DATA_DIR / args.split / "instance_masks"
    image_dir    = DATA_DIR / args.split / "images"

    for hex_char, wanted in sorted(mask_paths_by_hex.items()):
        zp = fetch_mask_zip(args.split, hex_char)
        extract_target_masks_from_zip(zp, wanted, instance_dir)

    download_images(args.split, list(by_image.keys()), image_dir)

    # Save instance index for prepare_masks step
    index_path = DATA_DIR / args.split / "instance_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        iid: [(Path(mp).name, mid) for mp, mid in insts]
        for iid, insts in by_image.items()
    }
    with open(index_path, "w") as f:
        json.dump(serializable, f)
    print(f"done. wrote {index_path}")


if __name__ == "__main__":
    main()
