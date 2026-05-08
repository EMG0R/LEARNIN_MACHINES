"""COCO 2017 supplementary download.

Pulls annotations + only the train/val images that contain at least one of
our 14 merged classes. Combined with OI V7 data this boosts under-represented
classes that hit OI's segmentation ceiling (phone, skateboard, bowl, couch,
bag, etc.).

Usage:
    python -m OBJECTIFICATION.data_pipeline.coco_download --check   # HEAD-only, no downloads
    python -m OBJECTIFICATION.data_pipeline.coco_download --split val
    python -m OBJECTIFICATION.data_pipeline.coco_download --split train

Outputs land under OBJECTIFICATION/data_coco/{split}/{images,annotations}/.
Masks are produced separately by coco_to_masks.py from these annotations.

Resumable. Use --check first to verify URLs respond 200 without touching disk.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get(
    "COCO_DATA_ROOT",
    str(ROOT / "data_coco"),
))


# Annotations are one zip covering both train + val (~241 MB).
ANNOT_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Per-image URL pattern (used by --split mode for slow throttled fetches).
IMG_URL_FMT = "http://images.cocodataset.org/{split}2017/{file_name}"

# Full-split zip URLs (used by --zip mode — much faster, single 19 GB file
# via Google Cloud CDN instead of 90K throttled per-image fetches).
SPLIT_ZIP_URL = {
    "train": "http://images.cocodataset.org/zips/train2017.zip",
    "val":   "http://images.cocodataset.org/zips/val2017.zip",
}


# COCO category_id -> our merged class_id (matches OBJECTIFICATION/seg/classes.py).
# NOTE COCO IDs are not contiguous (1-90 with gaps).
COCO_TO_OURS = {
    1:  1,   # person -> person
    2:  2,   # bicycle -> vehicle
    3:  2,   # car -> vehicle
    4:  2,   # motorcycle -> vehicle
    6:  2,   # bus -> vehicle
    8:  2,   # truck -> vehicle
    41: 3,   # skateboard -> skateboard
    77: 4,   # cell phone -> phone
    72: 5,   # tv -> device
    73: 5,   # laptop -> device
    75: 5,   # remote -> device
    76: 5,   # keyboard -> device
    16: 6,   # bird -> animal
    17: 6,   # cat -> animal
    18: 6,   # dog -> animal
    64: 7,   # potted plant -> plant
    44: 8,   # bottle -> cup
    46: 8,   # wine glass -> cup
    47: 8,   # cup -> cup
    51: 9,   # bowl -> bowl
    63: 11,  # couch -> couch
    84: 12,  # book -> book
    27: 13,  # backpack -> bag
    31: 13,  # handbag -> bag
    # No COCO mapping for: footwear (10), guitar (14) — COCO has neither
}


def download_file(url: str, dest: Path, retries: int = 3, backoff: float = 2.0):
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except Exception as e:
            last_err = e
            if dest.exists():
                dest.unlink()
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
    print(f"  FAIL {url}: {last_err}", file=sys.stderr)
    return False


def head_check(url: str) -> int:
    """HEAD-only request to check URL availability without downloading."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status
    except Exception as e:
        return -1


def cmd_check():
    """Verify COCO URLs respond. No actual downloads."""
    print(f"=== HEAD-checking COCO endpoints ===")
    print(f"annot zip:      {head_check(ANNOT_ZIP_URL)}  {ANNOT_ZIP_URL}")
    sample_train = IMG_URL_FMT.format(split="train", file_name="000000000009.jpg")
    sample_val   = IMG_URL_FMT.format(split="val",   file_name="000000000139.jpg")
    print(f"sample train:   {head_check(sample_train)}  {sample_train}")
    print(f"sample val:     {head_check(sample_val)}  {sample_val}")
    print()
    print(f"COCO_TO_OURS maps {len(COCO_TO_OURS)} COCO categories -> 12 of our 14 merged classes")
    print(f"Not supplemented by COCO: footwear (10), guitar (14)")


def fetch_annotations():
    """Download + extract the trainval2017 annotations zip if not present."""
    annot_dir = DATA_ROOT / "annotations"
    expected = annot_dir / "instances_train2017.json"
    if expected.exists():
        print(f"  annotations already present at {annot_dir}")
        return annot_dir
    zip_path = DATA_ROOT / "annotations_trainval2017.zip"
    print(f"  downloading {ANNOT_ZIP_URL} (~241 MB) ...")
    download_file(ANNOT_ZIP_URL, zip_path)
    print(f"  extracting to {DATA_ROOT}/annotations/ ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_ROOT)
    return annot_dir


def filter_images_by_class(coco_json_path: Path):
    """Return (image_id -> file_name) for images containing >=1 target class."""
    with open(coco_json_path) as f:
        data = json.load(f)
    target_ids = set(COCO_TO_OURS.keys())
    images_with_target = set()
    for ann in data["annotations"]:
        if ann["category_id"] in target_ids:
            images_with_target.add(ann["image_id"])
    out = {}
    for img in data["images"]:
        if img["id"] in images_with_target:
            out[img["id"]] = img["file_name"]
    return out


def cmd_split(split: str):
    """Download annotations + images for a split. NO per-class cap — pulls every
    image containing any target class, guaranteeing max-available data per class.
    Resumable: skips files already on disk.
    """
    annot_dir = fetch_annotations()
    annot_path = annot_dir / f"instances_{split}2017.json"
    print(f"  filtering {annot_path} for our 14 classes ...")
    needed = filter_images_by_class(annot_path)
    print(f"  {len(needed)} images contain at least one target class")

    # Per-class instance counts in this split (for verification)
    with open(annot_path) as f:
        data = json.load(f)
    from collections import Counter
    per_class = Counter()
    for ann in data["annotations"]:
        cat_id = ann["category_id"]
        if cat_id in COCO_TO_OURS:
            per_class[COCO_TO_OURS[cat_id]] += 1
    print(f"  per-merged-class instance counts in {split}:")
    for cls in sorted(per_class.keys()):
        print(f"    class {cls:2d}: {per_class[cls]:>7d} instances")

    image_dir = DATA_ROOT / split / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    pending = [(iid, fn) for iid, fn in needed.items()
               if not (image_dir / fn).exists()]
    if not pending:
        print(f"  all {len(needed)} images already on disk")
        return
    print(f"  fetching {len(pending)} images with 8 workers ...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {
            ex.submit(download_file,
                      IMG_URL_FMT.format(split=split, file_name=fn),
                      image_dir / fn): iid
            for iid, fn in pending
        }
        for i, _ in enumerate(as_completed(futs), 1):
            if i % 500 == 0:
                print(f"    {i}/{len(pending)}", flush=True)
    final = sum(1 for _ in image_dir.glob("*.jpg"))
    print(f"  done. {final} images on disk at {image_dir}")


def cmd_zip(split: str):
    """Fast path: download the full split2017.zip + extract only the images
    we need. ~30-60 min total vs ~14h for per-image throttled fetches.
    """
    annot_dir = fetch_annotations()
    annot_path = annot_dir / f"instances_{split}2017.json"
    print(f"  filtering {annot_path} for our 14 classes ...")
    needed = filter_images_by_class(annot_path)
    needed_files = set(needed.values())
    print(f"  {len(needed_files)} target images")

    image_dir = DATA_ROOT / split / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    already = {p.name for p in image_dir.glob("*.jpg")}
    remaining = needed_files - already
    print(f"  {len(already)} already on disk, {len(remaining)} to extract")
    if not remaining:
        print("  done.")
        return

    zip_path = DATA_ROOT / f"{split}2017.zip"
    if not zip_path.exists():
        url = SPLIT_ZIP_URL[split]
        print(f"  downloading {url} (single ~19 GB file) ...")
        download_file(url, zip_path)
    print(f"  extracting {len(remaining)} target images from zip ...")
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        # Inside zip, files are at "{split}2017/{file_name}"
        for info in zf.infolist():
            base = Path(info.filename).name
            if base in remaining:
                with zf.open(info) as src, open(image_dir / base, "wb") as dst:
                    dst.write(src.read())
                n += 1
                if n % 5000 == 0:
                    print(f"    extracted {n}/{len(remaining)}", flush=True)
    print(f"  extracted {n} images. delete {zip_path} when done if you need disk back.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="HEAD-only URL check, no downloads")
    ap.add_argument("--split", choices=["train", "val"],
                    help="per-image download for split (slow, throttled)")
    ap.add_argument("--zip", choices=["train", "val"],
                    help="full-zip download + filtered extract (FAST)")
    args = ap.parse_args()
    if args.check:
        cmd_check()
        return
    if args.zip:
        cmd_zip(args.zip)
        return
    if args.split:
        cmd_split(args.split)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
