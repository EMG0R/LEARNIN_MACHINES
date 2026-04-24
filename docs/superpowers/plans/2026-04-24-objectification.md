# OBJECTIFICATION Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a 23-class semantic-segmentation model (YOLO-flavored CSPDarknet backbone + U-Net-style FPN decoder) on a curated OpenImages V7 subset, producing a `best.pt` checkpoint that runs on Mac MPS.

**Architecture:** Single PyTorch project under `OBJECTIFICATION/`. CSPDarknet-lite encoder (C3 + SPPF blocks) → top-down FPN with lateral skips → 1×1 head → 24-channel softmax (23 classes + background). Trained on a class-merged OpenImages V7 subset (~10 GB, ~50K images) with weighted batch sampler and combined CE + Dice loss. From-scratch random init.

**Tech Stack:** Python 3, PyTorch (MPS backend), `torchvision`, `Pillow`, `opencv-python`, `numpy`, `requests`, `pytest`. No pretrained weights, no AMP, no EMA, no MixUp/CutMix (per `feedback_hagrid_v3_overengineering.md`).

**Spec:** `docs/superpowers/specs/2026-04-24-objectification-design.md`

**Out of scope (separate plans):** live renderer integration, OSC pipeline, combined live app, Pi/Hailo optimization.

---

## File Structure

```
OBJECTIFICATION/
├── seg/
│   ├── __init__.py            # empty package marker
│   ├── classes.py             # MERGE_GROUPS source-of-truth (English names)
│   ├── class_map.json         # generated: OI MID → merged class ID
│   ├── model.py               # C3, SPPF, Backbone, Neck, Head, ObjSegNet
│   ├── dataset.py             # OpenImagesSegDataset
│   ├── augment.py             # SegTransform (paired image+mask transforms)
│   ├── losses.py              # ce_dice_loss
│   ├── eval.py                # per_class_iou, macro_miou
│   ├── train.py               # training loop (env-var configured, hand_seg style)
│   ├── build_class_map.py     # one-shot script: OI CSV → class_map.json
│   └── checkpoints/           # best.pt + last.pt (gitignored)
├── data_pipeline/
│   ├── __init__.py
│   ├── download.py            # filter OI segmentation CSV + fetch images/masks
│   └── prepare_masks.py       # per-instance PNGs → single integer mask
├── shared/
│   └── datasets/openimages_v7/
│       ├── annotations/       # CSVs from OI
│       ├── train/{images,masks}/
│       └── val/{images,masks}/
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # fixtures (tiny synthetic images/masks)
│   ├── test_classes.py
│   ├── test_model.py
│   ├── test_dataset.py
│   ├── test_augment.py
│   ├── test_losses.py
│   ├── test_eval.py
│   └── test_prepare_masks.py
└── train_all.py               # orchestrator (port of HAND_JOB/train_all.py)
```

---

## Task 1: Scaffolding + class merge groups

**Files:**
- Create: `OBJECTIFICATION/seg/__init__.py`
- Create: `OBJECTIFICATION/seg/classes.py`
- Create: `OBJECTIFICATION/data_pipeline/__init__.py`
- Create: `OBJECTIFICATION/tests/__init__.py`
- Create: `OBJECTIFICATION/tests/test_classes.py`
- Create: `OBJECTIFICATION/.gitignore`

- [ ] **Step 1: Write the failing test**

```python
# OBJECTIFICATION/tests/test_classes.py
from OBJECTIFICATION.seg.classes import MERGE_GROUPS, CLASS_NAMES, NUM_CLASSES

def test_23_foreground_classes():
    assert NUM_CLASSES == 24  # 23 + background
    assert CLASS_NAMES[0] == "background"
    assert len(CLASS_NAMES) == 24

def test_every_class_has_at_least_one_oi_label():
    for class_id, oi_labels in MERGE_GROUPS.items():
        assert len(oi_labels) >= 1, f"class {class_id} has no OI labels"

def test_no_duplicate_oi_labels_across_classes():
    seen = set()
    for labels in MERGE_GROUPS.values():
        for label in labels:
            assert label not in seen, f"label '{label}' appears in multiple classes"
            seen.add(label)

def test_class_names_match_merge_groups_keys():
    # MERGE_GROUPS keyed by class_id 1..23; CLASS_NAMES[1..23] are the names
    for class_id in range(1, 24):
        assert class_id in MERGE_GROUPS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_classes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'OBJECTIFICATION.seg.classes'`

- [ ] **Step 3: Create the package files and class definitions**

```python
# OBJECTIFICATION/seg/__init__.py
# (empty)
```

```python
# OBJECTIFICATION/data_pipeline/__init__.py
# (empty)
```

```python
# OBJECTIFICATION/tests/__init__.py
# (empty)
```

```python
# OBJECTIFICATION/seg/classes.py
"""Source-of-truth class merge groups.

class_id 0 is background. class_ids 1..23 are foreground classes.
MERGE_GROUPS maps class_id -> list of OpenImages V7 English label names.
build_class_map.py converts these English names to OI MIDs via the
official class-descriptions CSV.
"""

CLASS_NAMES = [
    "background",   # 0
    "person",       # 1
    "vehicle",      # 2
    "skateboard",   # 3
    "phone",        # 4
    "device",       # 5
    "animal",       # 6
    "plant",        # 7
    "cup",          # 8
    "spork",        # 9
    "bowl",         # 10
    "footwear",     # 11
    "glasses",      # 12
    "headphones",   # 13
    "chair",        # 14
    "couch",        # 15
    "table",        # 16
    "lamp",         # 17
    "book",         # 18
    "clock",        # 19
    "bag",          # 20
    "guitar",       # 21
    "trumpet",      # 22
    "piano",        # 23
]

NUM_CLASSES = len(CLASS_NAMES)  # 24

MERGE_GROUPS = {
    1:  ["Person"],
    2:  ["Car", "Bicycle", "Motorcycle", "Bus", "Truck"],
    3:  ["Skateboard"],
    4:  ["Mobile phone"],
    5:  ["Television", "Laptop", "Computer monitor", "Tablet computer",
         "Computer keyboard", "Remote control"],
    6:  ["Bird", "Dog", "Cat"],
    7:  ["Tree", "Flower", "Plant", "Houseplant"],
    8:  ["Cup", "Bottle", "Wine glass", "Mug"],
    9:  ["Fork", "Knife", "Spoon"],
    10: ["Bowl", "Plate"],
    11: ["Footwear", "Boot", "Sandal", "High heels", "Sneakers"],
    12: ["Glasses", "Sunglasses"],
    13: ["Headphones"],
    14: ["Chair"],
    15: ["Couch"],
    16: ["Coffee table", "Kitchen & dining room table", "Desk"],
    17: ["Lamp"],
    18: ["Book"],
    19: ["Clock"],
    20: ["Handbag", "Backpack"],
    21: ["Guitar"],
    22: ["Trumpet"],
    23: ["Piano"],
}
```

```
# OBJECTIFICATION/.gitignore
shared/datasets/
seg/checkpoints/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_classes.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES
git add OBJECTIFICATION/seg/__init__.py OBJECTIFICATION/seg/classes.py \
        OBJECTIFICATION/data_pipeline/__init__.py \
        OBJECTIFICATION/tests/__init__.py OBJECTIFICATION/tests/test_classes.py \
        OBJECTIFICATION/.gitignore
git commit -m "feat(objectification): scaffolding + 23-class merge groups"
```

---

## Task 2: Build class_map.json from OI class descriptions

**Files:**
- Create: `OBJECTIFICATION/seg/build_class_map.py`

This is a one-shot utility script (not a recurring runtime path), so no unit tests — but it must run successfully end-to-end and produce a valid JSON.

- [ ] **Step 1: Write the script**

```python
# OBJECTIFICATION/seg/build_class_map.py
"""One-shot: download OpenImages V7 class descriptions, look up MIDs
for each English name in MERGE_GROUPS, write class_map.json.

Run from OBJECTIFICATION/seg/:
    python build_class_map.py
"""
import csv
import json
import sys
import urllib.request
from pathlib import Path

from classes import MERGE_GROUPS

OI_CLASS_CSV_URL = (
    "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv"
)
ANNOT_DIR = Path(__file__).resolve().parent.parent / "shared" / "datasets" / "openimages_v7" / "annotations"
CSV_PATH = ANNOT_DIR / "oidv7-class-descriptions.csv"
OUT_PATH = Path(__file__).resolve().parent / "class_map.json"

def download_csv():
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists():
        print(f"already have {CSV_PATH}")
        return
    print(f"downloading {OI_CLASS_CSV_URL}")
    urllib.request.urlretrieve(OI_CLASS_CSV_URL, CSV_PATH)

def load_name_to_mid():
    name_to_mid = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            mid, name = row[0].strip(), row[1].strip()
            name_to_mid[name.lower()] = mid
    return name_to_mid

def main():
    download_csv()
    name_to_mid = load_name_to_mid()
    out = {"by_mid": {}, "by_class": {}, "missing": []}
    for class_id, names in MERGE_GROUPS.items():
        out["by_class"][str(class_id)] = []
        for name in names:
            mid = name_to_mid.get(name.lower())
            if mid is None:
                out["missing"].append(name)
                print(f"WARN: no MID for '{name}'", file=sys.stderr)
                continue
            out["by_mid"][mid] = class_id
            out["by_class"][str(class_id)].append({"name": name, "mid": mid})
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    if out["missing"]:
        print(f"\nWARNING: {len(out['missing'])} name(s) missing — fix in classes.py and rerun")
        sys.exit(1)
    print(f"wrote {OUT_PATH} ({len(out['by_mid'])} MIDs across {len(MERGE_GROUPS)} classes)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES/OBJECTIFICATION/seg && python build_class_map.py`
Expected: prints `wrote .../class_map.json (N MIDs across 23 classes)` with no missing names.

If any names are missing (printed to stderr), edit `classes.py` to use the exact OI label spelling (case-insensitive lookup, but spelling matters — "Sneakers" vs "Sneaker"). Rerun until all resolve.

- [ ] **Step 3: Verify the generated JSON**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES/OBJECTIFICATION/seg && python -c "import json; d=json.load(open('class_map.json')); assert len(d['by_mid'])>=30 and len(d['by_class'])==23 and not d['missing']; print('OK', len(d['by_mid']), 'MIDs')"`
Expected: `OK <N> MIDs` (N around 35–45)

- [ ] **Step 4: Commit**

```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES
git add OBJECTIFICATION/seg/build_class_map.py OBJECTIFICATION/seg/class_map.json
git commit -m "feat(objectification): generate OI MID -> merged class map"
```

---

## Task 3: Mask preparation utility

**Files:**
- Create: `OBJECTIFICATION/data_pipeline/prepare_masks.py`
- Create: `OBJECTIFICATION/tests/test_prepare_masks.py`
- Create: `OBJECTIFICATION/tests/conftest.py`

Mask preparation is the deterministic, testable part of the data pipeline: given per-instance OI mask PNGs + their MIDs, produce a single uint8 integer mask where each pixel is the merged class ID.

- [ ] **Step 1: Write the conftest fixture**

```python
# OBJECTIFICATION/tests/conftest.py
"""Shared pytest fixtures."""
import numpy as np
import pytest
from PIL import Image

@pytest.fixture
def tiny_instance_masks(tmp_path):
    """Three 32×32 PNGs simulating OI instance masks for one image:
    - mask A: top-left 16×16 white (= person, MID '/m/01g317')
    - mask B: bottom-right 16×16 white (= chair, MID '/m/01mzpv')
    - mask C: a 4×4 white square overlapping A (= person again)
    """
    arr = np.zeros((32, 32), dtype=np.uint8)

    a = arr.copy(); a[0:16, 0:16] = 255
    b = arr.copy(); b[16:32, 16:32] = 255
    c = arr.copy(); c[8:12, 8:12] = 255

    pa = tmp_path / "img1_/m/01g317_a.png"
    pa.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(tmp_path / "a.png")
    Image.fromarray(b).save(tmp_path / "b.png")
    Image.fromarray(c).save(tmp_path / "c.png")
    return [
        (tmp_path / "a.png", "/m/01g317"),  # person
        (tmp_path / "b.png", "/m/01mzpv"),  # chair
        (tmp_path / "c.png", "/m/01g317"),  # person (overlap)
    ]

@pytest.fixture
def fake_class_map():
    return {"/m/01g317": 1, "/m/01mzpv": 14}  # person=1, chair=14
```

- [ ] **Step 2: Write the failing test**

```python
# OBJECTIFICATION/tests/test_prepare_masks.py
import numpy as np
from PIL import Image
from OBJECTIFICATION.data_pipeline.prepare_masks import combine_instance_masks

def test_merges_into_single_integer_mask(tiny_instance_masks, fake_class_map):
    out = combine_instance_masks(tiny_instance_masks, fake_class_map, image_size=(32, 32))
    arr = np.array(out)
    assert arr.shape == (32, 32)
    assert arr.dtype == np.uint8
    # top-left should be person (class 1)
    assert arr[5, 5] == 1
    # bottom-right should be chair (class 14)
    assert arr[20, 20] == 14
    # untouched corners should be background (0)
    assert arr[31, 0] == 0
    assert arr[0, 31] == 0

def test_unknown_mid_is_skipped(tmp_path, fake_class_map):
    arr = np.full((32, 32), 255, dtype=np.uint8)
    p = tmp_path / "u.png"
    Image.fromarray(arr).save(p)
    out = combine_instance_masks([(p, "/m/UNKNOWN")], fake_class_map, image_size=(32, 32))
    assert np.array(out).max() == 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_prepare_masks.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement prepare_masks.py**

```python
# OBJECTIFICATION/data_pipeline/prepare_masks.py
"""Combine per-instance OpenImages mask PNGs into a single integer mask
per image (uint8, pixel value = merged class ID, 0 = background).
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image

def combine_instance_masks(
    masks: Iterable[Tuple[Path, str]],
    mid_to_class: dict,
    image_size: Tuple[int, int],
) -> Image.Image:
    """Combine per-instance binary masks into a single integer mask.

    masks: iterable of (mask_path, oi_mid) pairs
    mid_to_class: dict OI MID -> merged class ID (1..23)
    image_size: (W, H) of the parent image; instance masks resize to this.
    Returns: PIL 'L' image, pixel value = class id (0 = background).
    Last-write-wins for overlapping instances.
    """
    W, H = image_size
    out = np.zeros((H, W), dtype=np.uint8)
    for mask_path, mid in masks:
        cls = mid_to_class.get(mid)
        if cls is None:
            continue
        m = Image.open(mask_path).convert("L").resize((W, H), Image.NEAREST)
        m_arr = (np.array(m) > 127)
        out[m_arr] = cls
    return Image.fromarray(out, mode="L")


def process_split(
    split_root: Path,
    instance_index: dict,  # image_id -> list of (mask_path, mid)
    mid_to_class: dict,
    out_dir: Path,
):
    """Process all images in a split. Produces out_dir/{image_id}.png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    overlap_count = 0
    for image_id, masks in instance_index.items():
        img_path = split_root / "images" / f"{image_id}.jpg"
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            size = im.size  # (W, H)
        merged = combine_instance_masks(masks, mid_to_class, size)
        merged.save(out_dir / f"{image_id}.png")
    print(f"wrote {len(list(out_dir.glob('*.png')))} masks to {out_dir}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_prepare_masks.py -v`
Expected: 2 PASSED

- [ ] **Step 6: Commit**

```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES
git add OBJECTIFICATION/data_pipeline/prepare_masks.py \
        OBJECTIFICATION/tests/conftest.py \
        OBJECTIFICATION/tests/test_prepare_masks.py
git commit -m "feat(objectification): mask combiner utility + tests"
```

---

## Task 4: OpenImages download script

**Files:**
- Create: `OBJECTIFICATION/data_pipeline/download.py`

This is an I/O-heavy script with network and filesystem side effects — no unit tests. Verification is end-to-end execution against a small subset.

- [ ] **Step 1: Write the script**

```python
# OBJECTIFICATION/data_pipeline/download.py
"""Filter OpenImages V7 segmentation annotations to our 23 merged
classes, then download only the matching images and instance masks.

Usage (from OBJECTIFICATION/data_pipeline):
    python download.py --split train --max-per-class 3000
    python download.py --split val --max-per-class 200

Resumable: skips files that already exist on disk.
"""
import argparse
import csv
import json
import sys
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANNOT_DIR = ROOT / "shared" / "datasets" / "openimages_v7" / "annotations"
DATA_DIR  = ROOT / "shared" / "datasets" / "openimages_v7"
CLASS_MAP_PATH = ROOT / "seg" / "class_map.json"

# OI V7 segmentation annotations (CSV per split)
SEG_ANNOT_URL = {
    "train": "https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv",
    "val":   "https://storage.googleapis.com/openimages/v5/validation-annotations-object-segmentation.csv",
    "test":  "https://storage.googleapis.com/openimages/v5/test-annotations-object-segmentation.csv",
}

# Image URLs follow this pattern (S3, public)
IMG_URL_FMT = "https://s3.amazonaws.com/open-images-dataset/{split}/{image_id}.jpg"

# Mask ZIPs are split into 16 buckets by first hex char of image_id
MASK_ZIP_URL_FMT = (
    "https://storage.googleapis.com/openimages/v5/"
    "{split_url}-masks/{split_url}-masks-{hex}.zip"
)
SPLIT_URL_NAME = {"train": "train", "val": "validation", "test": "test"}


def download_file(url: str, dest: Path):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"  FAIL {url}: {e}", file=sys.stderr)
        if dest.exists():
            dest.unlink()


def load_class_map():
    with open(CLASS_MAP_PATH) as f:
        m = json.load(f)
    return m["by_mid"]  # mid -> class_id


def filter_annotations(split: str, target_mids: set, max_per_class: int):
    """Return: dict image_id -> list of (mask_path_in_zip, mid)."""
    csv_path = ANNOT_DIR / f"{split}-annotations-object-segmentation.csv"
    if not csv_path.exists():
        print(f"downloading {SEG_ANNOT_URL[split]} -> {csv_path}")
        download_file(SEG_ANNOT_URL[split], csv_path)

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
            ex.submit(download_file, IMG_URL_FMT.format(split=split, image_id=iid), dest): iid
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
    serializable = {
        iid: [(Path(mp).name, mid) for mp, mid in insts]
        for iid, insts in by_image.items()
    }
    with open(index_path, "w") as f:
        json.dump(serializable, f)
    print(f"done. wrote {index_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run with a tiny per-class limit on val split**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES/OBJECTIFICATION/data_pipeline && python download.py --split val --max-per-class 5`
Expected: completes, `instance_index.json` exists, at least one image is in `shared/datasets/openimages_v7/val/images/`, at least one mask in `instance_masks/`.

(If a hex-bucket zip is huge and slow on first download, that's expected — it's cached after the first run.)

- [ ] **Step 3: Commit**

```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES
git add OBJECTIFICATION/data_pipeline/download.py
git commit -m "feat(objectification): OpenImages V7 segmentation downloader"
```

---

## Task 5: Wire mask preparation to download index

**Files:**
- Modify: `OBJECTIFICATION/data_pipeline/prepare_masks.py` (add `main()`)

- [ ] **Step 1: Add the `main()` entry point**

Add this to the bottom of `OBJECTIFICATION/data_pipeline/prepare_masks.py`:

```python
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent / "shared" / "datasets" / "openimages_v7"
    split_root = root / args.split
    index_path = split_root / "instance_index.json"
    with open(index_path) as f:
        index = json.load(f)

    cm_path = Path(__file__).resolve().parent.parent / "seg" / "class_map.json"
    with open(cm_path) as f:
        mid_to_class = json.load(f)["by_mid"]

    # Convert relative mask filenames to absolute paths
    instance_dir = split_root / "instance_masks"
    expanded = {
        iid: [(instance_dir / fname, mid) for fname, mid in entries]
        for iid, entries in index.items()
    }

    out_dir = split_root / "masks"
    process_split(split_root, expanded, mid_to_class, out_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run on the val data downloaded in Task 4**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES/OBJECTIFICATION/data_pipeline && python prepare_masks.py --split val`
Expected: prints `wrote N masks to .../val/masks` (N matches the number of images downloaded).

- [ ] **Step 3: Verify a generated mask is a valid integer mask**

Run:
```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES && python -c "
from pathlib import Path
import numpy as np
from PIL import Image
masks = list(Path('OBJECTIFICATION/shared/datasets/openimages_v7/val/masks').glob('*.png'))
assert masks, 'no masks generated'
arr = np.array(Image.open(masks[0]))
assert arr.dtype == np.uint8 and arr.ndim == 2
assert arr.max() <= 23, f'unexpected class id {arr.max()}'
print('OK', masks[0].name, 'shape', arr.shape, 'classes present:', sorted(np.unique(arr).tolist()))"
```
Expected: prints `OK ...` with at least one non-zero class.

- [ ] **Step 4: Commit**

```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES
git add OBJECTIFICATION/data_pipeline/prepare_masks.py
git commit -m "feat(objectification): prepare_masks CLI wired to download index"
```

---

## Task 6: Model — C3 block

**Files:**
- Create: `OBJECTIFICATION/seg/model.py`
- Create: `OBJECTIFICATION/tests/test_model.py`

- [ ] **Step 1: Write failing test**

```python
# OBJECTIFICATION/tests/test_model.py
import torch
from OBJECTIFICATION.seg.model import C3

def test_c3_preserves_spatial_dims():
    x = torch.randn(2, 64, 40, 40)
    block = C3(64, 64, n=2)
    y = block(x)
    assert y.shape == (2, 64, 40, 40)

def test_c3_changes_channels():
    x = torch.randn(2, 64, 40, 40)
    block = C3(64, 128, n=1)
    y = block(x)
    assert y.shape == (2, 128, 40, 40)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement C3**

```python
# OBJECTIFICATION/seg/model.py
"""YOLO-flavored CSPDarknet-lite backbone + U-Net-style FPN decoder
for 24-channel semantic segmentation (23 classes + background).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(ci, co, k=3, s=1, p=None):
    if p is None:
        p = k // 2
    return nn.Sequential(
        nn.Conv2d(ci, co, k, s, p, bias=False),
        nn.BatchNorm2d(co),
        nn.SiLU(inplace=True),
    )


class Bottleneck(nn.Module):
    """Residual bottleneck used inside C3."""
    def __init__(self, c, shortcut=True):
        super().__init__()
        self.conv1 = conv_bn_act(c, c, k=1)
        self.conv2 = conv_bn_act(c, c, k=3)
        self.add = shortcut

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.add else y


class C3(nn.Module):
    """CSP bottleneck with 3 convolutions (YOLOv5/v8-style).
    Splits via two 1x1 convs, runs n bottlenecks on one branch,
    concats with the other branch, fuses with a final 1x1 conv.
    """
    def __init__(self, ci, co, n=1, shortcut=True):
        super().__init__()
        c_h = co // 2  # hidden channel count per branch
        self.cv1 = conv_bn_act(ci, c_h, k=1)
        self.cv2 = conv_bn_act(ci, c_h, k=1)
        self.m   = nn.Sequential(*[Bottleneck(c_h, shortcut=shortcut) for _ in range(n)])
        self.cv3 = conv_bn_act(2 * c_h, co, k=1)

    def forward(self, x):
        a = self.m(self.cv1(x))
        b = self.cv2(x)
        return self.cv3(torch.cat([a, b], dim=1))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_model.py -v`
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES
git add OBJECTIFICATION/seg/model.py OBJECTIFICATION/tests/test_model.py
git commit -m "feat(objectification): C3 CSP bottleneck block"
```

---

## Task 7: Model — SPPF block

**Files:**
- Modify: `OBJECTIFICATION/seg/model.py`
- Modify: `OBJECTIFICATION/tests/test_model.py`

- [ ] **Step 1: Add failing test**

Append to `OBJECTIFICATION/tests/test_model.py`:

```python
from OBJECTIFICATION.seg.model import SPPF

def test_sppf_preserves_shape():
    x = torch.randn(2, 512, 10, 10)
    block = SPPF(512, 512, k=5)
    y = block(x)
    assert y.shape == (2, 512, 10, 10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_model.py::test_sppf_preserves_shape -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Add SPPF to model.py**

Append to `OBJECTIFICATION/seg/model.py`:

```python
class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (YOLOv5/v8). Three series 5x5
    maxpools form a multi-scale receptive field, concatenated and fused.
    """
    def __init__(self, ci, co, k=5):
        super().__init__()
        c_h = ci // 2
        self.cv1 = conv_bn_act(ci, c_h, k=1)
        self.cv2 = conv_bn_act(c_h * 4, co, k=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_model.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES
git add OBJECTIFICATION/seg/model.py OBJECTIFICATION/tests/test_model.py
git commit -m "feat(objectification): SPPF spatial pyramid pool block"
```

---

## Task 8: Model — Backbone

**Files:**
- Modify: `OBJECTIFICATION/seg/model.py`
- Modify: `OBJECTIFICATION/tests/test_model.py`

- [ ] **Step 1: Add failing test**

Append to `OBJECTIFICATION/tests/test_model.py`:

```python
from OBJECTIFICATION.seg.model import Backbone

def test_backbone_returns_four_pyramid_levels():
    x = torch.randn(1, 3, 320, 320)
    bb = Backbone()
    p2, p3, p4, p5 = bb(x)
    assert p2.shape == (1,  64,  80,  80)
    assert p3.shape == (1, 128,  40,  40)
    assert p4.shape == (1, 256,  20,  20)
    assert p5.shape == (1, 512,  10,  10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_model.py::test_backbone_returns_four_pyramid_levels -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement Backbone**

Append to `OBJECTIFICATION/seg/model.py`:

```python
class Backbone(nn.Module):
    """CSPDarknet-lite. Stem + 4 stages. Returns 4 feature pyramid levels.

    Input:  (B, 3, 320, 320)
    Output: (P2, P3, P4, P5) at strides 4, 8, 16, 32 with channels 64,128,256,512.
    """
    def __init__(self):
        super().__init__()
        self.stem = conv_bn_act(3, 32, k=3, s=2)        # 320 -> 160

        self.s1_down = conv_bn_act(32,  64,  k=3, s=2)  # 160 -> 80
        self.s1_c3   = C3(64,  64,  n=1)

        self.s2_down = conv_bn_act(64,  128, k=3, s=2)  # 80 -> 40
        self.s2_c3   = C3(128, 128, n=2)

        self.s3_down = conv_bn_act(128, 256, k=3, s=2)  # 40 -> 20
        self.s3_c3   = C3(256, 256, n=3)

        self.s4_down = conv_bn_act(256, 512, k=3, s=2)  # 20 -> 10
        self.s4_c3   = C3(512, 512, n=1)
        self.s4_sppf = SPPF(512, 512, k=5)

    def forward(self, x):
        x = self.stem(x)
        p2 = self.s1_c3(self.s1_down(x))            # 80x80
        p3 = self.s2_c3(self.s2_down(p2))           # 40x40
        p4 = self.s3_c3(self.s3_down(p3))           # 20x20
        p5 = self.s4_sppf(self.s4_c3(self.s4_down(p4)))  # 10x10
        return p2, p3, p4, p5
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_model.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES
git add OBJECTIFICATION/seg/model.py OBJECTIFICATION/tests/test_model.py
git commit -m "feat(objectification): CSPDarknet-lite backbone"
```

---

## Task 9: Model — Neck (top-down FPN)

**Files:**
- Modify: `OBJECTIFICATION/seg/model.py`
- Modify: `OBJECTIFICATION/tests/test_model.py`

- [ ] **Step 1: Add failing test**

Append to `OBJECTIFICATION/tests/test_model.py`:

```python
from OBJECTIFICATION.seg.model import Neck

def test_neck_top_down_fusion():
    p2 = torch.randn(1,  64, 80, 80)
    p3 = torch.randn(1, 128, 40, 40)
    p4 = torch.randn(1, 256, 20, 20)
    p5 = torch.randn(1, 512, 10, 10)
    neck = Neck()
    n2, n3, n4 = neck(p2, p3, p4, p5)
    assert n4.shape == (1, 256, 20, 20)
    assert n3.shape == (1, 128, 40, 40)
    assert n2.shape == (1,  64, 80, 80)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_model.py::test_neck_top_down_fusion -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement Neck**

Append to `OBJECTIFICATION/seg/model.py`:

```python
class Neck(nn.Module):
    """Top-down FPN. P5 -> upsample + lateral concat with P4 -> C3 fuse,
    repeat down to P2. Returns (N2, N3, N4) — N5 isn't used by the head.
    """
    def __init__(self):
        super().__init__()
        # Lateral 1x1 reductions before concat, to keep neck channels modest
        self.lat4 = conv_bn_act(512, 256, k=1)  # P5 -> match P4 channels
        self.fuse4 = C3(256 + 256, 256, n=1, shortcut=False)

        self.lat3 = conv_bn_act(256, 128, k=1)  # N4 -> match P3 channels
        self.fuse3 = C3(128 + 128, 128, n=1, shortcut=False)

        self.lat2 = conv_bn_act(128, 64, k=1)   # N3 -> match P2 channels
        self.fuse2 = C3(64 + 64, 64, n=1, shortcut=False)

    def _up(self, x):
        return F.interpolate(x, scale_factor=2, mode="nearest")

    def forward(self, p2, p3, p4, p5):
        n4 = self.fuse4(torch.cat([self._up(self.lat4(p5)), p4], dim=1))   # 20x20, 256
        n3 = self.fuse3(torch.cat([self._up(self.lat3(n4)), p3], dim=1))   # 40x40, 128
        n2 = self.fuse2(torch.cat([self._up(self.lat2(n3)), p2], dim=1))   # 80x80,  64
        return n2, n3, n4
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_model.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES
git add OBJECTIFICATION/seg/model.py OBJECTIFICATION/tests/test_model.py
git commit -m "feat(objectification): top-down FPN neck"
```

---

## Task 10: Model — Full ObjSegNet

**Files:**
- Modify: `OBJECTIFICATION/seg/model.py`
- Modify: `OBJECTIFICATION/tests/test_model.py`

- [ ] **Step 1: Add failing tests**

Append to `OBJECTIFICATION/tests/test_model.py`:

```python
from OBJECTIFICATION.seg.model import ObjSegNet

def test_full_model_output_shape():
    model = ObjSegNet(num_classes=24)
    x = torch.randn(2, 3, 320, 320)
    y = model(x)
    assert y.shape == (2, 24, 320, 320)

def test_param_count_in_range():
    model = ObjSegNet(num_classes=24)
    n = sum(p.numel() for p in model.parameters())
    assert 3_000_000 <= n <= 7_000_000, f"unexpected param count {n:,}"

def test_forward_no_nan():
    torch.manual_seed(0)
    model = ObjSegNet(num_classes=24)
    x = torch.randn(1, 3, 320, 320)
    y = model(x)
    assert torch.isfinite(y).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_model.py -v -k 'full_model or param_count or forward_no_nan'`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement ObjSegNet**

Append to `OBJECTIFICATION/seg/model.py`:

```python
class SegHead(nn.Module):
    """Final segmentation head. Takes N2 (stride 4) and upsamples 4x to
    full resolution, then 1x1 to num_classes channels.
    """
    def __init__(self, ci, num_classes):
        super().__init__()
        self.fuse = conv_bn_act(ci, 32, k=3)
        self.out  = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        x = self.fuse(x)
        return self.out(x)


class ObjSegNet(nn.Module):
    """Full model: backbone -> neck -> seg head. Returns logits (B, K, H, W)."""
    def __init__(self, num_classes=24):
        super().__init__()
        self.backbone = Backbone()
        self.neck = Neck()
        self.head = SegHead(64, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        p2, p3, p4, p5 = self.backbone(x)
        n2, _, _ = self.neck(p2, p3, p4, p5)
        return self.head(n2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_model.py -v`
Expected: 8 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES
git add OBJECTIFICATION/seg/model.py OBJECTIFICATION/tests/test_model.py
git commit -m "feat(objectification): ObjSegNet — backbone+neck+head"
```

---

## Task 11: Augmentation transforms

**Files:**
- Create: `OBJECTIFICATION/seg/augment.py`
- Create: `OBJECTIFICATION/tests/test_augment.py`

- [ ] **Step 1: Write failing tests**

```python
# OBJECTIFICATION/tests/test_augment.py
import numpy as np
import torch
from PIL import Image

from OBJECTIFICATION.seg.augment import SegTransform


def _img_mask(size=64):
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:size // 2, :, 0] = 255  # red top half
    arr[size // 2:, :, 2] = 255  # blue bottom half
    img = Image.fromarray(arr)
    m = np.zeros((size, size), dtype=np.uint8)
    m[:size // 2, :] = 1
    m[size // 2:, :] = 2
    mask = Image.fromarray(m)
    return img, mask


def test_train_returns_correct_shape_and_dtype():
    img, mask = _img_mask()
    t = SegTransform(img_size=320, mode="train")
    x, y = t(img, mask)
    assert x.shape == (3, 320, 320) and x.dtype == torch.float32
    assert y.shape == (320, 320)    and y.dtype == torch.long


def test_eval_is_deterministic():
    img, mask = _img_mask()
    t = SegTransform(img_size=320, mode="eval")
    x1, y1 = t(img, mask)
    x2, y2 = t(img, mask)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_mask_only_contains_valid_class_ids():
    img, mask = _img_mask()
    t = SegTransform(img_size=320, mode="train")
    _, y = t(img, mask)
    # input mask had values {0, 1, 2}; after rotation/crop only those + ignore=255 may appear
    valid = {0, 1, 2, 255}
    assert set(np.unique(y.numpy()).tolist()).issubset(valid)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_augment.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement augment.py**

```python
# OBJECTIFICATION/seg/augment.py
"""Paired image+mask transforms for semantic segmentation training.

Light, validated stack only — no MixUp / CutMix / RandAugment / EMA / AMP.
Per feedback_hagrid_v3_overengineering.md, stacking these on MPS collapses
training. HFlip + color jitter + small rotation + scale-and-crop only.
"""
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IGNORE_INDEX  = 255  # used for padding regions when mask doesn't fill the crop


class SegTransform:
    def __init__(self, img_size: int, mode: str = "train"):
        assert mode in ("train", "eval")
        self.img_size = img_size
        self.mode = mode

    def __call__(self, image: Image.Image, mask: Image.Image):
        if self.mode == "train":
            image, mask = self._random_scale_crop(image, mask)
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=IGNORE_INDEX)
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image,   random.uniform(0.8, 1.2))
            image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
            image = TF.adjust_hue(image,        random.uniform(-0.05, 0.05))
        else:
            image = TF.resize(image, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BILINEAR)
            mask  = TF.resize(mask,  [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST)

        x = TF.to_tensor(image)
        x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        y = torch.from_numpy(np.array(mask, dtype=np.int64))
        return x, y

    def _random_scale_crop(self, image, mask):
        """Random scale 0.8x..1.2x then random crop to img_size."""
        s = random.uniform(0.8, 1.2)
        W, H = image.size
        new_W, new_H = int(W * s), int(H * s)
        image = TF.resize(image, [new_H, new_W], interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  [new_H, new_W], interpolation=TF.InterpolationMode.NEAREST)

        # Pad if smaller than target
        pad_w = max(0, self.img_size - new_W)
        pad_h = max(0, self.img_size - new_H)
        if pad_w or pad_h:
            image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)
            mask  = TF.pad(mask,  [0, 0, pad_w, pad_h], fill=IGNORE_INDEX)
            new_W += pad_w
            new_H += pad_h

        # Random crop to img_size x img_size
        x0 = random.randint(0, new_W - self.img_size)
        y0 = random.randint(0, new_H - self.img_size)
        image = TF.crop(image, y0, x0, self.img_size, self.img_size)
        mask  = TF.crop(mask,  y0, x0, self.img_size, self.img_size)
        return image, mask
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_augment.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES
git add OBJECTIFICATION/seg/augment.py OBJECTIFICATION/tests/test_augment.py
git commit -m "feat(objectification): paired image+mask transforms"
```

---

## Task 12: Dataset class

**Files:**
- Create: `OBJECTIFICATION/seg/dataset.py`
- Create: `OBJECTIFICATION/tests/test_dataset.py`

- [ ] **Step 1: Write failing test**

```python
# OBJECTIFICATION/tests/test_dataset.py
import numpy as np
import torch
from PIL import Image

from OBJECTIFICATION.seg.augment import SegTransform
from OBJECTIFICATION.seg.dataset import OpenImagesSegDataset


def _seed_split(root, n=4):
    """Create n image+mask pairs in root/{images,masks}/."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    ids = []
    for i in range(n):
        iid = f"img{i:04d}"
        Image.fromarray(np.full((100, 100, 3), i * 30 + 50, dtype=np.uint8)).save(
            root / "images" / f"{iid}.jpg")
        m = np.zeros((100, 100), dtype=np.uint8)
        m[:50, :50] = (i % 23) + 1   # one foreground class per image
        Image.fromarray(m).save(root / "masks" / f"{iid}.png")
        ids.append(iid)
    return ids


def test_dataset_returns_correct_shapes(tmp_path):
    _seed_split(tmp_path)
    ds = OpenImagesSegDataset(tmp_path, transform=SegTransform(img_size=64, mode="eval"))
    x, y = ds[0]
    assert x.shape == (3, 64, 64)
    assert y.shape == (64, 64)
    assert y.dtype == torch.long


def test_dataset_skips_image_without_mask(tmp_path):
    _seed_split(tmp_path, n=2)
    # extra image with no matching mask
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(
        tmp_path / "images" / "orphan.jpg")
    ds = OpenImagesSegDataset(tmp_path, transform=SegTransform(img_size=64, mode="eval"))
    assert len(ds) == 2  # orphan is filtered out


def test_class_freq_counts_only_foreground(tmp_path):
    _seed_split(tmp_path, n=3)
    ds = OpenImagesSegDataset(tmp_path, transform=SegTransform(img_size=64, mode="eval"))
    freq = ds.class_freq(num_classes=24)
    assert freq.shape == (24,)
    assert freq[0] == 0  # background NOT counted as a foreground class
    assert freq.sum() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement dataset.py**

```python
# OBJECTIFICATION/seg/dataset.py
"""OpenImages V7 semantic-segmentation dataset.

Expects a directory tree:
    root/images/{image_id}.jpg
    root/masks/{image_id}.png   (uint8, pixel value = class id 0..23)

Built by OBJECTIFICATION/data_pipeline/{download,prepare_masks}.py.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class OpenImagesSegDataset(Dataset):
    def __init__(self, root, transform=None):
        root = Path(root)
        self.root = root
        self.transform = transform
        # Index = image stems that have BOTH an image and a mask file
        masks = {p.stem for p in (root / "masks").glob("*.png")}
        self.ids = sorted([
            p.stem for p in (root / "images").glob("*.jpg") if p.stem in masks
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        iid = self.ids[k]
        img  = Image.open(self.root / "images" / f"{iid}.jpg").convert("RGB")
        mask = Image.open(self.root / "masks"  / f"{iid}.png")
        if self.transform is not None:
            return self.transform(img, mask)
        return img, mask

    def class_freq(self, num_classes: int) -> np.ndarray:
        """Pixel count per class, summed over the whole dataset.
        Index 0 (background) is not counted (kept as 0 to avoid skewing
        loss/sampler weights toward background).
        """
        counts = np.zeros(num_classes, dtype=np.int64)
        for iid in self.ids:
            arr = np.array(Image.open(self.root / "masks" / f"{iid}.png"))
            uniq, c = np.unique(arr, return_counts=True)
            for u, n in zip(uniq, c):
                if 1 <= u < num_classes:
                    counts[u] += int(n)
        return counts

    def sample_weights(self, num_classes: int) -> np.ndarray:
        """Per-image sampling weight: 1 / sqrt(min_class_freq_in_image).
        Background ignored. Images with no foreground get weight 0.
        """
        global_freq = self.class_freq(num_classes).astype(np.float64)
        global_freq[global_freq == 0] = 1.0  # avoid div-by-zero for absent classes
        weights = np.zeros(len(self.ids), dtype=np.float64)
        for i, iid in enumerate(self.ids):
            arr = np.array(Image.open(self.root / "masks" / f"{iid}.png"))
            classes_in_image = [c for c in np.unique(arr) if 1 <= c < num_classes]
            if not classes_in_image:
                continue
            min_freq = min(global_freq[c] for c in classes_in_image)
            weights[i] = 1.0 / np.sqrt(min_freq)
        return weights
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_dataset.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES
git add OBJECTIFICATION/seg/dataset.py OBJECTIFICATION/tests/test_dataset.py
git commit -m "feat(objectification): OpenImagesSegDataset with class freq + sample weights"
```

---

## Task 13: Loss — combined CE + Dice

**Files:**
- Create: `OBJECTIFICATION/seg/losses.py`
- Create: `OBJECTIFICATION/tests/test_losses.py`

- [ ] **Step 1: Write failing tests**

```python
# OBJECTIFICATION/tests/test_losses.py
import torch
from OBJECTIFICATION.seg.losses import dice_loss, ce_dice_loss


def test_dice_perfect_prediction_is_near_zero():
    # 2-class problem, perfect one-hot logits
    target = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)  # (1, 2, 2)
    logits = torch.zeros(1, 2, 2, 2)  # (B, C, H, W)
    logits[0, 0, 0, 0] = 10; logits[0, 0, 1, 1] = 10  # predict class 0 strongly
    logits[0, 1, 0, 1] = 10; logits[0, 1, 1, 0] = 10  # predict class 1 strongly
    loss = dice_loss(logits, target, num_classes=2)
    assert loss.item() < 0.05


def test_dice_inverted_prediction_is_near_one():
    target = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)
    logits = torch.zeros(1, 2, 2, 2)
    logits[0, 1, 0, 0] = 10; logits[0, 1, 1, 1] = 10  # predict class 1 where it should be 0
    logits[0, 0, 0, 1] = 10; logits[0, 0, 1, 0] = 10  # predict class 0 where it should be 1
    loss = dice_loss(logits, target, num_classes=2)
    assert loss.item() > 0.9


def test_combined_loss_finite_random():
    torch.manual_seed(0)
    logits = torch.randn(2, 24, 32, 32)
    target = torch.randint(0, 24, (2, 32, 32), dtype=torch.long)
    loss = ce_dice_loss(logits, target, num_classes=24)
    assert torch.isfinite(loss).all()


def test_ignore_index_excluded_from_loss():
    """Pixels with ignore_index=255 contribute neither to CE nor Dice."""
    logits = torch.randn(1, 24, 4, 4)
    t_full = torch.zeros(1, 4, 4, dtype=torch.long)
    t_partial = t_full.clone()
    t_partial[0, 0, 0] = 255  # one pixel ignored
    l1 = ce_dice_loss(logits, t_full, num_classes=24)
    l2 = ce_dice_loss(logits, t_partial, num_classes=24)
    assert l1.item() != l2.item()  # they should differ but both be finite
    assert torch.isfinite(l2).all()


def test_class_weights_scale_loss():
    torch.manual_seed(0)
    logits = torch.randn(1, 24, 8, 8)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[0, 0, 0] = 1  # one pixel of class 1
    w_uniform = torch.ones(24)
    w_boosted = torch.ones(24); w_boosted[1] = 100.0
    l_uni  = ce_dice_loss(logits, target, num_classes=24, class_weights=w_uniform)
    l_boost = ce_dice_loss(logits, target, num_classes=24, class_weights=w_boosted)
    assert l_boost.item() > l_uni.item()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_losses.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement losses.py**

```python
# OBJECTIFICATION/seg/losses.py
"""Cross-entropy + multi-class Dice loss for semantic segmentation.

Combined loss: L = 0.5 * CE + 0.5 * Dice.
CE handles confidence; Dice handles small-object recall and class imbalance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = 255


def dice_loss(logits, target, num_classes, eps=1e-6, ignore_index=IGNORE_INDEX):
    """Multi-class soft Dice loss averaged over classes.

    logits: (B, C, H, W) raw scores
    target: (B, H, W) long, values in [0, C) or = ignore_index
    """
    valid = (target != ignore_index)
    target_clamped = target.clone()
    target_clamped[~valid] = 0
    one_hot = F.one_hot(target_clamped, num_classes=num_classes)  # (B, H, W, C)
    one_hot = one_hot.permute(0, 3, 1, 2).float()                  # (B, C, H, W)
    one_hot = one_hot * valid.unsqueeze(1).float()

    probs = F.softmax(logits, dim=1) * valid.unsqueeze(1).float()

    # Per-class Dice over the spatial+batch dims
    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dim=dims)
    denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def ce_dice_loss(logits, target, num_classes, class_weights=None,
                 ce_weight=0.5, dice_weight=0.5, ignore_index=IGNORE_INDEX):
    """Combined cross-entropy + Dice loss."""
    ce = F.cross_entropy(
        logits, target, weight=class_weights, ignore_index=ignore_index
    )
    dl = dice_loss(logits, target, num_classes=num_classes, ignore_index=ignore_index)
    return ce_weight * ce + dice_weight * dl
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_losses.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES
git add OBJECTIFICATION/seg/losses.py OBJECTIFICATION/tests/test_losses.py
git commit -m "feat(objectification): combined CE + Dice loss with ignore + class weights"
```

---

## Task 14: Evaluation — per-class IoU + macro mIoU

**Files:**
- Create: `OBJECTIFICATION/seg/eval.py`
- Create: `OBJECTIFICATION/tests/test_eval.py`

- [ ] **Step 1: Write failing tests**

```python
# OBJECTIFICATION/tests/test_eval.py
import torch
from OBJECTIFICATION.seg.eval import ConfusionAccumulator, per_class_iou, macro_miou


def test_perfect_prediction_iou_is_one():
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[0, :4, :] = 1
    target[0, 4:, :] = 2
    # logits that argmax to the same as target
    logits = torch.zeros(1, 3, 8, 8)
    logits[0, 0, :, :] = 1.0
    logits[0, 1, :4, :] = 5.0
    logits[0, 2, 4:, :] = 5.0
    acc = ConfusionAccumulator(num_classes=3)
    acc.update(logits, target)
    iou = per_class_iou(acc.confusion)
    assert torch.allclose(iou, torch.tensor([1.0, 1.0, 1.0]))


def test_zero_overlap_iou_is_zero():
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, :, :2] = 1
    logits = torch.zeros(1, 2, 4, 4)
    logits[0, 0, :, 2:] = 5.0  # predict class 0 where target is 0... wait, overlap
    logits[0, 1, :, :2] = -5.0
    # Re-craft: predict class 1 everywhere it isn't the target's class 1
    logits = torch.zeros(1, 2, 4, 4)
    logits[0, 1, :, :2] = -10.0  # don't predict class 1 where target says 1
    logits[0, 0, :, :2] = -10.0  # don't predict class 0 where target says 1 either... predictions
    # Simpler: argmax of (high-class-1 on right half) -> class 1 on right, but target's class 1 is on left
    logits = torch.full((1, 2, 4, 4), -1.0)
    logits[0, 1, :, 2:] = 10.0  # predict class 1 only on right
    logits[0, 0, :, :2] = 10.0  # predict class 0 only on left (where target is 1)
    acc = ConfusionAccumulator(num_classes=2)
    acc.update(logits, target)
    iou = per_class_iou(acc.confusion)
    assert iou[1].item() < 1e-6  # eps=1e-9 in per_class_iou makes it ~6e-11, not exactly 0


def test_macro_miou_excludes_background():
    # 3 classes; bg(0) perfect, fg(1)=0.5, fg(2)=0.0
    confusion = torch.tensor([
        [10, 0,  0],   # gt=0
        [0,  5,  5],   # gt=1
        [0,  10, 0],   # gt=2
    ], dtype=torch.float32)
    iou = per_class_iou(confusion)
    miou = macro_miou(iou, exclude_bg=True)
    # iou[1] = 5 / (5+5+0+10) = 5/20 = 0.25 ; iou[2] = 0/(0+10+10+0) = 0
    expected = (0.25 + 0.0) / 2
    assert abs(miou - expected) < 1e-6


def test_accumulator_handles_ignore_index():
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, 0, 0] = 255  # ignored
    logits = torch.zeros(1, 2, 4, 4)
    logits[0, 0, :, :] = 5.0  # predict class 0 everywhere
    acc = ConfusionAccumulator(num_classes=2)
    acc.update(logits, target, ignore_index=255)
    # 15 valid pixels, all class 0, all predicted class 0
    assert acc.confusion[0, 0].item() == 15
    assert acc.confusion.sum().item() == 15
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_eval.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement eval.py**

```python
# OBJECTIFICATION/seg/eval.py
"""Per-class IoU and macro mIoU via a streaming confusion matrix."""
import torch


class ConfusionAccumulator:
    """Streaming confusion matrix. Rows = ground truth, cols = prediction."""
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits, target, ignore_index: int = 255):
        """logits: (B, C, H, W); target: (B, H, W) long."""
        pred = logits.argmax(dim=1)
        valid = (target != ignore_index)
        t = target[valid].view(-1)
        p = pred[valid].view(-1)
        idx = t * self.num_classes + p
        binc = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion += binc.view(self.num_classes, self.num_classes).double().cpu()

    def reset(self):
        self.confusion.zero_()


def per_class_iou(confusion: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """confusion: (C, C) float. Returns (C,) IoU per class."""
    tp = confusion.diag()
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp
    return (tp + eps) / (tp + fp + fn + eps)


def macro_miou(per_class: torch.Tensor, exclude_bg: bool = True) -> float:
    """Macro-averaged mIoU. Excludes class 0 (background) by default."""
    if exclude_bg:
        return per_class[1:].mean().item()
    return per_class.mean().item()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/test_eval.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES
git add OBJECTIFICATION/seg/eval.py OBJECTIFICATION/tests/test_eval.py
git commit -m "feat(objectification): per-class IoU + macro mIoU eval"
```

---

## Task 15: Training loop

**Files:**
- Create: `OBJECTIFICATION/seg/train.py`

The training loop is integration-level — it wires together everything from Tasks 6–14. We've already unit-tested every component, so we verify the loop end-to-end with a 2-batch smoke run on a tiny synthetic dataset.

- [ ] **Step 1: Implement train.py**

```python
# OBJECTIFICATION/seg/train.py
"""OBJECTIFICATION Layer 1 training. Single-file, env-var configured —
mirrors the HAND_JOB/hand_seg/train.py pattern.

Usage (from OBJECTIFICATION/seg/):
    python train.py
Override via env vars:
    IMG_SIZE=320 BATCH=16 EPOCHS=60 LR=3e-4 WORKERS=6 RUN_TAG=v1 \
        DATA_ROOT=../shared/datasets/openimages_v7 SMOKE=0 python train.py

SMOKE=1 runs 2 batches per epoch over 1 epoch — for plumbing checks only.
"""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from OBJECTIFICATION.seg.augment import SegTransform
from OBJECTIFICATION.seg.classes import NUM_CLASSES
from OBJECTIFICATION.seg.dataset import OpenImagesSegDataset
from OBJECTIFICATION.seg.eval import ConfusionAccumulator, macro_miou, per_class_iou
from OBJECTIFICATION.seg.losses import ce_dice_loss
from OBJECTIFICATION.seg.model import ObjSegNet


# ---------------- CONFIG ----------------
IMG_SIZE  = int(os.environ.get("IMG_SIZE", 320))
BATCH     = int(os.environ.get("BATCH", 16))
EPOCHS    = int(os.environ.get("EPOCHS", 60))
LR        = float(os.environ.get("LR", 3e-4))
WD        = float(os.environ.get("WD", 5e-4))
WORKERS   = int(os.environ.get("WORKERS", 6))
PATIENCE  = int(os.environ.get("PATIENCE", 8))
RUN_TAG   = os.environ.get("RUN_TAG", "v1")
SMOKE     = bool(int(os.environ.get("SMOKE", 0)))
SEED      = 42

DATA_ROOT = Path(os.environ.get(
    "DATA_ROOT",
    str(Path(__file__).resolve().parent.parent / "shared" / "datasets" / "openimages_v7")
))
TRAIN_ROOT = DATA_ROOT / "train"
VAL_ROOT   = DATA_ROOT / "val"

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
CKPT_PATH = CKPT_DIR / f"obj_seg_{RUN_TAG}.pt"
LOG_PATH  = CKPT_DIR / f"obj_seg_{RUN_TAG}.log.json"


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Datasets
    train_tf = SegTransform(img_size=IMG_SIZE, mode="train")
    val_tf   = SegTransform(img_size=IMG_SIZE, mode="eval")
    tr_ds = OpenImagesSegDataset(TRAIN_ROOT, transform=train_tf)
    va_ds = OpenImagesSegDataset(VAL_ROOT,   transform=val_tf)
    assert len(tr_ds) > 0, f"no training images under {TRAIN_ROOT}"
    assert len(va_ds) > 0, f"no validation images under {VAL_ROOT}"

    # Class-weighted CE + weighted batch sampler
    class_freq = tr_ds.class_freq(NUM_CLASSES).astype(np.float64)
    nonzero = class_freq[class_freq > 0]
    median = float(np.median(nonzero)) if len(nonzero) else 1.0
    cw = np.ones(NUM_CLASSES, dtype=np.float32)
    for c in range(1, NUM_CLASSES):
        if class_freq[c] > 0:
            cw[c] = float(np.clip(median / class_freq[c], 0.5, 5.0))
    class_weights = torch.tensor(cw, device=device)

    sample_weights = tr_ds.sample_weights(NUM_CLASSES)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(tr_ds), replacement=True
    )

    kw = dict(num_workers=WORKERS, persistent_workers=(WORKERS > 0))
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, sampler=sampler, drop_last=True, **kw)
    va_ld = DataLoader(va_ds, batch_size=BATCH, shuffle=False, **kw)

    # Model
    model = ObjSegNet(num_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Optimizer + cosine LR with linear warmup
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total_steps = max(1, EPOCHS * len(tr_ld))
    warmup_steps = min(1000, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print(f"[{RUN_TAG}] device={device} params={n_params:,} "
          f"train={len(tr_ds)} val={len(va_ds)} batches/ep={len(tr_ld)}", flush=True)
    print(f"[{RUN_TAG}] img={IMG_SIZE} batch={BATCH} epochs={EPOCHS} lr={LR} "
          f"workers={WORKERS} smoke={SMOKE}", flush=True)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        acc = ConfusionAccumulator(num_classes=NUM_CLASSES)
        tot_loss, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce_dice_loss(logits, y, num_classes=NUM_CLASSES,
                                class_weights=class_weights)
            tot_loss += loss.item() * x.size(0); n += x.size(0)
            acc.update(logits, y)
        iou = per_class_iou(acc.confusion)
        return {"loss": tot_loss / max(1, n), "miou": macro_miou(iou),
                "per_class_iou": iou.tolist()}

    history = []
    best_miou = -1.0; no_improve = 0
    t0 = time.time()
    step = 0
    for ep in range(EPOCHS):
        model.train()
        tot_loss, n = 0.0, 0
        t_ep = time.time()
        for bi, (x, y) in enumerate(tr_ld):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce_dice_loss(model(x), y, num_classes=NUM_CLASSES,
                                class_weights=class_weights)
            loss.backward(); opt.step(); sched.step()
            tot_loss += loss.item() * x.size(0); n += x.size(0); step += 1
            if bi % 50 == 0:
                print(f"[{RUN_TAG}] ep {ep} batch {bi:4d}/{len(tr_ld)} | "
                      f"loss {tot_loss/n:.4f} | lr {opt.param_groups[0]['lr']:.2e} | "
                      f"t {time.time()-t_ep:.0f}s", flush=True)
            if SMOKE and bi >= 1:
                break
        tr_loss = tot_loss / max(1, n)

        val = evaluate(va_ld)
        improved = val["miou"] > best_miou
        if improved:
            best_miou = val["miou"]; no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "img_size": IMG_SIZE,
                "num_classes": NUM_CLASSES,
                "val_miou": best_miou,
                "epoch": ep,
            }, CKPT_PATH)
        else:
            no_improve += 1

        history.append({"epoch": ep, "tr_loss": tr_loss,
                        "val_loss": val["loss"], "val_miou": val["miou"]})
        flag = " *NEW BEST*" if improved else ""
        print(f"[{RUN_TAG}] ep {ep:2d} | tr {tr_loss:.4f} | "
              f"vl {val['loss']:.4f} | mIoU {val['miou']:.4f}{flag} | "
              f"total {time.time()-t0:.0f}s", flush=True)

        if SMOKE:
            break
        if no_improve >= PATIENCE:
            print(f"[{RUN_TAG}] early stop at epoch {ep}", flush=True); break

    with open(LOG_PATH, "w") as f:
        json.dump({
            "config": {"img_size": IMG_SIZE, "batch": BATCH, "epochs": EPOCHS,
                       "lr": LR, "wd": WD, "workers": WORKERS},
            "params": n_params, "history": history, "best_val_miou": best_miou,
        }, f, indent=2)
    print(f"[{RUN_TAG}] done. best_val_miou={best_miou:.4f}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run on the val data downloaded in Tasks 4–5**

The smoke run uses the val split for both train and val (no train data downloaded yet). It only verifies plumbing, not learning.

Run:
```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES && \
SMOKE=1 EPOCHS=1 BATCH=2 IMG_SIZE=128 WORKERS=0 \
DATA_ROOT=$(pwd)/OBJECTIFICATION/shared/datasets/openimages_v7 \
RUN_TAG=smoke \
python -c "
# Re-point val to itself for the smoke run
import os, shutil
from pathlib import Path
root = Path('OBJECTIFICATION/shared/datasets/openimages_v7')
(root/'train').mkdir(exist_ok=True)
for sub in ('images', 'masks'):
    src = root/'val'/sub; dst = root/'train'/sub
    if not dst.exists(): dst.symlink_to(src.resolve())
" && \
SMOKE=1 EPOCHS=1 BATCH=2 IMG_SIZE=128 WORKERS=0 RUN_TAG=smoke \
python -m OBJECTIFICATION.seg.train
```
Expected: prints epoch 0 with `tr ...` `vl ...` `mIoU ...`, no exceptions, writes `OBJECTIFICATION/seg/checkpoints/obj_seg_smoke.pt`.

- [ ] **Step 3: Verify the smoke checkpoint loads cleanly**

Run:
```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES && python -c "
import torch
from OBJECTIFICATION.seg.model import ObjSegNet
m = ObjSegNet(num_classes=24)
ckpt = torch.load('OBJECTIFICATION/seg/checkpoints/obj_seg_smoke.pt', map_location='cpu')
m.load_state_dict(ckpt['model_state_dict'])
print('OK epoch', ckpt['epoch'], 'mIoU', ckpt['val_miou'])"
```
Expected: `OK epoch 0 mIoU <float>`

- [ ] **Step 4: Commit**

```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES
git add OBJECTIFICATION/seg/train.py
git commit -m "feat(objectification): training loop (sampler, CE+Dice, cosine, mIoU ckpt)"
```

---

## Task 16: Top-level orchestrator

**Files:**
- Create: `OBJECTIFICATION/train_all.py`

This is a thin port of `HAND_JOB/train_all.py` adapted for OBJECTIFICATION's single training stage. Read `HAND_JOB/train_all.py` first to mirror its thermal-watchdog and subprocess pattern exactly. The only structural change is one stage instead of two.

- [ ] **Step 1: Read the HAND_JOB orchestrator to understand the pattern**

Run: `cat /Users/emgor/Documents/LEARNIN_MACHINES/HAND_JOB/train_all.py`

Mirror its structure: `subprocess.Popen` for the training stage, `powermetrics`-based thermal watchdog with `SIGSTOP`/`SIGCONT` on Trapping/Sleeping/Nominal states, JSON log of run summary.

- [ ] **Step 2: Implement train_all.py**

```python
# OBJECTIFICATION/train_all.py
"""Sequential orchestrator for OBJECTIFICATION training.

Currently a single stage (seg/train.py) but uses the same thermal-watchdog
+ subprocess pattern as HAND_JOB/train_all.py so a second stage (e.g. a
fine-tune pass) can be appended later without restructuring.

Run from OBJECTIFICATION/:
    python train_all.py
Env vars passed through to seg/train.py: IMG_SIZE, BATCH, EPOCHS, LR, WORKERS, RUN_TAG.
"""
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "seg" / "checkpoints"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = LOG_DIR / "train_all_summary.json"


# ---------------- THERMAL WATCHDOG ----------------
class ThermalWatchdog(threading.Thread):
    """Polls `powermetrics` for thermal pressure.
    SIGSTOP child on Trapping/Sleeping; SIGCONT on Nominal.
    Mirrors HAND_JOB/train_all.py.
    """
    def __init__(self, child_pid):
        super().__init__(daemon=True)
        self.child_pid = child_pid
        self.stop_event = threading.Event()
        self.paused = False

    def run(self):
        cmd = ["sudo", "powermetrics", "--samplers", "smc", "-i", "5000", "-n", "0"]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    text=True, bufsize=1)
        except Exception as e:
            print(f"[watchdog] could not start powermetrics: {e}", flush=True)
            return
        try:
            for line in proc.stdout:
                if self.stop_event.is_set():
                    break
                m = re.search(r"Thermal pressure:\s+(\w+)", line)
                if not m:
                    continue
                state = m.group(1)
                if state in ("Trapping", "Sleeping") and not self.paused:
                    print(f"[watchdog] {state} -> SIGSTOP", flush=True)
                    try: os.kill(self.child_pid, signal.SIGSTOP); self.paused = True
                    except ProcessLookupError: break
                elif state == "Nominal" and self.paused:
                    print(f"[watchdog] Nominal -> SIGCONT", flush=True)
                    try: os.kill(self.child_pid, signal.SIGCONT); self.paused = False
                    except ProcessLookupError: break
        finally:
            proc.terminate()

    def stop(self):
        self.stop_event.set()


# ---------------- STAGE RUNNER ----------------
def run_stage(name, cmd):
    print(f"\n=== [{name}] starting: {' '.join(cmd)} ===", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=os.environ.copy())
    wd = ThermalWatchdog(proc.pid); wd.start()
    try:
        rc = proc.wait()
    finally:
        wd.stop()
    dur = time.time() - t0
    print(f"=== [{name}] exit={rc} duration={dur:.0f}s ===", flush=True)
    return {"name": name, "exit": rc, "duration_s": dur}


def main():
    stages = [
        ("seg", [sys.executable, "-m", "OBJECTIFICATION.seg.train"]),
    ]
    summary = []
    for name, cmd in stages:
        r = run_stage(name, cmd)
        summary.append(r)
        if r["exit"] != 0:
            print(f"[{name}] failed -> aborting", flush=True)
            break
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"summary written to {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    # Run from repo root so the `-m OBJECTIFICATION.seg.train` import resolves
    os.chdir(ROOT.parent)
    main()
```

- [ ] **Step 3: Smoke-run the orchestrator**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && SMOKE=1 EPOCHS=1 BATCH=2 IMG_SIZE=128 WORKERS=0 RUN_TAG=orch python OBJECTIFICATION/train_all.py`
Expected: prints `=== [seg] starting ... ===`, training runs and exits 0, summary JSON appears under `OBJECTIFICATION/seg/checkpoints/train_all_summary.json`.

(The watchdog will print `could not start powermetrics` if you're not running with sudo — that's fine for the smoke run, the training stage still completes.)

- [ ] **Step 4: Commit**

```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES
git add OBJECTIFICATION/train_all.py
git commit -m "feat(objectification): top-level orchestrator with thermal watchdog"
```

---

## Task 17: Final integration check + full-suite test run

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `cd /Users/emgor/Documents/LEARNIN_MACHINES && python -m pytest OBJECTIFICATION/tests/ -v`
Expected: all tests pass (count ≈ 21).

- [ ] **Step 2: Verify the package imports cleanly**

Run:
```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES && python -c "
from OBJECTIFICATION.seg import classes, model, dataset, augment, losses, eval as eval_mod
from OBJECTIFICATION.seg.model import ObjSegNet
from OBJECTIFICATION.seg.classes import NUM_CLASSES, CLASS_NAMES
import torch
m = ObjSegNet(num_classes=NUM_CLASSES)
y = m(torch.randn(1, 3, 320, 320))
assert y.shape == (1, NUM_CLASSES, 320, 320)
print('OK end-to-end forward', y.shape, 'classes', len(CLASS_NAMES))
"
```
Expected: `OK end-to-end forward torch.Size([1, 24, 320, 320]) classes 24`

- [ ] **Step 3: Download a real training subset (optional — large)**

Run (background-friendly, may take 1–4 hours):
```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES/OBJECTIFICATION/data_pipeline && \
python download.py --split val   --max-per-class 200 && \
python prepare_masks.py --split val && \
python download.py --split train --max-per-class 3000 && \
python prepare_masks.py --split train
```
Expected: ~50K images and matching masks under `OBJECTIFICATION/shared/datasets/openimages_v7/{train,val}/`.

- [ ] **Step 4: Kick off the real training run**

Run (in foreground or via the orchestrator; expect 6–10 hours on M-series MPS):
```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES && \
RUN_TAG=v1 EPOCHS=60 BATCH=16 IMG_SIZE=320 WORKERS=6 \
python OBJECTIFICATION/train_all.py
```
Expected: per-epoch logs, `best.pt` checkpointed when mIoU improves, `train_all_summary.json` on completion.

- [ ] **Step 5: Confirm success criteria from the spec**

Open `OBJECTIFICATION/seg/checkpoints/obj_seg_v1.log.json` and check:
- `best_val_miou` ≥ 0.55 (target from spec §11)
- last entry's per-class IoU has every class ≥ 0.30 (no class collapse)

If either is missed, the spec marks this as expected — tune from here (more epochs, more data per class, augmentation tweaks) but that's iterative work, not a plan failure.

- [ ] **Step 6: Final commit (training results metadata, no model weights)**

```bash
cd /Users/emgor/Documents/LEARNIN_MACHINES
# Note: checkpoints/ is gitignored — only the JSON log gets committed
git add OBJECTIFICATION/seg/checkpoints/obj_seg_v1.log.json 2>/dev/null || true
git diff --cached --quiet && echo "no log changes to commit" || \
  git commit -m "chore(objectification): record v1 training run metrics"
```

---

## Done

The model checkpoint at `OBJECTIFICATION/seg/checkpoints/obj_seg_v1.pt` is the deliverable for this plan. Live-app integration (renderer + OSC + cascade gating) is a separate plan.
