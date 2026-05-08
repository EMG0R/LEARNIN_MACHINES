"""Convert COCO 2017 polygon segmentations to per-image integer masks
in our 14-class scheme.

Mirrors prepare_masks.py for the OI side, but rasterizes COCO's polygon
+ RLE annotations via pycocotools instead of merging per-instance PNGs.

Painting priority follows OI's PAINT_PRIORITY (person paints last).

Usage:
    python -m OBJECTIFICATION.data_pipeline.coco_to_masks --split val
    python -m OBJECTIFICATION.data_pipeline.coco_to_masks --split train

Outputs land at COCO_DATA_ROOT/{split}/masks/{file_id}.png as uint8.
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from OBJECTIFICATION.data_pipeline.coco_download import COCO_TO_OURS, DATA_ROOT
from OBJECTIFICATION.data_pipeline.prepare_masks import PAINT_PRIORITY


def _polygons_to_mask(segm, h, w):
    """COCO polygon segmentation -> binary mask (h, w) uint8.

    segm is either:
      - list of polygons (each a flat [x1,y1,x2,y2,...] list)
      - dict (RLE) — handled with mask.frPyObjects + mask.decode
    """
    from pycocotools import mask as cocomask
    if isinstance(segm, list):
        rles = cocomask.frPyObjects(segm, h, w)
        rle = cocomask.merge(rles)
    elif isinstance(segm, dict):
        if isinstance(segm.get("counts"), list):
            rle = cocomask.frPyObjects(segm, h, w)
        else:
            rle = segm
    else:
        return np.zeros((h, w), dtype=np.uint8)
    return cocomask.decode(rle).astype(np.uint8)


def process_split(split: str):
    annot_path = DATA_ROOT / "annotations" / f"instances_{split}2017.json"
    img_dir   = DATA_ROOT / split / "images"
    out_dir   = DATA_ROOT / split / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  loading {annot_path} ...")
    with open(annot_path) as f:
        data = json.load(f)
    img_meta = {im["id"]: im for im in data["images"]}

    # Group annotations by image_id, keep only target categories
    by_image = defaultdict(list)
    for ann in data["annotations"]:
        cat = COCO_TO_OURS.get(ann["category_id"])
        if cat is None:
            continue
        by_image[ann["image_id"]].append(ann)

    print(f"  {len(by_image)} images contain target classes; rasterizing ...")
    n = 0
    for image_id, anns in by_image.items():
        meta = img_meta[image_id]
        fn = meta["file_name"]
        if not (img_dir / fn).exists():
            continue
        H, W = meta["height"], meta["width"]
        out = np.zeros((H, W), dtype=np.uint8)

        # Sort by paint priority (low first, person last)
        anns_sorted = sorted(
            anns,
            key=lambda a: PAINT_PRIORITY.get(COCO_TO_OURS[a["category_id"]], 3),
        )
        for ann in anns_sorted:
            cls = COCO_TO_OURS[ann["category_id"]]
            try:
                m = _polygons_to_mask(ann["segmentation"], H, W).astype(bool)
                out[m] = cls
            except Exception as e:
                continue

        Image.fromarray(out, mode="L").save(out_dir / f"{Path(fn).stem}.png")
        n += 1
        if n % 5000 == 0:
            print(f"    {n}/{len(by_image)}", flush=True)
    print(f"  wrote {n} merged masks to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    args = ap.parse_args()
    process_split(args.split)


if __name__ == "__main__":
    main()
