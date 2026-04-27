"""Combine per-instance OpenImages mask PNGs into a single integer mask
per image (uint8, pixel value = merged class ID, 0 = background).
"""
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image

from OBJECTIFICATION.seg.classes import CLASS_NAMES


# Painting priority: bigger, more "background-like" objects painted FIRST,
# subjects painted LAST so they win on overlap. Person is ALWAYS top priority
# because v1 collapsed when person pixels got overwritten by co-occurring
# objects (vehicle, animal, couch). Lower number = painted earlier (loses on
# overlap). Higher number = painted later (wins on overlap). Class IDs are
# resolved by name at module-load time so renumbering classes.py is safe.
_PAINT_PRIORITY_BY_NAME = {
    "plant":      1,   # background foliage
    "couch":      1,   # large furniture
    "vehicle":    2,
    "device":     2,
    "skateboard": 5,
    "animal":     5,
    "person":     10,  # paint last, always wins on overlap
}
_NAME_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}
PAINT_PRIORITY = {
    _NAME_TO_ID[name]: pri
    for name, pri in _PAINT_PRIORITY_BY_NAME.items()
    if name in _NAME_TO_ID
}


def combine_instance_masks(
    masks: Iterable[Tuple[Path, str]],
    mid_to_class: dict,
    image_size: Tuple[int, int],
) -> Image.Image:
    """Combine per-instance binary masks into a single integer mask.

    masks: iterable of (mask_path, oi_mid) pairs
    mid_to_class: dict OI MID -> merged class ID
    image_size: (W, H) of the parent image; instance masks resize to this.
    Returns: PIL 'L' image, pixel value = class id (0 = background).

    Painting order is class-priority based (see PAINT_PRIORITY). Person (cls=1)
    is always painted last so it wins overlaps with co-occurring objects.
    """
    W, H = image_size
    out = np.zeros((H, W), dtype=np.uint8)
    # Resolve and sort by priority (low priority first; high last = wins on top)
    resolved = []
    for mask_path, mid in masks:
        cls = mid_to_class.get(mid)
        if cls is None:
            continue
        resolved.append((PAINT_PRIORITY.get(cls, 3), mask_path, cls))
    resolved.sort(key=lambda r: r[0])
    for _, mask_path, cls in resolved:
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
    for image_id, masks in instance_index.items():
        img_path = split_root / "images" / f"{image_id}.jpg"
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            size = im.size  # (W, H)
        merged = combine_instance_masks(masks, mid_to_class, size)
        merged.save(out_dir / f"{image_id}.png")
    print(f"wrote {len(list(out_dir.glob('*.png')))} masks to {out_dir}")


def main():
    import argparse
    import json
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
