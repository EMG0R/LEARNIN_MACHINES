"""
CelebAMask-HQ → 5-class segmentation.

CelebAMask-HQ provides per-part binary masks in CelebAMask-HQ-mask-anno/{0..14}/.
File name pattern: {id:05d}_{part}.png  (e.g. 00042_l_eye.png)

Sample ids 0-1999 live in subdir 0/, 2000-3999 in subdir 1/, etc.

Merge rule: tighter parts override broader ones. Priority: eye_L=1, eye_R=2, mouth=3 > skin=4 > background=0.
"""
from pathlib import Path
import numpy as np
from PIL import Image

PART_TO_CLASS = {
    "l_eye": 1,
    "r_eye": 2,
    "u_lip": 3, "l_lip": 3, "mouth": 3,
    "skin":  4,
}

IMG_ROOT_SUBDIR   = "CelebA-HQ-img"
MASK_ROOT_SUBDIR  = "CelebAMask-HQ-mask-anno"
MASK_SIZE         = 512   # native CelebAMask-HQ mask size


def _mask_subdir(sample_id: int) -> str:
    return str(sample_id // 2000)


def _load_part(root: Path, sample_id: int, part: str) -> np.ndarray | None:
    p = root / MASK_ROOT_SUBDIR / _mask_subdir(sample_id) / f"{sample_id:05d}_{part}.png"
    if not p.exists():
        return None
    return np.array(Image.open(p).convert("L"))


def merge_parts(root: Path, sample_id: int) -> np.ndarray:
    """Return a single H×W uint8 array with class ids 0..4."""
    canvas = None
    # Layer broadest → tightest so tight parts overwrite broad ones
    layer_order = ["skin", "u_lip", "l_lip", "mouth", "l_eye", "r_eye"]
    for part in layer_order:
        m = _load_part(root, sample_id, part)
        if m is None:
            continue
        if canvas is None:
            canvas = np.zeros(m.shape, dtype=np.uint8)
        canvas[m > 127] = PART_TO_CLASS[part]
    if canvas is None:
        canvas = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.uint8)
    return canvas


def list_samples(root: Path, max_id: int = 30000) -> list[int]:
    """Yield sample ids that have at least one part mask."""
    out = []
    for sid in range(max_id):
        sub = root / MASK_ROOT_SUBDIR / _mask_subdir(sid)
        if (sub / f"{sid:05d}_skin.png").exists():
            out.append(sid)
    return out


def image_path(root: Path, sample_id: int) -> Path:
    return root / IMG_ROOT_SUBDIR / f"{sample_id}.jpg"
