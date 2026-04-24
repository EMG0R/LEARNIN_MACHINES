"""
WIDER FACE parser + webcam-realistic filter.

Annotation format:
    image_path
    num_faces
    x y w h blur expr illum invalid occl pose   (repeated num_faces times)

If num_faces == 0 the file contains one dummy line with zeros; we treat those as empty.
"""
from pathlib import Path


def parse_wider_gt(gt_path: str) -> list[tuple[str, list[tuple[int, int, int, int]]]]:
    rows: list[tuple[str, list[tuple[int, int, int, int]]]] = []
    with open(gt_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    i = 0
    while i < len(lines):
        img = lines[i]; i += 1
        n = int(lines[i]); i += 1
        bboxes: list[tuple[int, int, int, int]] = []
        count = max(n, 1)  # zero-face entries still have 1 dummy line
        for _ in range(count):
            parts = lines[i].split(); i += 1
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            if n > 0 and w > 0 and h > 0:
                bboxes.append((x, y, w, h))
        rows.append((img, bboxes))
    return rows


def filter_rows(
    rows: list[tuple[str, list[tuple[int, int, int, int]]]],
    min_side: int = 40,
    max_faces: int = 8,
) -> list[tuple[str, list[tuple[int, int, int, int]]]]:
    out = []
    for path, bboxes in rows:
        if not bboxes or len(bboxes) > max_faces:
            continue
        if any(min(w, h) < min_side for _, _, w, h in bboxes):
            continue
        out.append((path, bboxes))
    return out


def load_filtered(root: Path, split: str = "train") -> list[tuple[Path, list[tuple[int, int, int, int]]]]:
    """Load + filter WIDER FACE. Returns absolute image paths."""
    gt = root / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
    img_root = root / f"WIDER_{split}" / "images"
    rows = parse_wider_gt(str(gt))
    rows = filter_rows(rows)
    return [(img_root / p, b) for p, b in rows]
