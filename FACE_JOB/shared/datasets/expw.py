"""
ExpW loader.

Label file format (label.lst):
    image_name face_id top left right bottom conf expression_label

Native ExpW expression labels:
    0 anger
    1 disgust
    2 fear
    3 happy
    4 sad
    5 surprise
    6 neutral
"""
from pathlib import Path

from FACE_JOB.shared.datasets.ferplus import CLASS_NAMES, CLASS_TO_IDX

EXPW_LABEL_TO_CLASS = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


def load_expw(root: Path) -> list[tuple[Path, int]]:
    label_file = root / "label.lst"
    img_dir = root / "image"
    samples: list[tuple[Path, int]] = []
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            name = parts[0]
            lab = int(parts[7])
            if lab not in EXPW_LABEL_TO_CLASS:
                continue
            cls = EXPW_LABEL_TO_CLASS[lab]
            samples.append((img_dir / name, CLASS_TO_IDX[cls]))
    return samples
