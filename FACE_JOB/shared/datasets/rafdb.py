"""
RAF-DB basic (7-class) loader.

Native label order (1-indexed in list_patition_label.txt):
    1 Surprise
    2 Fear
    3 Disgust
    4 Happiness
    5 Sadness
    6 Anger
    7 Neutral

Maps to our canonical order (same 7 classes as FER+).
"""
from pathlib import Path

from FACE_JOB.shared.datasets.ferplus import CLASS_NAMES, CLASS_TO_IDX

RAFDB_LABEL_TO_CLASS = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happy",
    5: "sad",
    6: "anger",
    7: "neutral",
}


def load_rafdb(root: Path, split: str = "train") -> list[tuple[Path, int]]:
    label_file = root / "list_patition_label.txt"
    aligned_dir = root / "Image" / "aligned"
    samples: list[tuple[Path, int]] = []
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            name, lab = parts[0], int(parts[1])
            prefix = "train_" if split == "train" else "test_"
            if not name.startswith(prefix):
                continue
            img_path = aligned_dir / name
            cls = RAFDB_LABEL_TO_CLASS[lab]
            samples.append((img_path, CLASS_TO_IDX[cls]))
    return samples
