"""
FER+ loader. Produces (image_array_48x48_uint8, class_idx) tuples.

FER2013 CSV:  emotion, pixels, Usage
FER+ CSV:     Usage, Image_name, neutral, happiness, surprise, sadness,
              anger, disgust, fear, contempt, unknown, NF

Majority vote over the 10 rater-count columns.
Drop samples where majority class is contempt/unknown/NF.

Align to 7-class intersection: happy, sad, neutral, surprise, anger, fear, disgust
"""
import csv
import numpy as np
from pathlib import Path


CLASS_NAMES = ["happy", "sad", "neutral", "surprise", "anger", "fear", "disgust"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# Column name in fer2013new.csv → our canonical class name
FERPLUS_COL_TO_CLASS = {
    "neutral":   "neutral",
    "happiness": "happy",
    "surprise":  "surprise",
    "sadness":   "sad",
    "anger":     "anger",
    "disgust":   "disgust",
    "fear":      "fear",
    # Dropped: contempt, unknown, NF
}


def load_ferplus(fer2013_csv: Path, ferplus_csv: Path) -> list[tuple[np.ndarray, int]]:
    # Read FER2013 pixels (indexed by row order, which matches FER+ image name order)
    fer_rows = []
    with open(fer2013_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
            fer_rows.append(pixels)

    # Read FER+ multi-rater labels
    samples: list[tuple[np.ndarray, int]] = []
    with open(ferplus_csv, newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            counts = {
                col: int(row.get(col, "0") or "0")
                for col in FERPLUS_COL_TO_CLASS
            }
            if not counts or max(counts.values()) == 0:
                continue
            winner_col = max(counts, key=counts.get)
            cls = FERPLUS_COL_TO_CLASS[winner_col]
            if i >= len(fer_rows):
                continue
            samples.append((fer_rows[i], CLASS_TO_IDX[cls]))
    return samples
