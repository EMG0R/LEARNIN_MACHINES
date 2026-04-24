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
