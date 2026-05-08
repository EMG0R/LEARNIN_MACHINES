"""Source-of-truth class merge groups.

class_id 0 is background. class_ids 1..14 are foreground classes.
MERGE_GROUPS maps class_id -> list of OpenImages V7 English label names.
build_class_map.py converts these English names to OI MIDs via the
official class-descriptions CSV.

History:
  v1 -> 23 fg classes (24 ch). 6 had ZERO seg masks in OI V7
        (glasses, headphones, chair, table, lamp, trumpet).
  v2 -> 17 fg classes (18 ch). Removed the 6 untrainable.
  v4 -> 14 fg classes (15 ch). Removed 3 user-requested:
        clock (977), spork (2339), piano (1211) — at OI ceiling and
        not high enough priority for the art install to keep.
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
    "bowl",         # 9
    "footwear",     # 10
    "couch",        # 11
    "book",         # 12
    "bag",          # 13
    "guitar",       # 14
]

NUM_CLASSES = len(CLASS_NAMES)  # 15

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
    9:  ["Bowl", "Plate"],
    10: ["Footwear", "Boot", "Sandal", "High heels", "Sneakers"],
    11: ["Couch"],
    12: ["Book"],
    13: ["Handbag", "Backpack"],
    14: ["Guitar"],
}
