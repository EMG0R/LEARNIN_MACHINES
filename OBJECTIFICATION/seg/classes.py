"""Source-of-truth class merge groups.

class_id 0 is background. class_ids 1..17 are foreground classes.
MERGE_GROUPS maps class_id -> list of OpenImages V7 English label names.
build_class_map.py converts these English names to OI MIDs via the
official class-descriptions CSV.

v2 dropped 6 original classes that have ZERO segmentation masks in
OpenImages V7: glasses, headphones, chair, table, lamp, trumpet.
Those classes can't be trained from this data source — verified by
counting `LabelName` matches in the seg-annotations CSV. Down from 23
to 17 foreground classes (24 -> 18 output channels).
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
    "couch",        # 12
    "book",         # 13
    "clock",        # 14
    "bag",          # 15
    "guitar",       # 16
    "piano",        # 17
]

NUM_CLASSES = len(CLASS_NAMES)  # 18

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
    12: ["Couch"],
    13: ["Book"],
    14: ["Clock"],
    15: ["Handbag", "Backpack"],
    16: ["Guitar"],
    17: ["Piano"],
}
