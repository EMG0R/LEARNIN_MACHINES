"""Source-of-truth class merge groups.

class_id 0 is background. class_ids 1..23 are foreground classes.
MERGE_GROUPS maps class_id -> list of OpenImages V7 English label names.
build_class_map.py converts these English names to OI MIDs via the
official class-descriptions CSV.
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
    "glasses",      # 12
    "headphones",   # 13
    "chair",        # 14
    "couch",        # 15
    "table",        # 16
    "lamp",         # 17
    "book",         # 18
    "clock",        # 19
    "bag",          # 20
    "guitar",       # 21
    "trumpet",      # 22
    "piano",        # 23
]

NUM_CLASSES = len(CLASS_NAMES)  # 24

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
    12: ["Glasses", "Sunglasses"],
    13: ["Headphones"],
    14: ["Chair"],
    15: ["Couch"],
    16: ["Coffee table", "Kitchen & dining room table", "Desk"],
    17: ["Lamp"],
    18: ["Book"],
    19: ["Clock"],
    20: ["Handbag", "Backpack"],
    21: ["Guitar"],
    22: ["Trumpet"],
    23: ["Piano"],
}
