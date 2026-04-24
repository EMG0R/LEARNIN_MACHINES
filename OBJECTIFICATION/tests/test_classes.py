from OBJECTIFICATION.seg.classes import MERGE_GROUPS, CLASS_NAMES, NUM_CLASSES

def test_23_foreground_classes():
    assert NUM_CLASSES == 24  # 23 + background
    assert CLASS_NAMES[0] == "background"
    assert len(CLASS_NAMES) == 24

def test_every_class_has_at_least_one_oi_label():
    for class_id, oi_labels in MERGE_GROUPS.items():
        assert len(oi_labels) >= 1, f"class {class_id} has no OI labels"

def test_no_duplicate_oi_labels_across_classes():
    seen = set()
    for labels in MERGE_GROUPS.values():
        for label in labels:
            assert label not in seen, f"label '{label}' appears in multiple classes"
            seen.add(label)

def test_class_names_match_merge_groups_keys():
    for class_id in range(1, 24):
        assert class_id in MERGE_GROUPS
