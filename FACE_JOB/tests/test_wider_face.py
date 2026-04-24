from pathlib import Path
import tempfile, textwrap

from FACE_JOB.shared.datasets.wider_face import parse_wider_gt, filter_rows


def test_parse_wider_gt_basic(tmp_path):
    gt = tmp_path / "wider_face_train_bbx_gt.txt"
    pixels_row = " ".join(["0"] * (48*48))
    gt.write_text(textwrap.dedent("""\
    0--Parade/0_Parade_marchingband_1_849.jpg
    2
    449 330 122 149 0 0 0 0 0 0
    361 98  263 339 0 0 0 0 0 0
    0--Parade/0_Parade_marchingband_1_799.jpg
    1
    50  50  30  30 0 0 0 0 0 0
    """))
    rows = parse_wider_gt(str(gt))
    assert len(rows) == 2
    assert rows[0][0].endswith("849.jpg")
    assert rows[0][1] == [(449, 330, 122, 149), (361, 98, 263, 339)]
    assert rows[1][1] == [(50, 50, 30, 30)]


def test_filter_rows_drops_tiny_and_crowds():
    rows = [
        ("a.jpg", [(0, 0, 50, 50), (0, 0, 60, 60)]),              # keep
        ("b.jpg", [(0, 0, 20, 20)]),                              # drop: too small
        ("c.jpg", [(0, 0, 50, 50)] * 9),                          # drop: crowd
        ("d.jpg", [(0, 0, 50, 50), (0, 0, 15, 15)]),              # drop: has a tiny face
    ]
    kept = filter_rows(rows, min_side=40, max_faces=8)
    assert len(kept) == 1
    assert kept[0][0] == "a.jpg"
