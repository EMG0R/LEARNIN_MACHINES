import textwrap
from pathlib import Path
from FACE_JOB.shared.datasets.expw import load_expw, CLASS_NAMES


def test_load_expw(tmp_path):
    labels = tmp_path / "label.lst"
    labels.write_text(textwrap.dedent("""\
    img001.jpg 0 10 10 100 100 0.99 3
    img002.jpg 0 5 5 80 80 0.95 6
    img003.jpg 0 0 0 50 50 0.80 7
    """))
    img_dir = tmp_path / "image"
    img_dir.mkdir()
    for n in ("img001.jpg", "img002.jpg", "img003.jpg"):
        (img_dir / n).write_bytes(b"")

    samples = load_expw(tmp_path)
    assert len(samples) == 2
    # ExpW native: 0 anger, 1 disgust, 2 fear, 3 happy, 4 sad, 5 surprise, 6 neutral
    idx = {s[0].name: s[1] for s in samples}
    assert CLASS_NAMES[idx["img001.jpg"]] == "happy"
    assert CLASS_NAMES[idx["img002.jpg"]] == "neutral"
    # label 7 is out-of-range → skipped
    assert "img003.jpg" not in idx
