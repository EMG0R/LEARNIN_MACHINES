import textwrap
from pathlib import Path
from FACE_JOB.shared.datasets.rafdb import load_rafdb, CLASS_NAMES


def test_load_rafdb(tmp_path):
    labels = tmp_path / "list_patition_label.txt"
    labels.write_text(textwrap.dedent("""\
    train_00001_aligned.jpg 4
    train_00002_aligned.jpg 7
    test_00001_aligned.jpg 1
    """))
    (tmp_path / "Image" / "aligned").mkdir(parents=True)
    for n in ("train_00001_aligned.jpg", "train_00002_aligned.jpg", "test_00001_aligned.jpg"):
        (tmp_path / "Image" / "aligned" / n).write_bytes(b"")

    samples = load_rafdb(tmp_path, split="train")
    assert len(samples) == 2
    paths = [p.name for p, _ in samples]
    assert "train_00001_aligned.jpg" in paths
    # 4 → happiness → "happy"
    idx = paths.index("train_00001_aligned.jpg")
    assert CLASS_NAMES[samples[idx][1]] == "happy"
    # 7 → neutral
    idx = paths.index("train_00002_aligned.jpg")
    assert CLASS_NAMES[samples[idx][1]] == "neutral"
