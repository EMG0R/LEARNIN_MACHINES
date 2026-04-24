import textwrap, numpy as np
from FACE_JOB.shared.datasets.ferplus import load_ferplus, CLASS_NAMES


def test_load_ferplus(tmp_path):
    fer = tmp_path / "fer2013.csv"
    pixels_row = " ".join(["0"] * (48*48))
    fer.write_text(
        "emotion,pixels,Usage\n"
        f"3,\"{pixels_row}\",Training\n"
        f"0,\"{pixels_row}\",Training\n"
    )
    ferplus = tmp_path / "fer2013new.csv"
    # FER+ columns: Usage, Image name, neutral, happiness, surprise, sadness,
    #               anger, disgust, fear, contempt, unknown, NF
    ferplus.write_text(
        "Usage,Image name,neutral,happiness,surprise,sadness,anger,disgust,fear,contempt,unknown,NF\n"
        "Training,fer0000000.png,0,10,0,0,0,0,0,0,0,0\n"  # happy winner
        "Training,fer0000001.png,10,0,0,0,0,0,0,0,0,0\n"  # neutral winner
    )
    samples = load_ferplus(fer, ferplus)
    assert len(samples) == 2
    img, label = samples[0]
    assert img.shape == (48, 48)
    assert img.dtype == np.uint8
    assert CLASS_NAMES[label] == "happy"
    assert CLASS_NAMES[samples[1][1]] == "neutral"
