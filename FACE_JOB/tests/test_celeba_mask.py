import numpy as np
from PIL import Image
from FACE_JOB.shared.datasets.celeba_mask import merge_parts, PART_TO_CLASS


def test_merge_parts_produces_5_class_mask(tmp_path):
    sample_id = 42
    mask_dir = tmp_path / "CelebAMask-HQ-mask-anno" / "0"
    mask_dir.mkdir(parents=True)

    def write_mask(name, box):
        img = np.zeros((64, 64), dtype=np.uint8)
        x0, y0, x1, y1 = box
        img[y0:y1, x0:x1] = 255
        Image.fromarray(img).save(mask_dir / f"{sample_id:05d}_{name}.png")

    write_mask("l_eye", (10, 10, 20, 20))
    write_mask("r_eye", (30, 10, 40, 20))
    write_mask("mouth", (20, 40, 40, 50))
    write_mask("skin",  (5,  5,  60, 60))

    merged = merge_parts(tmp_path, sample_id)
    assert merged.shape == (64, 64)
    assert set(np.unique(merged)) <= {0, 1, 2, 3, 4}
    assert merged[15, 15] == PART_TO_CLASS["l_eye"]
    assert merged[15, 35] == PART_TO_CLASS["r_eye"]
    assert merged[45, 30] == PART_TO_CLASS["mouth"]
    # eye pixels override skin (tight parts overwrite broad ones)
    assert merged[15, 15] != PART_TO_CLASS["skin"]
