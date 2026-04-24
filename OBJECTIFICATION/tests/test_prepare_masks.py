import numpy as np
from PIL import Image
from OBJECTIFICATION.data_pipeline.prepare_masks import combine_instance_masks

def test_merges_into_single_integer_mask(tiny_instance_masks, fake_class_map):
    out = combine_instance_masks(tiny_instance_masks, fake_class_map, image_size=(32, 32))
    arr = np.array(out)
    assert arr.shape == (32, 32)
    assert arr.dtype == np.uint8
    # top-left should be person (class 1)
    assert arr[5, 5] == 1
    # bottom-right should be chair (class 14)
    assert arr[20, 20] == 14
    # untouched corners should be background (0)
    assert arr[31, 0] == 0
    assert arr[0, 31] == 0

def test_unknown_mid_is_skipped(tmp_path, fake_class_map):
    arr = np.full((32, 32), 255, dtype=np.uint8)
    p = tmp_path / "u.png"
    Image.fromarray(arr).save(p)
    out = combine_instance_masks([(p, "/m/UNKNOWN")], fake_class_map, image_size=(32, 32))
    assert np.array(out).max() == 0
