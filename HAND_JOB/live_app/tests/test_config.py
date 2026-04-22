import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_config_imports():
    from live_app import config
    assert config.OSC_PORT == 9000
    assert config.CONTOUR_POINTS == 40
    assert len(config.GESTURE_COLORS_BGR) == 18
    for color in config.GESTURE_COLORS_BGR:
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)

def test_checkpoint_paths_exist():
    from live_app import config
    assert config.SEG_CKPT.exists(), f"Missing: {config.SEG_CKPT}"
    assert config.GESTURE_CKPT.exists(), f"Missing: {config.GESTURE_CKPT}"
